import dataclasses
import math
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.distributed.device_mesh import init_device_mesh
from tqdm.auto import tqdm
from typing_extensions import override

from gpatch_v4.configs.config import T2iDpoConfig
from gpatch_v4.extended_pipeline import ExtendPipelineFactory
from gpatch_v4.trainer.base_trainer import BaseTrainer
from gpatch_v4.utils import (
    TrainReporterSingleton,
    dataclass_from_args,
    init_train_reporter_singleton,
    logging_rank0,
    to_device,
)


class LoggerAdaptor:
    def info(self, *args):
        logging_rank0(*args)


def build_device_mesh():
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(dist.get_world_size() // 8, 8), mesh_dim_names=("replicate", "shard")
    )
    return device_mesh


def all_gather_loss_or_metric(output):
    output = output.view([1])
    output_list = [torch.empty_like(output) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(output_list, output.detach())
    return torch.cat(output_list, dim=0)


class BaseDpoTrainer(BaseTrainer):
    config_cls = None

    def __init__(self, args):
        args = dataclass_from_args(args, self.config_cls)
        return super().__init__(args)

    def init(self):
        self.init_distributed()
        self.init_log()

        if self.args.training.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.training.scale_lr:
            self.args.optimizer.lr = self.args.optimizer.lr * self.args.training.gradient_accumulation_steps * self.args.training.train_batch_size * dist.get_world_size(
            )
        dist.barrier()
        self.build_model_and_optimizer()

    @override
    def init_distributed(self):
        assert torch.cuda.is_available()
        dist.init_process_group("nccl")
        device = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device)
        self.device_mesh = build_device_mesh()

    def init_log(self):
        self.logger = LoggerAdaptor()
        if dist.get_rank() == 0:
            init_train_reporter_singleton(self.args.report, self.args)

    def forward_backward_step(self, batch):
        """Generic DPO forward_backward_step.

        Different models implement different ``prepare_batch`` and
        ``model_forward``.
        """

        train_loss = self.train_loss
        implicit_acc_accumulated = self.implicit_acc_accumulated
        unet = self.unet
        ref_unet = self.ref_unet
        args = self.args

        target, prepared_batch = self.prepare_batch(batch)

        #### END PREP BATCH ####
        with torch.amp.autocast("cuda", enabled=True, dtype=self.weight_dtype):
            model_pred = self.model_forward(unet, prepared_batch)
            #### START LOSS COMPUTATION ####
            if args.training.train_method == 'sft':  # SFT, casting for F.mse_loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            elif args.training.train_method == 'dpo':
                # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                # losses are both 2 * LBS
                # 1st half of tensors is preferred (y_w), second half is unpreferred
                assert model_pred.ndim == target.ndim
                dim_to_mean = list(range(1, model_pred.ndim))
                model_losses = (model_pred - target).pow(2).mean(dim=dim_to_mean)
                model_losses_w, model_losses_l = model_losses.chunk(2)
                # below for logging purposes
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())

                model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)

                with torch.no_grad():  # Get the reference policy (unet) prediction
                    ref_pred = self.model_forward(self.ref_unet, prepared_batch).detach()
                    ref_losses = (ref_pred - target).pow(2).mean(dim=dim_to_mean)
                    ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_losses.mean()

                scale_term = -0.5 * args.dpo.beta_dpo
                inside_term = scale_term * (model_diff - ref_diff)
                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                loss = -1 * F.logsigmoid(inside_term).mean()
            #### END LOSS COMPUTATION ###

        avg_loss = all_gather_loss_or_metric(loss).mean()
        train_loss += avg_loss.item() / args.training.gradient_accumulation_steps

        # collect metrics
        metrics = [("train_loss", train_loss)]
        if args.training.train_method == 'dpo':
            avg_model_mse = all_gather_loss_or_metric(raw_model_loss).mean().item()
            avg_ref_mse = all_gather_loss_or_metric(raw_ref_loss).mean().item()
            avg_acc = all_gather_loss_or_metric(implicit_acc).mean().item()
            implicit_acc_accumulated += avg_acc / args.training.gradient_accumulation_steps

            metrics.append(("model_mse_unaccumulated", avg_model_mse))
            metrics.append(("ref_mse_unaccumulated", avg_ref_mse))
            metrics.append(("implicit_acc_accumulated", implicit_acc_accumulated))

        # Backpropagate
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        self.implicit_acc_accumulated = implicit_acc_accumulated
        self.train_loss = train_loss

        return metrics

    def optimize_step(self):
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.unet.parameters(), self.args.optimizer.max_grad_norm
        )
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return total_norm

    def reset_metrics(self):
        self.train_loss = 0.0
        self.implicit_acc_accumulated = 0.0

    @override
    def train_loop(self):

        args = self.args
        global_step = self.global_step
        unet = self.unet
        logger = self.logger
        train_dataloader = self.build_train_valid_test_data_iter()

        start_global_step = global_step
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.training.gradient_accumulation_steps
        )
        if args.training.max_train_steps is None:
            assert False
        # Afterwards we recalculate our number of training epochs
        args.training.num_train_epochs = math.ceil(
            args.training.max_train_steps / num_update_steps_per_epoch
        )
        resume_global_step = global_step * args.training.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (
            num_update_steps_per_epoch * args.training.gradient_accumulation_steps
        )

        total_batch_size = args.training.train_batch_size * dist.get_world_size(
        ) * args.training.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.training.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.training.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(f"  Gradient Accumulation steps = {args.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.training.max_train_steps}")

        # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
        progress_bar = tqdm(
            range(global_step, args.training.max_train_steps), disable=dist.get_rank()
        )
        progress_bar.set_description("Steps")

        #### START MAIN TRAINING LOOP #####
        step_index = 0
        self.reset_metrics()
        for epoch in range(first_epoch, args.training.num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if epoch == first_epoch and step < resume_step and (
                    not args.training.hard_skip_resume
                ):
                    if step % args.training.gradient_accumulation_steps == 0:
                        print(f"Dummy processing step {step}, will start training at {resume_step}")
                    continue
                step_index = step_index + 1

                batch = to_device(batch, "cuda", non_blocking=True)
                metrics = self.forward_backward_step(batch)
                # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
                if (
                    step_index + 1
                ) % args.training.gradient_accumulation_steps == 0:  #accelerator.sync_gradients:
                    total_norm = self.optimize_step()
                    lr = self.optimizer.param_groups[0]['lr']
                    progress_bar.update(1)
                    log_metrics = [("global_step", global_step),
                                   ("lr", lr)] + metrics + [("grad_norm", total_norm.item())]
                    if dist.get_rank() == 0:
                        TrainReporterSingleton.log_and_report(
                            {k: v
                             for (k, v) in log_metrics}, global_step, log_prefix="dpo"
                        )
                    log_str = ";".join([f"{k}:{v}" for (k, v) in log_metrics])
                    logger.info(log_str)
                    self.reset_metrics()
                    global_step += 1
                    self.global_step = global_step
                    if global_step % args.training.save_interval == 0:
                        self.save_ckpt()
                if global_step >= args.training.max_train_steps:
                    break

    def finalize(self):
        if dist.get_rank() == 0:
            TrainReporterSingleton.finish()


class T2iDpoTrainer(BaseDpoTrainer):
    config_cls = T2iDpoConfig

    @override
    def build_model_and_optimizer(self):
        """Build pipeline; pipeline builds the model and optimizer."""
        self.extended_pipeline = ExtendPipelineFactory.get_pipeline(self.args)
        self.global_step = self.extended_pipeline.setup_pipeline()
        self.extended_pipeline.setup_scheduler_and_timesteps()
        self.weight_dtype = torch.bfloat16

    @property
    def optimizer(self):
        return self.extended_pipeline.model.optimizer

    @property
    def lr_scheduler(self):
        return self.extended_pipeline.model.lr_scheduler

    @property
    def unet(self):
        return self.extended_pipeline.model.model

    @property
    def ref_unet(self):
        return self.extended_pipeline.model.ref_model

    @property
    def vae(self):
        return self.extended_pipeline.vae

    @property
    def grad_scaler(self):
        return None

    @property
    def image_processor(self):
        return self.extended_pipeline.image_processor

    def repeat_tensors(self, batch):
        """Repeat tensors for DPO."""
        for k, v in batch.items():
            if k != "text_ids":
                repeat = [1] * len(v.shape)
                repeat[0] = 2
                batch[k] = v.repeat(*repeat)
        return batch

    #TODO(hessianliu): move this to pipeline
    def prepare_batch(self, batch, promt_key="caption", pixel_value_key="pixel_values"):
        """Batch with pixel_values and caption as keys."""
        prepared_batch = {}
        # print(f"{batch}", flush=True)
        # print(f"uncond_feats {self.extended_pipeline.uncond_feats}", flush=True)
        # 1、encode text
        prompt = batch[promt_key]
        encoded_text_cond = self.extended_pipeline.encode_prompt(prompt)
        encoded_text_cond = self.repeat_tensors(encoded_text_cond)
        prepared_batch.update(encoded_text_cond)
        del encoded_text_cond
        # 2、repeat text cond、split pixel value
        #[bs, 2*chunel, height, width]
        pixel_values = batch[pixel_value_key]
        #[2*bs, chunel, height, width]
        # print(f"pixel_values {pixel_values.shape}", flush=True)
        pixel_values = torch.cat(pixel_values.chunk(2, dim=1))
        latents = self.vae.encode(pixel_values.to(torch.bfloat16)).latent_dist.sample()
        #[2*bs, chunel,height, width]
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        # [2*bs, height*width, chunel]
        latents = self.extended_pipeline.pack_latents(latents, *latents.shape)
        # 3、prepare image id
        latent_image_ids = self.extended_pipeline.prepare_latent_image_ids(
            1, self.args.training.height // 8, self.args.training.width // 8, "cuda", torch.bfloat16
        )
        # print(f"{latents.shape} {latent_image_ids.shape}")
        # 4、sample and make traget and latent
        # Sample a random timestep for each image, timestep and noise is same for the dpo pair
        noise = torch.randn_like(latents)
        noise = noise.chunk(2)[0].repeat(2, 1, 1)
        target = noise - latents
        bsz = latents.shape[0] // 2
        timesteps_index = torch.randint(
            0,
            self.args.training.sampling_steps,
            (bsz, ),  # device=latents.device
        ).repeat(2)
        timesteps = self.extended_pipeline.timesteps[timesteps_index].detach().clone()
        sigmas = self.extended_pipeline.sigma_schedule[timesteps_index].reshape(
            [latents.shape[0], 1, 1]
        ).cuda()
        # flow matching modeling:
        # x_t = (1-t)*x_0 + t*x_1
        # 1、model_output predict （x_1 - x_0) from x_t
        # 2、x_t = (1-t)*x_0 + t*x_1 => x_0 = x_t - t* (x_1 - x_0)
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
        prepared_batch["latents"] = noisy_latents
        prepared_batch["timesteps"] = timesteps
        prepared_batch["latent_image_ids"] = latent_image_ids
        # print(f"target {target.shape} ")
        return target, prepared_batch

    #TODO(hessianliu): move this to pipeline
    def model_forward(self, model, prepared_batch):
        # print(f"prepared_batch {prepared_batch.keys()}")
        latents = prepared_batch["latents"]
        timesteps = prepared_batch["timesteps"]
        prompt_embeds = prepared_batch["prompt_embeds"]
        pooled_prompt_embeds = prepared_batch["pooled_prompt_embeds"]
        byt5_embeds = prepared_batch["byt5_embeds"]
        hidden_states_mask = prepared_batch["hidden_states_mask"]
        text_ids = prepared_batch["text_ids"]
        latent_image_ids = prepared_batch["latent_image_ids"]

        model_mbs = latents.shape[0]
        if self.args.training.guidance_scale > 1:
            latents = latents.repeat((2, 1, 1))  # just repeat
            timesteps = timesteps.repeat((2, ))  # just repeat
            tmp = self.extended_pipeline.uncond_feats["prompt_embeds"].repeat_interleave(
                model_mbs, dim=0
            )
            prompt_embeds = torch.cat([tmp, prompt_embeds], dim=0)
            tmp = self.extended_pipeline.uncond_feats["pooled_prompt_embeds"].repeat_interleave(
                model_mbs, dim=0
            )
            pooled_prompt_embeds = torch.cat([tmp, pooled_prompt_embeds], dim=0)
            tmp = self.extended_pipeline.uncond_feats["byt5_embeds"].repeat_interleave(
                model_mbs, dim=0
            )
            byt5_embeds = torch.cat([tmp, byt5_embeds], dim=0)
            tmp = self.extended_pipeline.uncond_feats["hidden_states_mask"].repeat_interleave(
                model_mbs, dim=0
            )
            hidden_states_mask = torch.cat([tmp, hidden_states_mask], dim=0)

        assert text_ids.ndim == 3 and latent_image_ids.ndim == 3
        noise_pred = model(
            hidden_states=latents.bfloat16(),
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids[0],
            img_ids=latent_image_ids[0],
            timestep=(timesteps / 1000).bfloat16(),  # bf16 / 1000 跟 long / 1000 -> bf16 差别较大
            guidance=None,
            joint_attention_kwargs=None,
            return_dict=False,
            encoder_hidden_states_byt5=byt5_embeds,
            encoder_hidden_states_mask=hidden_states_mask,
        )[0]
        if self.args.training.guidance_scale > 1:
            model_pred_uncond, model_pred_text = noise_pred.chunk(2)
            noise_pred = model_pred_uncond + \
                         self.args.training.guidance_scale * (model_pred_text - model_pred_uncond)
        return noise_pred

    @override
    def save_ckpt(self):
        self.extended_pipeline.model.save(self.global_step)

    @override
    def load_ckpt(self):
        pass
