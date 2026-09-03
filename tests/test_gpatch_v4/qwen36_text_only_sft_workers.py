"""Ray workers for Qwen3.6 text-only SFT tests.

Workers are nested @ray.remote callables with no free variables so cloudpickle
serializes bytecode only (avoids ConfigModuleInstance in driver process globals).
"""

import ray
import torch


def _logits_cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-position cosine similarity with exp (softmax-like) normalization."""
    a, b = a.float(), b.float()
    a = torch.exp(a - a.max(dim=-1, keepdim=True)[0])
    b = torch.exp(b - b.max(dim=-1, keepdim=True)[0])
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a * b).sum(dim=-1)


def _build_hf_fwd_bwd_worker():
    @ray.remote(num_gpus=1)
    def _hf_fwd_bwd_worker(hf_model_path: str, seq_len: int = 2048):
        import torch
        from transformers import AutoTokenizer
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError:
            Qwen3_5ForConditionalGeneration = None
        assert Qwen3_5ForConditionalGeneration is not None, (
            "Qwen3_5ForConditionalGeneration not found; need transformers >= 5.2.0"
        )

        torch.cuda.set_device(0)

        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
        rng = torch.Generator().manual_seed(42)
        input_ids = torch.randint(
            0,
            tokenizer.vocab_size,
            (1, seq_len),
            generator=rng,
        ).cuda()

        print(f"[HF] loading {hf_model_path} ...")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            hf_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda()
        model.train()
        print(f"[HF] model loaded, running forward + backward ...")

        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        loss = logits.sum()
        loss.backward()
        logits = logits.detach()

        grad_norms = {}
        total_sq = 0.0
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                gn = param.grad.float().norm().item()
                grad_norms[name] = gn
                total_sq += gn**2
                if not torch.isfinite(param.grad).all():
                    has_nan = True
        total_grad_norm = total_sq**0.5

        print(
            f"[HF] logits.shape={list(logits.shape)} "
            f"logits.mean={logits.float().mean().item():.4f} "
            f"total_grad_norm={total_grad_norm:.4f} num_grads={len(grad_norms)} "
            f"has_nan={has_nan}"
        )

        return {
            "logits": logits.cpu(),
            "grad_norms": grad_norms,
            "logits_shape": list(logits.shape),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "logits_mean": logits.float().mean().item(),
            "total_grad_norm": total_grad_norm,
            "num_grads": len(grad_norms),
            "has_nan": has_nan,
        }

    return _hf_fwd_bwd_worker


def _build_mbridge_fwd_bwd_worker():
    @ray.remote(num_gpus=1)
    def _mbridge_fwd_bwd_worker(
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        hf_model_path: str,
        tp: int,
        ep: int,
        cp: int = 1,
        save_dir: str = None,
        return_logits: bool = False,
        return_grad_norms: bool = False,
        mcore_extra_config: dict = None,
        seq_len: int = 2048,
    ):
        import os

        import torch
        import torch.nn.functional as F
        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        from megatron.core.tensor_parallel.mappings import (
            gather_from_tensor_model_parallel_region,
        )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from mbridge import AutoBridge
        from transformers import AutoTokenizer

        def _gather_from_cp(tensor, seq_dim, cp_size, cp_group):
            assert seq_dim in (0, 1) and tensor.dim() > seq_dim
            tensor = tensor.view(
                *tensor.shape[:seq_dim],
                2,
                tensor.shape[seq_dim] // 2,
                *tensor.shape[seq_dim + 1:],
            )
            gathered = [torch.zeros_like(tensor) for _ in range(cp_size)]
            torch.distributed.all_gather(gathered, tensor, group=cp_group)
            reordered = [None] * (2 * cp_size)
            for r in range(cp_size):
                if seq_dim == 1:
                    reordered[r] = gathered[r][:, 0]
                    reordered[2 * cp_size - r - 1] = gathered[r][:, 1]
                else:
                    reordered[r] = gathered[r][0]
                    reordered[2 * cp_size - r - 1] = gathered[r][1]
            return torch.cat(reordered, dim=seq_dim)

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"
        torch.cuda.set_device(0)

        torch.distributed.init_process_group(backend="nccl")
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp,
            expert_model_parallel_size=ep,
        )
        model_parallel_cuda_manual_seed(0)

        print(f"[rank {rank}] loading model (tp={tp}, cp={cp}, ep={ep}) ...")
        bridge = AutoBridge.from_pretrained(
            hf_model_path,
            trust_remote_code=True,
            mcore_extra_config=mcore_extra_config,
        )
        bridge.config.sequence_parallel = tp > 1
        model = bridge.get_model()
        if rank == 0:
            print(f"[rank {rank}] model len={len(model)} config={bridge.config}")
        bridge.load_weights(model, hf_model_path, memory_efficient=True)
        print(f"[rank {rank}] weights loaded, preparing fwd+bwd ...")
        torch.distributed.barrier()

        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
        rng = torch.Generator().manual_seed(42)
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), generator=rng)
        micro_batch_size, real_seq_length = input_ids.size()

        seq_length_factor = tp
        if cp > 1:
            seq_length_factor *= cp * 2
        seq_length = real_seq_length
        if real_seq_length % seq_length_factor != 0:
            seq_length = (
                (real_seq_length + seq_length_factor - 1) // seq_length_factor
                * seq_length_factor
            )
            input_ids = F.pad(input_ids, (0, seq_length - real_seq_length), value=0)

        sample_list = [{"input_ids": input_ids}]
        logits_container = []

        def fwd_fn(data_iter, model):
            sample = next(data_iter)
            output_tensor = model(
                input_ids=sample["input_ids"].cuda(),
                position_ids=None,
                attention_mask=None,
            )
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            assert isinstance(output_tensor, torch.Tensor)

            def loss_func(output_tensor, non_loss_data=True):
                logits_container.append(output_tensor.detach())
                cp_size = mpu.get_context_parallel_world_size()
                loss = output_tensor.sum() / cp_size
                return loss, {"loss": loss.detach()}

            return output_tensor, loss_func

        fwd_bwd_function = get_forward_backward_func()
        fwd_bwd_function(
            forward_step_func=fwd_fn,
            data_iterator=iter(sample_list),
            model=model,
            num_microbatches=1,
            forward_only=False,
            seq_length=seq_length,
            decoder_seq_length=seq_length,
            micro_batch_size=micro_batch_size,
        )

        fwd_result = {}
        if mpu.is_pipeline_last_stage() and logits_container:
            logits = logits_container[0]
            if mpu.get_context_parallel_world_size() > 1:
                logits = _gather_from_cp(
                    logits,
                    1,
                    mpu.get_context_parallel_world_size(),
                    mpu.get_context_parallel_group(),
                )
            if mpu.get_tensor_model_parallel_world_size() > 1:
                logits = gather_from_tensor_model_parallel_region(logits)
            logits = logits[:, :real_seq_length, :]

            fwd_result = {
                "logits_shape": list(logits.shape),
                "logits_finite": bool(torch.isfinite(logits).all()),
                "logits_mean": logits.float().mean().item(),
            }
            print(
                f"[rank {rank}] fwd OK  logits.shape={fwd_result['logits_shape']} "
                f"logits.mean={fwd_result['logits_mean']:.4f}"
            )
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"logits_tp{tp}_cp{cp}_ep{ep}.pt")
                torch.save(logits.cpu(), path)
                print(f"[rank {rank}] saved logits to {path}")
            if return_logits:
                fwd_result["logits"] = logits.cpu()

        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            cp_group = mpu.get_context_parallel_group()
            for param in model[0].parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad,
                        op=torch.distributed.ReduceOp.SUM,
                        group=cp_group,
                    )

        grad_norms = {}
        grad_norm_sq = torch.zeros(1, dtype=torch.float32, device="cuda")
        has_nan = False
        has_zero = True
        for name, param in model[0].named_parameters():
            if param.grad is not None:
                gn = param.grad.float().norm().item()
                grad_norms[name] = gn
                grad_norm_sq += gn**2
                if not torch.isfinite(param.grad).all():
                    has_nan = True
                if gn > 0:
                    has_zero = False

        model_parallel_group = mpu.get_model_parallel_group()
        torch.distributed.all_reduce(
            grad_norm_sq,
            op=torch.distributed.ReduceOp.SUM,
            group=model_parallel_group,
        )
        total_grad_norm = grad_norm_sq.item()**0.5

        bwd_result = {
            "total_grad_norm": total_grad_norm,
            "num_grads": len(grad_norms),
            "has_nan": has_nan,
            "all_zero": has_zero,
        }
        print(
            f"[rank {rank}] bwd OK  total_grad_norm={total_grad_norm:.4f} "
            f"num_grads={len(grad_norms)} has_nan={has_nan} all_zero={has_zero}"
        )
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(
                save_dir, f"grad_norms_tp{tp}_cp{cp}_ep{ep}_rank{rank}.pt"
            )
            torch.save(grad_norms, path)
            print(f"[rank {rank}] saved grad_norms to {path}")
        if return_grad_norms:
            bwd_result["grad_norms"] = grad_norms

        config_snapshot = {
            "cp_comm_type": bridge.config.cp_comm_type,
        }

        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return {"rank": rank, **fwd_result, **bwd_result, **config_snapshot}

    return _mbridge_fwd_bwd_worker


_hf_fwd_bwd_worker = _build_hf_fwd_bwd_worker()
_mbridge_fwd_bwd_worker = _build_mbridge_fwd_bwd_worker()
