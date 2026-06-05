import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore

from gpatch_v4.configs.config import RlConfig
from megatron_datasets.qwenvl_dataset_map import (
    QwenVlDatasetV3,
    DataCollatorForQwenVlGRPO,
    get_processor,
)
from gpatch_v4.configs.config import OnPolicyDistillConfig


def get_dataset_and_dataloader(config: RlConfig = None, tokenizer=None, dp_rank=0, dp_size=1):
    data_config = config.data
    if data_config.data_extra.processor_path is None:
        data_config.data_extra.processor_path = config.policy.hf_tokenizer_path
    processor = get_processor(data_config.data_extra, model_arch=config.policy.model_arch)
    hf_config = AutoConfig.from_pretrained(data_config.data_extra.processor_path)

    v3_config = data_config.dataset_v3
    dataset = QwenVlDatasetV3(
        tokenizer=tokenizer,
        max_seq_len=config.training.seq_length,
        train_path_likes=v3_config.train_data_path,
        domain_probabilities=v3_config.train_probability,
        domain_names=v3_config.train_data_domain_names,
        total_nums=v3_config.train_total_nums,
        global_batch_size=config.training.train_gbs,
        # TODO(guanyouhe): 续训
        train_data_consuming_progresses=None,
        rank=torch.distributed.get_rank(),
        dp_rank=dp_rank,
        dp_size=dp_size,
        num_workers=data_config.dataloader_num_workers,
        shuffle_buffer_size=data_config.shuffle_buffer_size,
        seed=42,
        use_grpo=True,
        grpo_resp_length=config.sampler.infer_engine_configs[0].generate_max_tokens,
        lmdb_port=data_config.data_extra.lmdb_port,
        hf_config=hf_config,
        min_pixels=data_config.data_extra.min_pixels_num,
        max_pixels=data_config.data_extra.max_pixels_num,
        processor=processor,
        mask_history=data_config.data_extra.mask_history,
        meta_info_key="meta_info",
        moe_pad_with_random_token=False,
    )

    collate_func = DataCollatorForQwenVlGRPO(
        hw_factor=1,
        model_arch=config.policy.model_arch,
        tokenizer=tokenizer,
        is_dpo=False,
        use_grpo=True,
        cp_size=config.policy.dist_config.context_parallel_size,
        hf_config_path=data_config.data_extra.processor_path,
    )

    # TODO(guanyouhe): add cyclic_iter 的参数让它搞一个 iter
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_func,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.rollout_mbs,
        num_workers=config.data.dataloader_num_workers,
        prefetch_factor=config.data.dataloader_prefetch_factor,
        drop_last=True,
    )
    return {
        'train_dataset': dataset,
        'train_sampler': None,
        'train_dataloader': dataloader,
    }


def get_batched_data(batched_data=None):
    assert batched_data is not None
    # 看起来不太有必要保留这个
    return batched_data


from transformers import AutoTokenizer

cs = ConfigStore.instance()
cs.store(name="config_root", node=OnPolicyDistillConfig)


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: OnPolicyDistillConfig):
    default_config = OnPolicyDistillConfig()
    merged_config = OmegaConf.merge(default_config, cfg)
    merged_obj = OmegaConf.to_object(merged_config)

    tokenizer = AutoTokenizer.from_pretrained(merged_obj.policy.hf_tokenizer_path)
    ds_dict = get_dataset_and_dataloader(merged_obj, tokenizer)

    train_iter = iter(ds_dict['train_dataloader'])
    for i in range(7000):
        batch = next(train_iter)
        print(f"{i=} {batch.keys()=}")


if __name__ == "__main__":
    main()
