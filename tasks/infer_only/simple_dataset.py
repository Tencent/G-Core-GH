"""
Simple dataset for inference only.
"""
import json
import os
from typing import Dict, List

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SimpleInferenceDataset(Dataset):
    """Simple dataset that loads data from jsonl files."""
    def __init__(self, data_pathes: List[str]):
        self.data = []
        for data_path in data_pathes:
            if os.path.isfile(data_path):
                # Single file
                self._load_file(data_path)
            elif os.path.isdir(data_path):
                # Directory: load all .jsonl files
                for filename in sorted(os.listdir(data_path)):
                    if filename.endswith('.jsonl') or filename.endswith('.json'):
                        filepath = os.path.join(data_path, filename)
                        self._load_file(filepath)
            else:
                raise ValueError(f"Invalid data path: {data_path}")

        print(f"Loaded {len(self.data)} samples from {data_pathes}")

    def _load_file(self, filepath: str):
        """Load data from a jsonl file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line in {filepath}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_batch_data(batch):
    """
    Convert list of dicts to dict of lists.
    
    Input: [{"query": "...", "search_res": "..."}, ...]
    Output: {"query": ["...", ...], "search_res": ["...", ...]}
    """
    if not batch:
        return {}

    # 获取所有键
    keys = batch[0].keys()

    # 转换为 dict of lists
    collated = {key: [sample[key] for sample in batch] for key in keys}

    return collated


def get_dataset_and_dataloader(config, tokenizer, dp_rank, dp_size):
    """
    Create dataset and dataloader for inference.
    
    Args:
        config: InferenceConfig object
        tokenizer: Tokenizer (not used in this simple version)
        dp_rank: Data parallel rank
        dp_size: Data parallel world size
    
    Returns:
        dict with keys: train_dataset, train_dataloader, train_sampler
    """
    # Create dataset
    dataset = SimpleInferenceDataset(config.data.data_pathes)

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=False,  # Don't shuffle for inference
        drop_last=False,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time
        sampler=sampler,
        num_workers=config.data.dataloader_num_workers,
        pin_memory=config.data.dataloader_pin_memory,
        collate_fn=collate_batch_data  #lambda x: x,  # Return list of dicts as-is
    )

    return {
        'train_dataset': dataset,
        'train_dataloader': dataloader,
        'train_sampler': sampler,
    }
