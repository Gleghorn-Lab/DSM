import os
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, Sampler
from typing import Optional

from data.data_collators import SequenceCollator
from dsm2.dsm2_config import DSM2DataConfig


class HFSequenceDataset(Dataset):
    def __init__(self, dataset, sequence_column: str):
        self.dataset = dataset
        self.sequence_column = sequence_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return row[self.sequence_column]


@dataclass
class DSM2DataBundle:
    train_dataset: Dataset
    valid_dataset: Dataset
    test_dataset: Dataset
    data_collator: SequenceCollator


@dataclass
class DSM2DataLoaders:
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    train_sampler: Optional[Sampler]
    valid_sampler: Optional[Sampler]
    test_sampler: Optional[Sampler]


def _shuffle_and_limit_split(split_dataset, split_name: str, limit: int, shuffle_seed: int):
    shuffled = split_dataset.shuffle(seed=shuffle_seed)
    if limit <= 0:
        return shuffled

    upper = min(limit, len(shuffled))
    assert upper > 0, f"{split_name} split is empty after applying limit={limit}."
    return shuffled.select(range(upper))


def build_dsm2_data_bundle(config: DSM2DataConfig, tokenizer, bugfix: bool = False) -> DSM2DataBundle:
    if bugfix:
        data_path, sequence_column = 'GleghornLab/MB_reg', 'seqs'
    else:
        data_path, sequence_column = config.data_path, config.sequence_column

    hf_dataset = load_dataset(data_path)
    assert "train" in hf_dataset, "Dataset must contain a 'train' split."
    assert "valid" in hf_dataset, "Dataset must contain a 'valid' split."
    assert "test" in hf_dataset, "Dataset must contain a 'test' split."

    train_split = _shuffle_and_limit_split(hf_dataset["train"], "train", config.train_limit, config.shuffle_seed)
    valid_split = _shuffle_and_limit_split(hf_dataset["valid"], "valid", config.valid_limit, config.shuffle_seed)
    test_split = _shuffle_and_limit_split(hf_dataset["test"], "test", config.test_limit, config.shuffle_seed)

    assert sequence_column in train_split.column_names, f"Missing column '{sequence_column}' in train split."
    assert sequence_column in valid_split.column_names, f"Missing column '{sequence_column}' in valid split."
    assert sequence_column in test_split.column_names, f"Missing column '{sequence_column}' in test split."

    train_dataset = HFSequenceDataset(train_split, sequence_column)
    valid_dataset = HFSequenceDataset(valid_split, sequence_column)
    test_dataset = HFSequenceDataset(test_split, sequence_column)
    data_collator = SequenceCollator(tokenizer, max_length=config.max_length)

    return DSM2DataBundle(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
    )


def build_dsm2_dataloaders(
    data_bundle: DSM2DataBundle,
    patch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    is_distributed: bool,
    rank: int,
    world_size: int,
) -> DSM2DataLoaders:
    assert patch_size > 0, "patch_size must be > 0."
    assert num_workers >= 0, "num_workers must be >= 0."
    assert prefetch_factor > 0, "prefetch_factor must be > 0."

    if is_distributed:
        train_sampler = DistributedSampler(
            data_bundle.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        valid_sampler = DistributedSampler(
            data_bundle.valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            data_bundle.test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = RandomSampler(data_bundle.train_dataset)
        valid_sampler = RandomSampler(data_bundle.valid_dataset)
        test_sampler = RandomSampler(data_bundle.test_dataset)

    common_collate_kwargs = {
        "pin_memory": pin_memory,
        "collate_fn": data_bundle.data_collator,
    }

    train_loader_kwargs = {
        "num_workers": num_workers,
        **common_collate_kwargs,
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = prefetch_factor

    is_windows = os.name == "nt"
    eval_num_workers = 0 if is_windows else num_workers
    # On Windows, keep eval/test single-process to avoid expensive spawn latency.
    # On Linux, allow eval/test workers for better throughput.
    eval_loader_kwargs = {
        "num_workers": eval_num_workers,
        **common_collate_kwargs,
    }
    if eval_num_workers > 0:
        eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        data_bundle.train_dataset,
        sampler=train_sampler,
        batch_size=patch_size,
        drop_last=True,
        **train_loader_kwargs,
    )
    valid_loader = DataLoader(
        data_bundle.valid_dataset,
        sampler=valid_sampler,
        batch_size=patch_size,
        drop_last=False,
        **eval_loader_kwargs,
    )
    test_loader = DataLoader(
        data_bundle.test_dataset,
        sampler=test_sampler,
        batch_size=patch_size,
        drop_last=False,
        **eval_loader_kwargs,
    )

    return DSM2DataLoaders(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_sampler=train_sampler,
        valid_sampler=valid_sampler,
        test_sampler=test_sampler,
    )
