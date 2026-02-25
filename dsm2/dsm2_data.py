from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import Dataset

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


def _shuffle_and_limit_split(split_dataset, split_name: str, limit: int, shuffle_seed: int):
    shuffled = split_dataset.shuffle(seed=shuffle_seed)
    if limit <= 0:
        return shuffled

    upper = min(limit, len(shuffled))
    assert upper > 0, f"{split_name} split is empty after applying limit={limit}."
    return shuffled.select(range(upper))


def build_dsm2_data_bundle(config: DSM2DataConfig, tokenizer) -> DSM2DataBundle:
    hf_dataset = load_dataset(config.data_path)
    assert "train" in hf_dataset, "Dataset must contain a 'train' split."
    assert "valid" in hf_dataset, "Dataset must contain a 'valid' split."
    assert "test" in hf_dataset, "Dataset must contain a 'test' split."

    train_split = _shuffle_and_limit_split(hf_dataset["train"], "train", config.train_limit, config.shuffle_seed)
    valid_split = _shuffle_and_limit_split(hf_dataset["valid"], "valid", config.valid_limit, config.shuffle_seed)
    test_split = _shuffle_and_limit_split(hf_dataset["test"], "test", config.test_limit, config.shuffle_seed)

    assert config.sequence_column in train_split.column_names, f"Missing column '{config.sequence_column}' in train split."
    assert config.sequence_column in valid_split.column_names, f"Missing column '{config.sequence_column}' in valid split."
    assert config.sequence_column in test_split.column_names, f"Missing column '{config.sequence_column}' in test split."

    train_dataset = HFSequenceDataset(train_split, config.sequence_column)
    valid_dataset = HFSequenceDataset(valid_split, config.sequence_column)
    test_dataset = HFSequenceDataset(test_split, config.sequence_column)
    data_collator = SequenceCollator(tokenizer, max_length=config.max_length)

    return DSM2DataBundle(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
    )
