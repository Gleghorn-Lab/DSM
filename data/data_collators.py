import torch
import numpy as np
import random
from torch.nn.functional import pad
from typing import Tuple, List, Dict, Union


def standard_data_collator(batch):
    batch = {k: torch.stack([ex[k] for ex in batch]) for k in batch[0].keys()}
    return batch


class AutoencoderCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Tuple[List[str], List[str]]) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            batch,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        tokenized['labels'] = tokenized['input_ids'].clone()
        tokenized['labels'][tokenized['attention_mask'] == 0] = -100
        return tokenized


class SequenceLabelCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Tuple[str, List[float]]]) -> Dict[str, torch.Tensor]:
        seqs = [ex[0] for ex in batch]
        labels = [ex[1] for ex in batch]
        batch = self.tokenizer(seqs,
                          padding='longest',
                          truncation=False,
                          return_tensors='pt',
                          add_special_tokens=True)
        batch['labels'] = torch.stack([torch.tensor(label, dtype=torch.float) for label in labels])
        return batch


class SequenceCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Tuple[List[str], List[str]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(batch,
                          padding='longest',
                          truncation=True,
                          max_length=self.max_length,
                          return_tensors='pt',
                          add_special_tokens=True)
        return batch


class PairCollator_input_ids:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)
        tokenized = self.tokenizer(
            seqs_a, seqs_b,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }


class PairCollator_ab:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)
        tokenized_a = self.tokenizer(
            seqs_a,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        tokenized_b = self.tokenizer(
            seqs_b,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids_a': tokenized_a['input_ids'],
            'input_ids_b': tokenized_b['input_ids'],
            'attention_mask_a': tokenized_a['attention_mask'],
            'attention_mask_b': tokenized_b['attention_mask'],
            'labels': labels
        }
