import torch
from typing import Tuple, List, Dict, Union, Any

try:
    from data.utils import ProteinMasker
except:
    from .utils import ProteinMasker


def standard_data_collator(batch):
    batch = {k: torch.stack([ex[k] for ex in batch]) for k in batch[0].keys()}
    return batch


class AutoencoderCollator:
    def __init__(self, tokenizer, max_length, **kwargs):
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
    def __init__(self, tokenizer, **kwargs):
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
    def __init__(self, tokenizer, max_length=512, **kwargs):
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
    

class SequenceCollator_mask:
    def __init__(self, tokenizer, max_length=512, mask_rate=0.15, **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masker = ProteinMasker(tokenizer, mask_rate)

    def __call__(self, batch: Tuple[List[str], List[str]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(batch,
                          padding='longest',
                          truncation=True,
                          max_length=self.max_length,
                          return_tensors='pt',
                          add_special_tokens=True)
        batch['original_ids'] = batch['input_ids'].clone()
        batch['input_ids'], batch['labels'] = self.masker(batch['input_ids'], batch['attention_mask'])
        return batch


class PairCollator_input_ids:
    def __init__(self, tokenizer, max_length=2048, **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)
        tokenized = self.tokenizer(
            seqs_a, seqs_b,
            padding='longest',
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }
    

class DummyPairCollator_input_ids:
    def __init__(self, tokenizer, max_length=2048, **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        _, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)
        tokenized = self.tokenizer(seqs_b,
            padding='longest',
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }


class PairCollator_ab:
    def __init__(self, tokenizer, max_length=2048, **kwargs):
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


class DiffATCollator:
    def __init__(self, tokenizer, at_vocab_size: int):
        self.tokenizer = tokenizer
        self.pad_token_id = 0
        self.at_vocab_size = at_vocab_size

    def _pad_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [self.pad_token_id] * (max_length - len(seq)) for seq in sequences]
        return torch.tensor(padded_sequences, dtype=torch.long)

    def _create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (input_ids != self.pad_token_id).long()
        return attention_mask

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = [example[0] for example in examples]
        anns = [example[1] for example in examples]
        tokenized = self.tokenizer(seqs, padding='longest', return_tensors='pt')
        at_ids = self._pad_sequences(anns)
        at_attention_mask = self._create_attention_mask(at_ids)
        batch = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'at_ids': at_ids,
            'at_attention_mask': at_attention_mask,
        }
        return batch
