from datasets import load_dataset
import torch.nn as nn
from peft import LoraConfig, LoraModel


def wrap_lora(module: nn.Module, r: int, lora_alpha: float, lora_dropout: float) -> nn.Module:
    # these modules handle ESM++ and ESM2 attention types, as well as any additional transformer blocks from Syndev
    target_modules=["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    module = LoraModel(module, lora_config, 'default')
    for name, param in module.named_parameters():
        if 'classifier' in name.lower():
            param.requires_grad = True
    return module


train_labels = train_set['labels']
unique_tags = set(tag for doc in train_labels for tag in doc)
tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
# add cls token to labels
train_set = train_set.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)})
valid_set = valid_set.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)})
test_set = test_set.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)})
label_type = 'tokenwise'
num_labels = len(unique_tags)
