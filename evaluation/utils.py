
import torch
import numpy as np
import random
from typing import List
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from models.modeling_dsm import DSM_Binders, DSMConfig
from models.utils import wrap_lora


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_random_aa_sequence(length: int, alphabet: str) -> str:
    """Generates a random amino acid sequence of a given length."""
    return "".join(random.choice(alphabet) for _ in range(length))


def load_binder_model(model_path):
    local_weight_file = hf_hub_download(
        repo_id=model_path,
        filename='model.safetensors',
        repo_type='model',
    )

    config = DSMConfig.from_pretrained(model_path)
    model = DSM_Binders(config=config)
    model = wrap_lora(model, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
    state_dict = load_file(local_weight_file)

    loaded_params = set()
    missing_params = set()

    for name, param in model.named_parameters():
        found = False
        for key in state_dict.keys():
            if key in name: # Simplified matching, original was more complex but this should work for LoRA
                param.data = state_dict[key]
                loaded_params.add(name)
                found = True
                break
        if not found:
            missing_params.add(name)

    print(f"Loaded {len(loaded_params)} parameters")
    print(f"Missing {len(missing_params)} parameters")
    if missing_params:
        print("Missing parameters:")
        for param in sorted(missing_params):
            print(f"  - {param}")
    return model



def get_eval_data(num_samples: int = None) -> List[str]:
    local_file = hf_hub_download(
        repo_id="Synthyra/omg_prot50",
        filename=f"data/valid-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=888)
    if num_samples is not None:
        data = data.select(range(num_samples))
    data = data.filter(lambda x: len(x['sequence']) > 20 and len(x['sequence']) < 2048)
    print(data)
    valid_seqs = sorted(data['sequence'], key=len, reverse=True)
    return valid_seqs


def get_ppi_examples(num_samples: int = 100, max_combined_length: int = 1024):
    positives = load_dataset('Synthyra/BIOGRID-MV', split='train')
    positives = positives.filter(lambda x: len(x['SeqA']) > 20 and len(x['SeqB']) > 20 and len(x['SeqA']) + len(x['SeqB']) < max_combined_length)
    negatives = load_dataset('Synthyra/NEGATOME', split='combined')
    negatives = negatives.filter(lambda x: len(x['SeqA']) > 20 and len(x['SeqB']) > 20 and len(x['SeqA']) + len(x['SeqB']) < max_combined_length)
    positives = positives.shuffle(seed=42).select(range(num_samples))
    negatives = negatives.shuffle(seed=42).select(range(num_samples))
    return positives, negatives