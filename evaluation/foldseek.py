import torch
import argparse
from huggingface_hub import login, hf_hub_download
from safetensors.torch import load_file
from transformers import EsmTokenizer
from datasets import load_dataset
from typing import List
from tqdm import tqdm

from models.modeling_dsm import DSM, DSMConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='lhallee/DSM_650_fs')
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()


args = parse_args()

if args.token is not None:
    login(args.token)

model_path = args.model_path
tokenizer_path = 'lhallee/joint_tokenizer'


# Download the safetensors file
local_weight_file = hf_hub_download(
    repo_id=model_path,
    filename='model.safetensors',
    repo_type='model',
)

config = DSMConfig.from_pretrained(model_path)
model = DSM(config)

# Load the state dict and remove _orig_mod prefixes
state_dict = load_file(local_weight_file)

# Remove _orig_mod. prefix from all keys
cleaned_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('_orig_mod.'):
        # Remove the _orig_mod. prefix
        cleaned_key = key[len('_orig_mod.'):]
        cleaned_state_dict[cleaned_key] = value
    else:
        # Keep the key as is if it doesn't have the prefix
        cleaned_state_dict[key] = value

# Load the cleaned state dict into the model
missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

if missing_keys:
    print(f"Missing keys when loading: {len(missing_keys)} keys")
    print(f"First few missing keys: {missing_keys[:3]}")
if unexpected_keys:
    print(f"Unexpected keys when loading: {len(unexpected_keys)} keys") 
    print(f"First few unexpected keys: {unexpected_keys[:3]}")

if not missing_keys and not unexpected_keys:
    print("Successfully loaded all weights!")
elif len(missing_keys) == 0:
    print("All expected weights loaded (some unexpected keys found)")
else:
    print(f"Loaded with {len(missing_keys)} missing keys")


print(model)

tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)
model.tokenizer = tokenizer
extra_tokens = ['<aa>', '<fs>', '<sep>', '<bos>', '<eos>', '<cls>']
model.get_special_token_ids(extra_tokens)

dataset = load_dataset('lhallee/foldseek_dataset')
dataset = dataset.rename_columns({'seqs': 'aa_seqs', 'labels': 'fs_seqs'})
test_dataset = dataset['test'].filter(lambda x: len(x['aa_seqs']) <= 128).select(range(100))
print(test_dataset)


class ProteinFolder:
    def __init__(self, model, batch_size: int = 4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def string_accuracy(self, y_true: List[str], y_pred: List[str]):
        total_correct, total = 0, 0
        for y_true_i, y_pred_i in zip(y_true, y_pred):
            for t, p in zip(y_true_i, y_pred_i):
                total_correct += t == p
            total += len(y_true_i)
        return total_correct / total

    @torch.no_grad()
    def fold(self, aa_seqs: List[str], fs_seqs: List[str]):
        #seqs = [
        #    '<aa>' + '<mask>' * len(aa) + '<eos>' + '<fs>' + '<mask>' * len(fs) for aa, fs in zip(aa_seqs, fs_seqs)
        #]
        seqs = [
            '<aa>' + aa + '<eos>' + '<fs>' + fs[0] + '<mask>' * (len(fs) - 1) for aa, fs in zip(aa_seqs, fs_seqs)
        ]
        tokenizer = self.model.tokenizer

        final_preds, final_true = [], []
        for i in tqdm(range(0, len(seqs), self.batch_size)):
            batch_seqs = seqs[i:i+self.batch_size]
            batch_aa_seqs = aa_seqs[i:i+self.batch_size]
            batch_fs_seqs = fs_seqs[i:i+self.batch_size]
            tokenized = tokenizer(
                batch_seqs,
                padding='longest',
                return_tensors='pt',
                add_special_tokens=True,
            )
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            outputs = self.model.mask_diffusion_generate(
                tokenizer=tokenizer,
                extra_tokens=extra_tokens,
                input_tokens=input_ids,
                attention_mask=attention_mask,
                step_divisor=1,
                temperature=1.0,
                remasking='random',
                preview=True,
                slow=False,
            )
            aa_preds, fs_preds = self.model.decode_dual_input(outputs, attention_mask, '<eos>')
            for aa, fs, fs_true in zip(aa_preds, fs_preds, batch_fs_seqs):
                aa = aa.replace('<bos>', '').replace('<aa>', '')
                fs = fs.replace('<fs>', '').replace('<eos>', '')
                final_preds.append(fs)
                final_true.append(fs_true)

        return self.string_accuracy(final_true, final_preds)


protein_folder = ProteinFolder(model, args.batch_size)
print(protein_folder.fold(test_dataset['aa_seqs'], test_dataset['fs_seqs']))
