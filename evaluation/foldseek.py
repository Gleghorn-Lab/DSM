import torch
from transformers import EsmTokenizer
from datasets import load_dataset
from typing import List
from tqdm import tqdm

from models.modeling_dsm import DSM


model_path = 'lhallee/DSM_150_fs'
tokenizer_path = 'lhallee/joint_tokenizer'

model = DSM.from_pretrained(model_path)
tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)
model.tokenizer = tokenizer
extra_tokens = ['<aa>', '<fs>', '<sep>', '<bos>', '<eos>', '<cls>']
model.get_special_token_ids(extra_tokens)

dataset = load_dataset('lhallee/foldseek_dataset')
dataset = dataset.rename_columns({'seqs': 'aa_seqs', 'labels': 'fs_seqs'})
test_dataset = dataset['test'].filter(lambda x: len(x['aa_seqs']) <= 256).select(range(100))
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
        seqs = [
            '<aa>' + aa + '<sep>' + '<fs>' + fs[:len(fs)//2] + ''.join(['<mask>'] * (len(fs) - len(fs)//2)) for aa, fs in zip(aa_seqs, fs_seqs)
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
                steps=10,
                temperature=1.0,
                remasking='random',
                preview=False,
                slow=False,
            )
            aa_preds, fs_preds = self.model.decode_dual_input(outputs, attention_mask, '<sep>')
            for aa, fs, fs_true in zip(aa_preds, fs_preds, batch_fs_seqs):
                aa = aa.replace('<bos>', '').replace('<aa>', '')
                fs = fs.replace('<fs>', '').replace('<eos>', '')
                final_preds.append(fs)
                final_true.append(fs_true)

        return self.string_accuracy(final_true, final_preds)


protein_folder = ProteinFolder(model)
print(protein_folder.fold(test_dataset['aa_seqs'], test_dataset['fs_seqs']))
