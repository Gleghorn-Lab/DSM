import torch
from transformers import EsmTokenizer
from datasets import load_dataset
from typing import List
from tqdm import tqdm

from models.modeling_dsm import DSM


model_path = 'lhallee/DSM_fs'
tokenizer_path = 'lhallee/joint_tokenizer'

model = DSM.from_pretrained(model_path)
tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)
model.tokenizer = tokenizer
model.get_special_token_ids()

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
        #seqs = [
        #    a + '<sep>' + ''.join(['<mask>'] * len(a)) for a in aa_seqs
        #]
        # aa + <sep> + fs
        # randomly mask 15% of the fs tokens
        tokenizer = self.model.tokenizer
        mask_token = tokenizer.mask_token
        sep_token = tokenizer.sep_token

        seqs = []
        for aa, fs in zip(aa_seqs, fs_seqs):
            # Mask 15% of fs tokens at random
            fs_chars = list(fs)
            num_to_mask = max(1, int(0.15 * len(fs_chars)))
            mask_indices = torch.randperm(len(fs_chars))[:num_to_mask].tolist()
            for idx in mask_indices:
                fs_chars[idx] = mask_token
            masked_fs = ''.join(fs_chars)
            seqs.append(aa + sep_token + masked_fs)

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
            logits = self.model._get_logits(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu()
            for pred, seq, aa, fs, mask in zip(preds, batch_seqs, batch_aa_seqs, batch_fs_seqs, attention_mask):
                pred = pred[len(aa)+1:mask.sum().item() - 1]
                pred = tokenizer.decode(pred, skip_special_tokens=True).replace(' ', '')

                #assert len(pred) == len(aa)
                if len(pred) != len(fs):
                    continue
                final_preds.append(pred)
                final_true.append(fs)
        #for true, pred in zip(final_true, final_preds):
        #    print(true)
        #    print(pred)
        #    print(len(true), len(pred))
        #    print('-'*100)
        return self.string_accuracy(final_true, final_preds)


protein_folder = ProteinFolder(model)
print(protein_folder.fold(test_dataset['aa_seqs'], test_dataset['fs_seqs']))

