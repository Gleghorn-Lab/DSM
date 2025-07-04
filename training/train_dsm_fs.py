#! /usr/bin/env python3
# py -m train_dsm_ppi_both
import argparse
import copy
import torch
import torch.nn.functional as F
import random
import pathlib
import os
from torch.utils.data import Dataset as TorchDataset
from torchinfo import summary
from typing import List, Tuple, Dict


base_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/lhallee-ppi/code"
cache_root = f"{base_path}/hf_cache"
tmp_root   = f"{base_path}/tmp"

if os.path.exists(base_path):
    # Create the directories once
    pathlib.Path(cache_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmp_root).mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"]            = cache_root              # master switch
    os.environ["HF_DATASETS_CACHE"]  = f"{cache_root}/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_root}/transformers"
    os.environ["HF_HUB_CACHE"]       = f"{cache_root}/hub"
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
    print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    print(f"HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")
else:
    print("HF_HOME not set, skipping cache setup")

### HF imports, needs to happen after HOME is set
from transformers import TrainingArguments, EvalPrediction, Trainer, EsmTokenizer
from huggingface_hub import login
from datasets import load_dataset

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

from models.modeling_dsm import DSM


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class ComputeMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Build sets of amino acid and foldseek token ids
        # Amino acids: single uppercase letters (A-Z)
        # Foldseek: single lowercase letters (a-z)
        self.amino_acid_tokens = list('ALGVSREDTIPKFQNYMHWCXBOUZ')
        self.foldseek_tokens = list('algvsredtipkfqnymhwc')
        self.amino_acid_token_ids = set(self.tokenizer.convert_tokens_to_ids(t) for t in self.amino_acid_tokens)
        self.foldseek_token_ids = set(self.tokenizer.convert_tokens_to_ids(t) for t in self.foldseek_tokens)

    def __call__(self, eval_preds: EvalPrediction):
        ### NOTE the eval mask percentage is fixed at 15%
        metrics = {}
        lm_logits = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
        lm_logits, labels = lm_logits

        # labels are already -100 for non-masked tokens
        lm_logits_torch = torch.tensor(lm_logits)
        labels_torch = torch.tensor(labels)
        # We need to do this because the eval loss is scaled by the mask rate
        cross_entropy_loss = F.cross_entropy(
            lm_logits_torch.view(-1, lm_logits_torch.shape[-1]), 
            labels_torch.view(-1),
            ignore_index=-100
        )

        metrics['cross_entropy_loss'] = cross_entropy_loss.item() if hasattr(cross_entropy_loss, "item") else float(cross_entropy_loss)

        y_pred = lm_logits.argmax(axis=-1).flatten()
        y_true = labels.flatten()
        valid_indices = y_true != -100
        y_pred = y_pred[valid_indices]
        y_true = y_true[valid_indices]

        # Compute mask for amino acid and foldseek tokens
        y_true_np = y_true.cpu().numpy() if hasattr(y_true, "cpu") else y_true
        y_pred_np = y_pred.cpu().numpy() if hasattr(y_pred, "cpu") else y_pred

        aa_mask = [t in self.amino_acid_token_ids for t in y_true_np]
        fs_mask = [t in self.foldseek_token_ids for t in y_true_np]

        # All tokens
        metrics["f1"] = f1_score(y_true, y_pred, average='weighted')
        metrics["prec"] = precision_score(y_true, y_pred, average='weighted')
        metrics["rec"] = recall_score(y_true, y_pred, average='weighted')
        metrics["acc"] = accuracy_score(y_true, y_pred)
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

        # Amino acid tokens
        if any(aa_mask):
            y_true_aa = y_true_np[aa_mask]
            y_pred_aa = y_pred_np[aa_mask]
            metrics["f1_aa"] = f1_score(y_true_aa, y_pred_aa, average='weighted')
            metrics["prec_aa"] = precision_score(y_true_aa, y_pred_aa, average='weighted')
            metrics["rec_aa"] = recall_score(y_true_aa, y_pred_aa, average='weighted')
            metrics["acc_aa"] = accuracy_score(y_true_aa, y_pred_aa)
            metrics["mcc_aa"] = matthews_corrcoef(y_true_aa, y_pred_aa)
        else:
            metrics["f1_aa"] = metrics["prec_aa"] = metrics["rec_aa"] = metrics["acc_aa"] = metrics["mcc_aa"] = float('nan')

        # Foldseek tokens
        if any(fs_mask):
            y_true_fs = y_true_np[fs_mask]
            y_pred_fs = y_pred_np[fs_mask]
            metrics["f1_fs"] = f1_score(y_true_fs, y_pred_fs, average='weighted')
            metrics["prec_fs"] = precision_score(y_true_fs, y_pred_fs, average='weighted')
            metrics["rec_fs"] = recall_score(y_true_fs, y_pred_fs, average='weighted')
            metrics["acc_fs"] = accuracy_score(y_true_fs, y_pred_fs)
            metrics["mcc_fs"] = matthews_corrcoef(y_true_fs, y_pred_fs)
        else:
            metrics["f1_fs"] = metrics["prec_fs"] = metrics["rec_fs"] = metrics["acc_fs"] = metrics["mcc_fs"] = float('nan')

        del lm_logits, labels, lm_logits_torch, labels_torch
        torch.cuda.empty_cache()
        return metrics


class PairDataset(TorchDataset):
    def __init__(
            self,
            data,
            a_col: str = 'seqs',
            b_col: str = 'labels',
            aa_token: str = '<aa>',
            fs_token: str = '<fs>',
            **kwargs
    ):
        self.seqs_a = data[a_col]
        self.seqs_b = data[b_col]
        self.aa_token = aa_token
        self.fs_token = fs_token

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        seq_a = self.seqs_a[idx]
        seq_b = self.seqs_b[idx]
        seq_a = self.aa_token + seq_a
        seq_b = self.fs_token + seq_b
        if random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a
        return seq_a, seq_b


class PairCollator_input_ids:
    def __init__(self, tokenizer, max_length=2048, **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b = zip(*batch)
        
        if random.random() < 0.9:
            seqs = [a + '<sep>' + b for a, b in zip(seqs_a, seqs_b)]
        else:
            seqs = seqs_a
        
        tokenized = self.tokenizer(
            seqs,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True,
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
        }


def map_token_embedding_matrix(old_tokenizer, new_tokenizer, model):
    # extend from vocabulary 33 -> 64
    new_model = copy.deepcopy(model)
    hidden_size = model.config.hidden_size
    new_model.esm.embeddings.word_embeddings.weight = torch.nn.Parameter(torch.randn(64, hidden_size))
    new_model.lm_head.decoder.weight = torch.nn.Parameter(torch.randn(64, hidden_size))
    new_model.lm_head.decoder.bias = torch.nn.Parameter(torch.randn(64))
    new_model.vocab_size = 64
    new_model.config.vocab_size = 64
    map_from_cls = ['<bos>', '<sep>', '<aa>', '<fs>']
    with torch.no_grad():
        for i in range(33):
            for j in range(64):
                old_token = old_tokenizer.decode(i).lower()
                new_token = new_tokenizer.decode(j).lower()
                if old_token == new_token:
                    print(f"Mapping {old_token} {i} to {new_token} {j}")
                    new_model.esm.embeddings.word_embeddings.weight[j] = model.esm.embeddings.word_embeddings.weight[i]
                    new_model.lm_head.decoder.weight[j] = model.lm_head.decoder.weight[i]
                    new_model.lm_head.decoder.bias[j] = model.lm_head.decoder.bias[i]
                elif old_token == '<cls>' and new_token in map_from_cls:
                    print(f"Mapping <cls> to {new_token} {j}")
                    new_model.esm.embeddings.word_embeddings.weight[j] = model.esm.embeddings.word_embeddings.weight[i]
                    new_model.lm_head.decoder.weight[j] = model.lm_head.decoder.weight[i]
                    new_model.lm_head.decoder.bias[j] = model.lm_head.decoder.bias[i]
    return new_model


def parse_args():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--model_path", type=str, default="GleghornLab/DSM_150", help="Path to the model to train")
    parser.add_argument("--save_path", type=str, default="lhallee/DSM_150_fs", help="Path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--wandb_project", type=str, default="DSM", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of sequences fed to the model")
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for training")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    args = parser.parse_args()
    return args


def main(args):
    ### Load model
    model = DSM.from_pretrained(args.model_path)
    old_tokenizer = model.tokenizer
    tokenizer = EsmTokenizer.from_pretrained("lhallee/joint_tokenizer")
    model = map_token_embedding_matrix(old_tokenizer, tokenizer, model)
    model.tokenizer = tokenizer
    model.get_special_token_ids()
    print(model)

    ### Load Dataset
    dataset = load_dataset('lhallee/foldseek_dataset')
    dataset = dataset.filter(lambda x: len(x['seqs']) <= args.max_length // 2)
    train_dataset = dataset['train']
    valid_dataset = dataset['valid']
    test_dataset = dataset['test']

    if args.bugfix:
        train_dataset = train_dataset.select(range(10000))
        valid_dataset = valid_dataset.select(range(10))
        test_dataset = test_dataset.select(range(10))

    # the labels are not actually used, we include them to play nice with existing collators
    train_dataset = PairDataset(train_dataset)
    valid_dataset = PairDataset(valid_dataset)
    test_dataset = PairDataset(test_dataset)
    data_collator = PairCollator_input_ids(tokenizer, args.max_length)

    ### Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_path.split('/')[-1],
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.save_every,
        eval_steps=args.save_every,
        warmup_steps=args.save_every,
        logging_dir="./logs",
        learning_rate=args.lr,
        fp16=args.fp16,
        dataloader_num_workers=4 if not args.bugfix else 0,
        report_to="wandb" if WANDB_AVAILABLE else 'none',
        save_total_limit=5,
        max_grad_norm=10.0,
        label_names=['input_ids'],
        hub_always_push=False if args.bugfix else True,
        push_to_hub=True,
        save_only_model=True,
        hub_strategy='every_save',
        hub_model_id=args.save_path,
        hub_private_repo=True,
    )

    ### Create a trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=ComputeMetrics(tokenizer),
    )

    ### Train
    metrics = trainer.evaluate(test_dataset)
    print('Initial Metrics: \n', metrics)
    trainer.train()
    metrics = trainer.evaluate(test_dataset)
    print('Final Metrics: \n', metrics)
    trainer.model.push_to_hub(args.save_path, private=True)
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    # py -m train_dsm_ppi_both
    args = parse_args()

    if WANDB_AVAILABLE:
        run_name = args.save_path.split('/')[-1]
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if args.token is not None:
        login(args.token)    

    if args.bugfix:
        args.batch_size = 2
        args.max_length = 256
        args.save_every = 10

    main(args)
