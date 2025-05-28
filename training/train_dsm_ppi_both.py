#! /usr/bin/env python3
# py -m train_dsm_ppi_both
import argparse
import torch
import torch.nn.functional as F
import random
from torch.utils.data import Dataset as TorchDataset
from torchinfo import summary
from transformers import TrainingArguments, EvalPrediction, Trainer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from huggingface_hub import login
from datasets import load_dataset

from models.modeling_dsm import DSM
from data.data_collators import PairCollator_input_ids


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


def compute_dsm_metrics(eval_preds: EvalPrediction):
    ### NOTE the eval mask percentage is fixed at 15%
    metrics = {}
    lm_logits = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
    lm_logits, labels = lm_logits

    # labels are already -100 for non-masked tokens
    lm_logits_torch = torch.tensor(lm_logits)
    labels_torch = torch.tensor(labels)
    # We need ot do this because the eval loss is scaled by the mask rate
    cross_entropy_loss = F.cross_entropy(
        lm_logits_torch.view(-1, lm_logits_torch.shape[-1]), 
        labels_torch.view(-1),
        ignore_index=-100
    )

    metrics['cross_entropy_loss'] = cross_entropy_loss

    y_pred = lm_logits.argmax(axis=-1).flatten()
    y_true = labels.flatten()
    valid_indices = y_true != -100
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    metrics["f1"] = f1
    metrics["prec"] = prec
    metrics["rec"] = rec
    metrics["acc"] = acc
    metrics["mcc"] = mcc

    del lm_logits, labels, lm_logits_torch, labels_torch
    torch.cuda.empty_cache()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--model_path", type=str, default="GleghornLab/DSM_650", help="Path to the model to train")
    parser.add_argument("--save_path", type=str, default="lhallee/DSM_650_ppi_both", help="Path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of steps to train for")
    parser.add_argument("--wandb_project", type=str, default="DSM", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of sequences fed to the model")
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for training")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    args = parser.parse_args()
    return args


def main(args):
    ### Load model
    model = DSM.from_pretrained(args.model_path)
    tokenizer = model.tokenizer
    summary(model)

    ### Load Dataset
    seq_dict = load_dataset('Synthyra/StringDBSeqsv12_unique', split='train')
    seq_dict = dict(zip(seq_dict['id'], seq_dict['sequence']))

    train_dataset = load_dataset("Synthyra/all_string_pairs_900_unique", split='train')

    def filter_func(x):
        id_a, id_b = x['pairs'].split('|')
        seq_a = seq_dict[id_a]
        seq_b = seq_dict[id_b]
        return len(seq_a) > 20 and len(seq_b) > 20 and len(seq_a) + len(seq_b) < args.max_length

    train_dataset = train_dataset.filter(filter_func)
    train_dataset = train_dataset.shuffle(seed=42)

    ### make random eval split
    train_dataset, valid_dataset = train_dataset.train_test_split(test_size=1000, seed=42)
    
    if args.bugfix:
        train_dataset = train_dataset.select(range(10000))
        valid_dataset = valid_dataset.select(range(10))


    class PairDatasetTrainHF(TorchDataset):
        def __init__(self, data, pairs_col: str = 'pairs', label_col: str = 'score', **kwargs):
            self.pairs = data[pairs_col]
            self.labels = data[label_col]

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            pair = self.pairs[idx]
            id_a, id_b = pair.split('|')
            seq_a = seq_dict[id_a]
            seq_b = seq_dict[id_b]
            if random.random() < 0.5:
                seq_a, seq_b = seq_b, seq_a
            return seq_a, seq_b, self.labels[idx]
        

    class PairDatasetTestHF(TorchDataset):
        def __init__(self, data, pairs_col: str = 'pairs', label_col: str = 'score', **kwargs):
            self.pairs = data[pairs_col]
            self.labels = data[label_col]

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            pair = self.pairs[idx]
            id_a, id_b = pair.split('|')
            seq_a = seq_dict[id_a]
            seq_b = seq_dict[id_b]
            return seq_a, seq_b, self.labels[idx]

    # the labels are not actually used, we include them to play nice with existing collators
    train_dataset = PairDatasetTrainHF(train_dataset)
    valid_dataset = PairDatasetTestHF(valid_dataset)
    test_dataset = PairDatasetTestHF(test_dataset)
    data_collator = PairCollator_input_ids(tokenizer, args.max_length)

    ### Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_path.split('/')[-1],
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
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
        save_total_limit=3,
        max_grad_norm=10.0,
        label_names=['input_ids'],
    )

    ### Create a trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_dsm_metrics,
    )

    ### Train
    metrics = trainer.evaluate(valid_dataset)
    print('Initial Metrics: \n', metrics)
    trainer.train()
    metrics = trainer.evaluate(valid_dataset)
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
