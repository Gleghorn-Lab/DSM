#! /usr/bin/env python3
# py -m train_dsm
import argparse
import torch
import torch.nn.functional as F
from torchinfo import summary
from transformers import TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from huggingface_hub import login
from datasets import load_dataset

from data.dataset_classes import PairDatasetTrainHF, PairDatasetTestHF
from data.data_collators import PairCollator_input_ids
from models.modeling_dsm import DSM_Binders
from models.utils import wrap_lora
from utils import set_seed

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
    parser.add_argument("--save_path", type=str, default="lhallee/DSM_bind_650", help="Path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--wandb_project", type=str, default="DSM", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of sequences fed to the model")
    parser.add_argument("--save_every", type=int, default=500, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for training")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for training")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.01, help="LoRA dropout")
    args = parser.parse_args()
    return args


def main(args):
    set_seed(42)
    ### Load model
    model = DSM_Binders.from_pretrained(args.model_path)
    if args.lora:
        model = wrap_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    tokenizer = model.tokenizer
    summary(model)
    
    ### Load Dataset
    train_dataset = load_dataset("lhallee/string_model_org_90_90_split")

    train_dataset = train_dataset.filter(
        lambda x: len(x['SeqA']) > 20 and len(x['SeqB']) > 20 and len(x['SeqA']) + len(x['SeqB']) < args.max_length
    )
    train_dataset = train_dataset.shuffle(seed=42)

    valid_dataset = train_dataset['valid'].select(range(1000))
    test_dataset = train_dataset['test']
    train_dataset = train_dataset['train']
    
    if args.bugfix:
        train_dataset = train_dataset.select(range(10000))
        valid_dataset = valid_dataset.select(range(10))
        test_dataset = test_dataset.select(range(10))
    
    # the labels are not actually used, we include them to play nice with existing collators
    train_dataset = PairDatasetTrainHF(train_dataset, col_a='SeqA', col_b='SeqB', label_col='score')
    valid_dataset = PairDatasetTestHF(valid_dataset, col_a='SeqA', col_b='SeqB', label_col='score')
    test_dataset = PairDatasetTestHF(test_dataset, col_a='SeqA', col_b='SeqB', label_col='score')
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
        warmup_steps=args.save_every * 2 if args.lora else args.save_every,
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
    metrics = trainer.evaluate(test_dataset)
    print('Initial Metrics: \n', metrics)
    trainer.train()
    metrics = trainer.evaluate(test_dataset)
    print('Final Metrics: \n', metrics)
    trainer.model.push_to_hub(args.save_path, private=True)
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    # py -m train_dsm_ppi
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
