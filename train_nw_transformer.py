import torch
import argparse
import random
from torchinfo import summary
from typing import List, Tuple, Dict
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, EsmTokenizer
from huggingface_hub import login
from datasets import load_dataset

from metrics.regression import compute_metrics_regression
from models.alignment_helpers import AlignmentScorer
from models.modeling_nw_transformer import NWTransformerFull, NWTransformerCross, NWTransformerConfig

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class NWDataset(TorchDataset):
    def __init__(self, dataset, sequence_col: str = 'Sequence'):
        self.sequences = list(set(dataset[sequence_col]))

    def __len__(self):
        return len(self.sequences)

    def _mutate_seq(self, seq: str) -> str:
        # pick to random indicies, then shuffle in between
        if len(seq) < 3:
            return seq
        
        idx1 = random.randint(0, len(seq) - 3)
        idx2 = random.randint(idx1 + 2, len(seq) - 1)
        
        # Extract the segment to shuffle
        segment = list(seq[idx1:idx2])
        random.shuffle(segment)
        shuffled_segment = ''.join(segment)
        # Reconstruct the sequence with the shuffled segment
        return seq[:idx1] + shuffled_segment + seq[idx2:]
        
    def __getitem__(self, idx):
        seq_a = random.choice(self.sequences)
        if random.random() < 0.5:
            seq_b = random.choice(self.sequences)
        else:
            seq_b = self._mutate_seq(seq_a)
        
        if random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a

        return seq_a, seq_b


class NWCollatorFull:
    def __init__(self, tokenizer, max_length=2048, asinh=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scorer = AlignmentScorer()
        self.asinh = asinh

    def __call__(self, batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b = zip(*batch)
        
        # Truncate sequences if their combined length exceeds max_length
        truncated_seqs_a = []
        truncated_seqs_b = []
        for seq_a, seq_b in zip(seqs_a, seqs_b):
            # Make copies to avoid modifying the original sequences
            trunc_a, trunc_b = seq_a, seq_b
            
            # Use a while loop to gradually truncate sequences
            while len(trunc_a) + len(trunc_b) > self.max_length:
                # Determine which sequence is longer
                if len(trunc_a) > len(trunc_b):
                    # Remove two characters from the longer sequence
                    trunc_a = trunc_a[:-2]
                elif len(trunc_b) > len(trunc_a):
                    # Remove two characters from the longer sequence
                    trunc_b = trunc_b[:-2]
                else:
                    # If both sequences are the same length, remove from both
                    trunc_a = trunc_a[:-2]
                    trunc_b = trunc_b[:-2]
            
            truncated_seqs_a.append(trunc_a)
            truncated_seqs_b.append(trunc_b)
        
        # Calculate NW scores using the truncated sequences
        labels = [self.scorer(seq_a, seq_b) for seq_a, seq_b in zip(truncated_seqs_a, truncated_seqs_b)]
        labels = torch.tensor(labels, dtype=torch.float32)
        if self.asinh:
            labels = torch.asinh(labels)
        
        # Tokenize the truncated sequences
        tokenized = self.tokenizer(
            truncated_seqs_a, truncated_seqs_b,
            padding='longest',
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }


class NWCollatorCross:
    def __init__(self, tokenizer, max_length=2048, asinh=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scorer = AlignmentScorer()
        self.asinh = asinh

    def __call__(self, batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b = zip(*batch)
        
        # Truncate sequences if their combined length exceeds max_length
        truncated_seqs_a = []
        truncated_seqs_b = []
        for seq_a, seq_b in zip(seqs_a, seqs_b):
            # Make copies to avoid modifying the original sequences
            trunc_a, trunc_b = seq_a, seq_b
            
            # Use a while loop to gradually truncate sequences
            while len(trunc_a) + len(trunc_b) > self.max_length:
                # Determine which sequence is longer
                if len(trunc_a) > len(trunc_b):
                    # Remove two characters from the longer sequence
                    trunc_a = trunc_a[:-2]
                elif len(trunc_b) > len(trunc_a):
                    # Remove two characters from the longer sequence
                    trunc_b = trunc_b[:-2]
                else:
                    # If both sequences are the same length, remove from both
                    trunc_a = trunc_a[:-2]
                    trunc_b = trunc_b[:-2]
            
            truncated_seqs_a.append(trunc_a)
            truncated_seqs_b.append(trunc_b)
        
        # Calculate NW scores using the truncated sequences
        labels = [self.scorer(seq_a, seq_b) for seq_a, seq_b in zip(truncated_seqs_a, truncated_seqs_b)]
        labels = torch.tensor(labels, dtype=torch.float32)
        if self.asinh:
            labels = torch.asinh(labels)
        
        # Tokenize the truncated sequences
        tokenized_a = self.tokenizer(
            truncated_seqs_a,
            padding='longest',
            return_tensors='pt'
        )
        tokenized_b = self.tokenizer(
            truncated_seqs_b,
            padding='longest',
            return_tensors='pt'
        )
        return {
            'input_ids_a': tokenized_a['input_ids'],
            'attention_mask_a': tokenized_a['attention_mask'],
            'input_ids_b': tokenized_b['input_ids'],
            'attention_mask_b': tokenized_b['attention_mask'],
            'labels': labels
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--save_path", type=str, default="GleghornLab/AlignmentTransformer", help="Path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--wandb_project", type=str, default="nw-transformer", help="Wandb project name")
    parser.add_argument("--dataset_name", type=str, default="Synthyra/SwissProt", help="Name of the dataset to use for training")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum length of sequences fed to the model")
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for training")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of the model")
    parser.add_argument("--head_dim", type=int, default=256, help="Head dimension of the model")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--model_type", type=str, default="cross", help="Type of model to train")
    parser.add_argument("--pooling_type", type=str, default="max", help="Type of pooling to use")
    parser.add_argument("--asinh", action="store_true", help="Use asinh transform for labels")
    args = parser.parse_args()
    return args


def main(args):
    ### Load Model
    MODEL_CLASS = NWTransformerFull if args.model_type == "full" else NWTransformerCross
    COLLATOR_CLASS = NWCollatorFull if args.model_type == "full" else NWCollatorCross
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = MODEL_CLASS(NWTransformerConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        n_heads=2,
        n_layers=1,
        head_dim=args.head_dim,
        pooling_type=args.pooling_type
    ))
    summary(model)
    ### Load Dataset
    train_dataset = load_dataset(args.dataset_name, split='train').shuffle(seed=42)
    if args.bugfix:
        train_dataset = train_dataset.filter(lambda x: len(x['Sequence']) < args.max_length // 2)
    train_dataset = train_dataset.train_test_split(test_size=2000)
    valid_dataset = train_dataset['test']
    train_dataset = train_dataset['train']
    valid_dataset = valid_dataset.train_test_split(test_size=1000)
    test_dataset = valid_dataset['test']
    valid_dataset = valid_dataset['train']

    train_dataset = NWDataset(train_dataset)
    valid_dataset = NWDataset(valid_dataset)
    test_dataset = NWDataset(test_dataset)

    data_collator = COLLATOR_CLASS(tokenizer, args.max_length, args.asinh)

    ### Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_path.split('/')[-1],
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        logging_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.save_every,
        eval_steps=args.save_every,
        warmup_steps=args.save_every // 5,
        logging_dir="./logs",
        learning_rate=args.lr,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers if not args.bugfix else 0,
        report_to="wandb" if WANDB_AVAILABLE else 'none',
        save_only_model=True,
        hub_strategy='every_save',
        hub_model_id=args.save_path,
        hub_private_repo=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
    )

    ### Create a trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_regression, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    ### Train
    metrics = trainer.evaluate(test_dataset)
    print('Initial Metrics: \n', metrics)
    trainer.train()
    metrics = trainer.evaluate(test_dataset)
    print('Final Metrics: \n', metrics)
    trainer.push_to_hub()
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    # py -m train_nw_transformer
    args = parse_args()

    if WANDB_AVAILABLE:
        run_name = args.save_path.split('/')[-1]
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if args.token is not None:
        login(args.token)    

    if args.bugfix:
        args.batch_size = 2
        args.max_length = 256
        args.save_every = 1000

    main(args)
