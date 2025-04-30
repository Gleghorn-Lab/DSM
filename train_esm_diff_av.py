import argparse
import torch
import ast
from torchinfo import summary
from huggingface_hub import login
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from typing import Dict

from torch.utils.data import Dataset as TorchDataset
from typing import List, Dict, Any
from random import shuffle, randint
from metrics.LM import compute_lm_metrics_with_logits

from models.modeling_esm_diff import ESM_Diff_AV, ESMDiffConfig

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class DiffATDataset(TorchDataset):
    def __init__(self, data, max_seq_length: int = 512, max_ann_length: int = 64):
        self.seqs = data['sequence']
        self.annotations = data['annotations']
        self.max_seq_length = max_seq_length
        self.max_ann_length = max_ann_length

    def _get_total_length(self, sequence: str, annotations: List[int]) -> int:
        return len(sequence) + len(annotations)

    def _get_ann(self, idx: int) -> List[int]:
        ann = self.annotations[idx]
        length = randint(8, self.max_ann_length)
        shuffle(ann)
        return sorted(ann[:length])

    def avg(self):
        total_len = sum(self._get_total_length(self.seqs[i], self.anns[i]) for i in range(len(self)))
        return total_len / len(self)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][:self.max_seq_length]
        return seq, self._get_ann(idx)


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


def get_max_from_list_of_lists(lst):
    """
    Given a nested list, return the maximum value in all possible elements
    """
    return max(max(sub_list) for sub_list in lst)


def split_dataset(data, eval_size):
    train = data.train_test_split(test_size=eval_size, seed=42)
    valid = train['test']
    train = train['train']
    valid = valid.train_test_split(test_size=eval_size // 2, seed=42)
    test = valid['test']
    valid = valid['train']
    return train, valid, test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='Synthyra/esm_diff_150')
    parser.add_argument('--dataset_name', type=str, default='lhallee/AV')
    parser.add_argument('--save_path', type=str, default='Synthyra/esm_diff_at')
    parser.add_argument("--wandb_project", type=str, default="esm_diff", help="Wandb project name")
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--eval_size', type=int, default=10000)
    parser.add_argument('--bugfix', action='store_true')
    return parser.parse_args()


def main(args):
    data = load_dataset(args.dataset_name, split='train')
    #data = data.map(lambda x: {'annotations': ast.literal_eval(x['annotations'])})
    num_annotations = get_max_from_list_of_lists(data['annotations'])
    print(num_annotations)
    at_vocab_size = num_annotations + 3
    print(f'Number of annotations: {num_annotations}')
    print(f'AT vocab size: {at_vocab_size}')

    train, valid, test = split_dataset(data, args.eval_size)

    train_dataset = DiffATDataset(train, max_seq_length=args.max_length)
    valid_dataset = DiffATDataset(valid, max_seq_length=args.max_length)
    test_dataset = DiffATDataset(test, max_seq_length=args.max_length)

    config = ESMDiffConfig.from_pretrained(args.model_path)
    config.at_vocab_size = at_vocab_size
    config.at_layers = 1
    model = ESM_Diff_AV.from_pretrained(args.model_path, config=config)
    tokenizer = model.tokenizer
    summary(model)

    data_collator = DiffATCollator(tokenizer, at_vocab_size)

    ### Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_path.split('/')[-1],
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        logging_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.eval_steps,
        logging_dir="./logs",
        learning_rate=args.lr,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4 if not args.bugfix else 0,
        report_to="wandb" if WANDB_AVAILABLE else 'none',
        save_total_limit=3,
        max_grad_norm=10.0,
        label_names=['input_ids'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_lm_metrics_with_logits,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
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
    # py -m trainers.fine_tune_esm_diff
    args = parse_args()

    if WANDB_AVAILABLE:
        run_name = args.save_path.split('/')[-1]
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if args.token is not None:
        login(args.token)    

    if args.bugfix:
        args.batch_size = 2
        args.max_length = 32
        args.save_every = 100

    main(args)

