import argparse
import pickle
from torchinfo import summary
from huggingface_hub import login, hf_hub_download
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from data.dataset_classes import DiffATDataset
from data.data_collators import DiffATCollator
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
    parser.add_argument('--dataset_name', type=str, default='lhallee/AV_large')
    parser.add_argument('--save_path', type=str, default='Synthyra/esm_diff_at')
    parser.add_argument('--wandb_project', type=str, default='ESM-Diff')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--eval_size', type=int, default=10000)
    parser.add_argument('--bugfix', action='store_true')
    parser.add_argument('--max_ann_length', type=int, default=64)
    return parser.parse_args()


def main(args):
    data = load_dataset(args.dataset_name, split='train')
    #data = data.map(lambda x: {'annotations': ast.literal_eval(x['annotations'])})
    local_file = hf_hub_download(
        repo_id="lhallee/AV_large",
        filename=f"id2label.pkl",
        repo_type="dataset"
    )
    with open(local_file, 'rb') as f:
        id2label = pickle.load(f)
    num_annotations = len(id2label)
    print(f'Number of annotations: {num_annotations}')
    print(f'Before filtering: {len(data)}')
    data = data.filter(
        lambda x: len(x['annotations']) > 4 and len(x['annotations']) < args.max_ann_length and len(x['sequence']) < args.max_length and len(x['sequence']) > 20
    )
    print(f'After filtering: {len(data)}')

    at_vocab_size = num_annotations + 3
    print(f'AT vocab size: {at_vocab_size}')

    train, valid, test = split_dataset(data, args.eval_size)

    train_dataset = DiffATDataset(train)
    valid_dataset = DiffATDataset(valid)
    test_dataset = DiffATDataset(test)

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
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.eval_steps,
        logging_dir="./logs",
        learning_rate=args.lr,
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
    # py -m train_esm_diff_av
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

