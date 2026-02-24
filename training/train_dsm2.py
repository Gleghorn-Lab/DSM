#! /usr/bin/env python3
# py -m training.train_dsm2
import entrypoint_setup

import os
import argparse
import torch
import torch.nn.functional as F
from torchinfo import summary
from transformers import TrainingArguments, EvalPrediction, TrainerCallback
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from huggingface_hub import login, hf_hub_download
from datasets import load_dataset, Dataset
import math

from models.modeling_dsm2 import DSM2
from models.alignment_helpers import GetAlignmentScoreFromLogits
from data.dataset_classes import SequenceDatasetFromList
from data.data_collators import SequenceCollator
from training.iterable_trainer import get_iterable_trainer

from models.FastPLMs.esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM
from models.FastPLMs.esm2.modeling_fastesm import FastEsmForMaskedLM
from models.FastPLMs.dplm_fastplms.modeling_dplm import DPLMForMaskedLM
from models.FastPLMs.dplm2_fastplms.modeling_dplm2 import DPLM2ForMaskedLM

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_teacher(teacher_path: str, device: str = "cuda"):
    model_lower = teacher_path.lower()
    print(f"Loading Teacher Model from {teacher_path}...")
    
    if "dplm2" in model_lower:
        teacher = DPLM2ForMaskedLM.from_pretrained(teacher_path)
    elif "dplm" in model_lower:
        teacher = DPLMForMaskedLM.from_pretrained(teacher_path)
    elif "esm2" in model_lower or "fastesm" in model_lower:
        teacher = FastEsmForMaskedLM.from_pretrained(teacher_path)
    else:
        # Default fallback is ESM++
        teacher = ESMplusplusForMaskedLM.from_pretrained(teacher_path)
        
    teacher.attn_backend = "flex"
    teacher = teacher.to(torch.bfloat16).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    return teacher


def compute_dsm2_metrics(eval_preds: EvalPrediction):
    ### NOTE the eval mask percentage is fixed at 15%
    metrics = {}
    
    # The Trainer might return a tuple of all fields from DSM2Output, or just the logits.
    # We search for the 3D tensor with the expected vocab size.
    preds = eval_preds.predictions
    lm_logits = None
    mask_labels = None
    
    if isinstance(preds, (tuple, list)):
        for p in preds:
            if hasattr(p, "shape"):
                if len(p.shape) == 3 and p.shape[-1] < 100: # Heuristic for vocab size
                    lm_logits = p
                elif len(p.shape) == 2 and mask_labels is None:
                    mask_labels = p
    else:
        lm_logits = preds

    if lm_logits is None:
        return {}

    # input_ids is the original sequence (used as label_ids in Trainer)
    input_ids = eval_preds.label_ids[0] if isinstance(eval_preds.label_ids, tuple) else eval_preds.label_ids

    # For cross entropy, we prefer using the mask_labels if provided by the model
    # mask_labels has -100 for non-masked tokens.
    if mask_labels is None:
        # Fallback: if we don't have mask_labels, we use the original input_ids
        # but cross_entropy will calculate loss over ALL tokens, which is not ideal for MLM.
        labels_to_use = input_ids
    else:
        labels_to_use = mask_labels

    scores = GetAlignmentScoreFromLogits().batched_call(lm_logits, input_ids)

    lm_logits_torch = torch.as_tensor(lm_logits)
    labels_torch = torch.as_tensor(labels_to_use).long()
    
    # We need to do this because the eval loss is scaled by the mask rate
    cross_entropy_loss = F.cross_entropy(
        lm_logits_torch.view(-1, lm_logits_torch.shape[-1]), 
        labels_torch.view(-1),
        ignore_index=-100
    )

    metrics['cross_entropy_loss'] = cross_entropy_loss.item()
    metrics['alignment_score'] = scores.mean()

    # Calculate other metrics only on valid (non -100) tokens
    y_pred = lm_logits.argmax(axis=-1).flatten()
    y_true = labels_to_use.flatten()
    valid_indices = y_true != -100
    
    if valid_indices.any():
        y_pred = y_pred[valid_indices]
        y_true = y_true[valid_indices]
        metrics["f1"] = f1_score(y_true, y_pred, average='weighted')
        metrics["prec"] = precision_score(y_true, y_pred, average='weighted')
        metrics["rec"] = recall_score(y_true, y_pred, average='weighted')
        metrics["acc"] = accuracy_score(y_true, y_pred)
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    else:
        # Fallback if no tokens are masked (shouldn't happen with 15% rate)
        metrics["acc"] = 0.0

    return metrics


class DynamicLengthCallback(TrainerCallback):
    """
    Linearly scales the maximum sequence length used by the collator from start_len to end_len
    over the course of training in `interval` buckets.
    """
    def __init__(self, data_collator, total_steps: int, start_len: int = 128, end_len: int = 2048, interval: int = 64):
        self.data_collator = data_collator
        self.total_steps = total_steps
        self.start_len = start_len
        self.end_len = end_len
        self.interval = interval

    def on_step_begin(self, args, state, control, **kwargs):
        progress = state.global_step / max(1, self.total_steps)
        # Interpolate between start and end
        current_target = self.start_len + (self.end_len - self.start_len) * progress
        # Round down to nearest `interval` multiple
        current_len = max(self.start_len, min(self.end_len, int(math.floor(current_target / self.interval) * self.interval)))
        
        if self.data_collator.max_length != current_len:
            self.data_collator.max_length = current_len
            if wandb is not None and WANDB_AVAILABLE:
                wandb.log({"train/dynamic_max_length": current_len}, step=state.global_step)
            # You can also uncomment this if you want it to print 
            # print(f"Step {state.global_step}: Updated dynamic max token length to {current_len}")


def get_eval_data(data_path: str, bugfix: bool = False):
    local_file = hf_hub_download(
        repo_id=data_path,
        filename=f"data/valid-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=42)
    if bugfix:
        data = data.select(range(10))
    else:
        data = data.select(range(1000))
    print(data)
    valid_seqs = data['sequence']
    local_file = hf_hub_download(
        repo_id=data_path,
        filename=f"data/test-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=42)
    if bugfix:
        data = data.select(range(10))
    else:
        data = data.select(range(1000))
    print(data)
    test_seqs = data['sequence']
    return valid_seqs, test_seqs


def parse_args():
    parser = argparse.ArgumentParser(description="DSM2 Trainer")
    parser.add_argument("--hf_token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--wandb_token", type=str, default=None, help="Wandb token")
    parser.add_argument("--wandb_project", type=str, default="DSM2", help="Wandb project name")
    parser.add_argument("--teacher_model_path", type=str, default="Synthyra/DPLM-3B", help="Path to initialize the teacher model from")
    
    # Student Architecture Arguments
    parser.add_argument("--student_hidden_size", type=int, default=256, help="Hidden size for the student model.")
    parser.add_argument("--student_expansion_ratio", type=float, default=4.0, help="FFN expansion ratio for the student model.")
    
    parser.add_argument("--data_path", type=str, default="Synthyra/uniref50", help="HuggingFace dataset repository containing train, valid and test splits")
    parser.add_argument("--save_path", type=str, default="GleghornLab/DSM2_600", help="Path to save the model and report to wandb")
    
    parser.add_argument("--alpha_ce", type=float, default=1.0, help="Weight for CE Loss")
    parser.add_argument("--alpha_jepa", type=float, default=1.0, help="Weight for JEPA Loss")
    parser.add_argument("--alpha_contrastive", type=float, default=1.0, help="Weight for Contrastive Loss")
    
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of steps to train for (typically 1 epoch)")
    
    parser.add_argument("--start_max_length", type=int, default=128, help="Starting Maximum length of sequences")
    parser.add_argument("--end_max_length", type=int, default=2048, help="Ending Maximum length of sequences")
    parser.add_argument("--len_interval", type=int, default=128, help="Interval by which max length jumps")
    
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size, max length, and fast exit for debugging")
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Load Teacher Model
    teacher_model = load_teacher(args.teacher_model_path, device=device)
    temp_teacher_config = teacher_model.config
    
    ### Load Student Model and Set Settings
    print(f"Initializing Student DSM2 from scratch")
    from models.modeling_dsm2 import DSM2Config
    
    student_config = DSM2Config(
        vocab_size=temp_teacher_config.vocab_size,
        hidden_size=args.student_hidden_size,
        num_attention_heads=args.student_hidden_size // 64,
        num_hidden_layers=temp_teacher_config.num_hidden_layers,
        teacher_hidden_size=temp_teacher_config.hidden_size,
        expansion_ratio=args.student_expansion_ratio,
        attn_backend="flex"
    )
    
    student_model = DSM2(student_config)
    student_model.attn_backend = "flex"
    
    tokenizer = student_model.tokenizer
    
    if args.bugfix:
        # Avoid massive text spam on debugging but show something
        try:
            summary(student_model)
        except:
            pass

    # Compile the models dynamically to handle length changes gracefully
    print("Compiling teacher model (dynamic=True)...")
    try:
        teacher_model = torch.compile(teacher_model, dynamic=True)
    except Exception as e:
        print(f"Warning: Teacher torch.compile(dynamic=True) failed: {e}")
        
    print("Compiling student model (dynamic=True)...")
    try:
        student_model = torch.compile(student_model, dynamic=True)
    except Exception as e:
        print(f"Warning: Student torch.compile(dynamic=True) failed: {e}")

    ### Load Dataset
    train_dataset = load_dataset(args.data_path, split="train", streaming=True).shuffle(seed=42)
    valid_seqs, test_seqs = get_eval_data(args.data_path, bugfix=args.bugfix)
    
    valid_dataset = SequenceDatasetFromList(valid_seqs)
    test_dataset = SequenceDatasetFromList(test_seqs)
    
    # Initialize collator with starting length
    data_collator = SequenceCollator(tokenizer, max_length=args.start_max_length)

    ### Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_path.split('/')[-1],
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
        learning_rate=args.lr,
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=4 if not args.bugfix else 0,
        dataloader_prefetch_factor=2 if not args.bugfix else None,
        report_to="wandb" if os.environ["WANDB_AVAILABLE"] == 'true' and args.wandb_token is not None else 'none',
        save_total_limit=3,
        max_grad_norm=10.0,
        label_names=['input_ids'],
        remove_unused_columns=False,
    )

    ### Create an Iterable Trainer base
    trainer = get_iterable_trainer(
        model=student_model,
        hf_dataset=train_dataset,
        data_collator=data_collator,
        training_args=training_args,
        batch_size=args.batch_size,
        col_name="sequence",
        num_workers=4 if not args.bugfix else 0, # usually 0 is safer on windows
        prefetch_factor=10 if not args.bugfix else None,
        compute_metrics=compute_dsm2_metrics,
        callbacks=[DynamicLengthCallback(
            data_collator=data_collator,
            total_steps=args.max_steps,
            start_len=args.start_max_length,
            end_len=args.end_max_length,
            interval=args.len_interval,
        )],
        eval_dataset=valid_dataset,
    )

    # We patch compute_loss to handle the custom teacher forward logic
    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        # 1. Forward pass through frozen Teacher
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
            )
            # DPLM outputs the embeddings at index 0, hidden states at following indices. Same as ESM/ESM++
            teacher_hidden_states = teacher_outputs.hidden_states
            # Filter first element to match num_hidden_layers logic
            if len(teacher_hidden_states) > teacher_model.config.num_hidden_layers:
                teacher_hidden_states = teacher_hidden_states[1:] 

        # 2. Forward pass through Student DSM2
        dsm2_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            teacher_hidden_states=teacher_hidden_states,
            alpha_ce=args.alpha_ce,
            alpha_jepa=args.alpha_jepa,
            alpha_contrastive=args.alpha_contrastive,
        )
        
        loss = dsm2_output.loss
        
        if trainer.state.global_step % training_args.logging_steps == 0 and trainer.is_world_process_zero():
            logs = {
                "train/ce_loss": dsm2_output.ce_loss.item() if dsm2_output.ce_loss is not None else 0.0,
                "train/jepa_loss": dsm2_output.jepa_loss.item() if dsm2_output.jepa_loss is not None else 0.0,
                "train/contrastive_loss": dsm2_output.contrastive_loss.item() if dsm2_output.contrastive_loss is not None else 0.0,
            }
            if os.environ["WANDB_AVAILABLE"] == 'true' and args.wandb_token is not None:
                wandb.log(logs, step=trainer.state.global_step)
            
        return (loss, dsm2_output) if return_outputs else loss

    trainer.compute_loss = compute_loss

    ### Train
    try:
        metrics = trainer.evaluate(test_dataset)
        print('Initial Metrics: \n', metrics)
    except Exception as e:
        print(f"Initial evaluation failed, moving to training (often caused by strict dtype issues): {e}")

    trainer.train()
    
    try:
        metrics = trainer.evaluate(test_dataset)
        print('Final Metrics: \n', metrics)
    except:
        pass
        
    trainer.model.push_to_hub(args.save_path, private=True)
    if os.environ["WANDB_AVAILABLE"] == 'true' and args.wandb_token is not None:
        wandb.finish()


if __name__ == "__main__":
    # py -m training.train_dsm2
    args = parse_args()

    if os.environ["WANDB_AVAILABLE"] == 'true' and args.wandb_token is not None:
        import wandb
        wandb.login(args.wandb_token)
        run_name = args.save_path.split('/')[-1]
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if args.hf_token is not None:
        login(args.hf_token)    

    if args.bugfix:
        args.batch_size = 2
        args.start_max_length = 16
        args.end_max_length = 64
        args.len_interval = 16
        args.save_every = 10
        args.max_steps = 1000
        args.student_hidden_size = 256
        args.student_expansion_ratio = 2.0
        args.teacher_model_path = "Synthyra/ESM2-8M"

    main(args)
