#! /usr/bin/env python3
# py -m training.train_dsm2
import entrypoint_setup

import os
import argparse
import torch
import torch.nn.functional as F
from torchinfo import summary
from transformers import TrainingArguments, EvalPrediction, TrainerCallback, Trainer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from huggingface_hub import login
from datasets import load_dataset, Dataset
import math

from models.modeling_dsm2 import DSM2, DSM2Config
from models.alignment_helpers import GetAlignmentScoreFromLogits
from data.dataset_classes import SequenceDatasetFromList
from data.data_collators import SequenceCollator

from models.FastPLMs.esm_plusplus.modeling_esm_plusplus import ESMplusplusModel
from models.FastPLMs.esm2.modeling_fastesm import FastEsmModel
from models.FastPLMs.dplm_fastplms.modeling_dplm import DPLMModel
from models.FastPLMs.dplm2_fastplms.modeling_dplm2 import DPLM2Model
from models.muonclip.muonclip import MuonClip


def load_teacher(teacher_path: str, device: str = "cuda"):
    model_lower = teacher_path.lower()
    print(f"Loading Teacher Model from {teacher_path}...")
    
    if "dplm2" in model_lower:
        teacher = DPLM2Model.from_pretrained(
            teacher_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device
        ).eval()
    elif "dplm" in model_lower:
        teacher = DPLMModel.from_pretrained(
            teacher_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device
        ).eval()
    elif "esm2" in model_lower or "fastesm" in model_lower:
        teacher = FastEsmModel.from_pretrained(
            teacher_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device
        ).eval()
    else:
        # Default fallback is ESM++
        teacher = ESMplusplusModel.from_pretrained(
            teacher_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device
        ).eval()
        
    teacher.attn_backend = "flex"
    for param in teacher.parameters():
        param.requires_grad = False
        
    return teacher


class ComputeDSM2Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.alignment_scorer = GetAlignmentScoreFromLogits(tokenizer)
    
    def __call__(self, eval_preds: EvalPrediction):
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

        scores = self.alignment_scorer.batched_call(lm_logits, input_ids)

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


class MuonAdamWWrapper(torch.optim.Optimizer):
    def __init__(self, muonclip, adamw):
        self.muonclip = muonclip
        self.adamw = adamw
        self.defaults = adamw.defaults
        self.param_groups = muonclip.param_groups + adamw.param_groups
        self.state = {}
        for opt in [muonclip, adamw]:
            self.state.update(opt.state)
        self.last_s_max = None
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        if self.last_s_max is not None:
            self.muonclip.step(self.last_s_max)
        else:
            print("Warning: last_s_max is None")
            
        self.adamw.step()
        return loss

    def zero_grad(self, set_to_none=True):
        self.muonclip.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.last_s_max = None


class EMATeacherCallback(TrainerCallback):
    def __init__(self, total_steps: int, ema_start_percent: float, ema_decay: float):
        self.total_steps = total_steps
        self.ema_start_percent = ema_start_percent
        self.ema_decay = ema_decay
        self.ema_active = False

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step >= int(self.total_steps * self.ema_start_percent) and not self.ema_active:
            self.ema_active = True
            print(f"Initializing EMA Teacher at step {state.global_step}")
            import copy
            model = kwargs['model']
            unwrapped = model.module if hasattr(model, 'module') else model
            ema_teacher = copy.deepcopy(unwrapped)
            for param in ema_teacher.parameters():
                param.requires_grad = False
            ema_teacher.eval()
            unwrapped.ema_teacher = ema_teacher

    def on_step_end(self, args, state, control, **kwargs):
        if self.ema_active:
            unwrapped = kwargs['model'].module if hasattr(kwargs['model'], 'module') else kwargs['model']
            ema_teacher = getattr(unwrapped, 'ema_teacher', None)
            if ema_teacher is not None:
                with torch.no_grad():
                    for s_param, t_param in zip(unwrapped.parameters(), ema_teacher.parameters()):
                        t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)


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
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=8, help="Max batch size to put through a single forward pass")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of steps to train for (typically 1 epoch)")
    
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of sequences")
    parser.add_argument("--sliding_window_size", type=int, default=512, help="Size of sliding window for attention")
    parser.add_argument("--dilation", type=int, default=16, help="Dilation factor for attention")
    
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--ema_start_percent", type=float, default=0.25, help="Percentage of steps before EMA teacher is initialized.")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Exponential moving average decay factor for teacher.")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size, max length, and fast exit for debugging")
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Load Teacher Model
    teacher_model = load_teacher(args.teacher_model_path, device=device)
    temp_teacher_config = teacher_model.config
    tokenizer = teacher_model.tokenizer
    summary(teacher_model)

    ### Load Student Model and Set Settings
    print(f"Initializing Student DSM2 from scratch")
    
    
    student_config = DSM2Config(
        vocab_size=temp_teacher_config.vocab_size,
        hidden_size=args.student_hidden_size,
        num_attention_heads=args.student_hidden_size // 64,
        num_hidden_layers=temp_teacher_config.num_hidden_layers,
        teacher_hidden_size=temp_teacher_config.hidden_size,
        expansion_ratio=args.student_expansion_ratio,
        attn_backend="flex",
        sliding_window_size=args.sliding_window_size,
        dilation=args.dilation,
    )
    
    student_model = DSM2(student_config).to(torch.bfloat16)
    student_model.attn_backend = "flex"
    summary(student_model)

    print("Compiling teacher model...")
    try:
        teacher_model = torch.compile(teacher_model)
    except Exception as e:
        print(f"Warning: Teacher torch.compile() failed: {e}")
        
    print("Compiling student model...")
    try:
        student_model = torch.compile(student_model)
    except Exception as e:
        print(f"Warning: Student torch.compile() failed: {e}")

    ### Load Dataset
    hf_dataset = load_dataset(args.data_path)
    print('Loading and shuffling training dataset')
    hf_train_dataset = hf_dataset['train'].shuffle(seed=42)
    print('Loading and shuffling validation dataset')
    hf_valid_dataset = hf_dataset['valid'].shuffle(seed=42)
    print('Loading and shuffling test dataset')
    hf_test_dataset = hf_dataset['test'].shuffle(seed=42)
    print('Trimming datasets')
    if args.bugfix:
        hf_train_dataset = hf_train_dataset.select(range(100))
        hf_valid_dataset = hf_valid_dataset.select(range(10))
        hf_test_dataset = hf_test_dataset.select(range(10))
    else:
        hf_train_dataset = hf_train_dataset.select(range(int(1e5)))
        hf_valid_dataset = hf_valid_dataset.select(range(int(1000)))
        hf_test_dataset = hf_test_dataset.select(range(int(1000)))
        
    print('Converting datasets to lists')
    train_seqs = list(hf_train_dataset['sequence'])
    valid_seqs = list(hf_valid_dataset['sequence'])
    test_seqs = list(hf_test_dataset['sequence'])

    print('Creating torch datasets')
    train_dataset = SequenceDatasetFromList(train_seqs)
    valid_dataset = SequenceDatasetFromList(valid_seqs)
    test_dataset = SequenceDatasetFromList(test_seqs)
    
    # Initialize collator with starting length
    data_collator = SequenceCollator(tokenizer, max_length=args.max_length)

    print('Initializing trainer')

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

    ### Create a Trainer base
    class DSM2Trainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
                
                muon_params = []
                adamw_params = []
                attn_params = []
                
                for n, p in unwrapped.named_parameters():
                    if p.ndim >= 2 and 'embed' not in n and 'lm_head' not in n:
                        muon_params.append(p)
                    else:
                        adamw_params.append(p)
                
                for block in unwrapped.transformer.blocks:
                    attn_params.append(block.attn)
                
                import torch.distributed as dist
                is_ddp = dist.is_initialized()
                rank = dist.get_rank() if is_ddp else 0
                world_size = dist.get_world_size() if is_ddp else 1
                
                muonclip = MuonClip(
                    params=muon_params,
                    attention_params=attn_params,
                    mode='mha',
                    metadata={'w_q': 'W_q', 'w_k': 'W_k'},
                    n_head=unwrapped.config.num_attention_heads,
                    tau=100.0,
                    lr=0.02, # Default muon LR
                    rank=rank,
                    world_size=world_size
                )
                
                adamw = torch.optim.AdamW(adamw_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
                self.optimizer = MuonAdamWWrapper(muonclip, adamw)
            return self.optimizer


    trainer = DSM2Trainer(
        model=student_model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=ComputeDSM2Metrics(tokenizer),
        callbacks=[
            EMATeacherCallback(
                total_steps=args.max_steps,
                ema_start_percent=args.ema_start_percent,
                ema_decay=args.ema_decay
            )
        ],
        eval_dataset=valid_dataset,
    )

    print('Trainer initialized')

    # We patch compute_loss to handle the custom teacher forward logic
    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        from models.modeling_dsm2 import pool_states, contrastive_loss_from_pooled

        batch_size = input_ids.size(0)
        patch_size = args.patch_size if args.patch_size > 0 else batch_size

        total_ce_loss = 0.0
        total_jepa_loss = 0.0
        total_contrastive_loss = 0.0

        all_teacher_pooled = []
        all_student_pooled = []
        all_s_max_patches = []
        
        unwrapped_model = model.module if hasattr(model, 'module') else model
        active_teacher = getattr(unwrapped_model, 'ema_teacher', teacher_model)

        # Iterate over patches
        for start_idx in range(0, batch_size, patch_size):
            end_idx = min(start_idx + patch_size, batch_size)
            patch_input_ids = input_ids[start_idx:end_idx]
            patch_attention_mask = attention_mask[start_idx:end_idx]
            current_patch_size = end_idx - start_idx
            
            # 1. Forward pass through frozen Teacher
            with torch.no_grad():
                if active_teacher is teacher_model:
                    teacher_outputs = teacher_model(
                        input_ids=patch_input_ids,
                        attention_mask=patch_attention_mask,
                        output_hidden_states=True,
                        output_attentions=False,
                    )
                    # DPLM outputs the embeddings at index 0, hidden states at following indices. Same as ESM/ESM++
                    teacher_hidden_states = teacher_outputs.hidden_states
                    # Filter first element to match num_hidden_layers logic
                    if len(teacher_hidden_states) > teacher_model.config.num_hidden_layers:
                        teacher_hidden_states = teacher_hidden_states[1:] 
                else:
                    teacher_outputs = active_teacher(
                        input_ids=patch_input_ids,
                        attention_mask=patch_attention_mask,
                        alpha_ce=0.0,
                        alpha_jepa=0.0,
                        alpha_contrastive=0.0,
                    )
                    teacher_hidden_states = teacher_outputs.student_hidden_states

            # 2. Forward pass through Student DSM2
            # We enforce alpha_contrastive=0.0 during patch forward to avoid redundant calculation
            dsm2_patch_output = model(
                input_ids=patch_input_ids,
                attention_mask=patch_attention_mask,
                teacher_hidden_states=teacher_hidden_states,
                alpha_ce=args.alpha_ce,
                alpha_jepa=args.alpha_jepa,
                alpha_contrastive=0.0,
            )
            
            weight = current_patch_size / batch_size
            total_ce_loss += (dsm2_patch_output.ce_loss * weight) if dsm2_patch_output.ce_loss is not None else 0.0
            total_jepa_loss += (dsm2_patch_output.jepa_loss * weight) if dsm2_patch_output.jepa_loss is not None else 0.0

            # Last step variables for potential logging/returning
            dsm2_output = dsm2_patch_output

            if args.alpha_contrastive > 0.0:
                with torch.no_grad():
                    teacher_pooled = pool_states(teacher_hidden_states)
                student_pooled = pool_states(dsm2_patch_output.student_hidden_states)
                
                all_teacher_pooled.append(teacher_pooled)
                all_student_pooled.append(student_pooled)
                
            if getattr(dsm2_patch_output, 's_max', None) is not None:
                all_s_max_patches.append(dsm2_patch_output.s_max)

        # 3. Compute contrastive loss over the entire aggregated batch
        if args.alpha_contrastive > 0.0 and len(all_teacher_pooled) > 0:
            # teacher_pooled: (num_layers, patch_size, 2d)
            stacked_teacher_pooled = torch.cat(all_teacher_pooled, dim=1)
            stacked_student_pooled = torch.cat(all_student_pooled, dim=1)
            
            contrastive_val = contrastive_loss_from_pooled(
                s_pooled=stacked_student_pooled,
                t_pooled=stacked_teacher_pooled,
            )
            total_contrastive_loss = contrastive_val

        # Reduce s_max
        if len(all_s_max_patches) > 0:
            num_layers = len(all_s_max_patches[0])
            num_heads = len(all_s_max_patches[0][0])
            reduced_s_max = []
            for l in range(num_layers):
                layer_maxes = []
                for h in range(num_heads):
                    max_val = max(patch[l][h] for patch in all_s_max_patches)
                    layer_maxes.append(max_val)
                reduced_s_max.append(layer_maxes)
            
            if hasattr(trainer, 'optimizer') and hasattr(trainer.optimizer, 'last_s_max'):
                trainer.optimizer.last_s_max = reduced_s_max

        loss = (args.alpha_ce * total_ce_loss) + (args.alpha_jepa * total_jepa_loss) + (args.alpha_contrastive * total_contrastive_loss)
        
        if trainer.state.global_step % training_args.logging_steps == 0 and trainer.is_world_process_zero():
            logs = {
                "train/ce_loss": total_ce_loss.item() if isinstance(total_ce_loss, torch.Tensor) else total_ce_loss,
                "train/jepa_loss": total_jepa_loss.item() if isinstance(total_jepa_loss, torch.Tensor) else total_jepa_loss,
                "train/contrastive_loss": total_contrastive_loss.item() if isinstance(total_contrastive_loss, torch.Tensor) else total_contrastive_loss,
            }
            if os.environ["WANDB_AVAILABLE"] == 'true' and args.wandb_token is not None:
                wandb.log(logs, step=trainer.state.global_step)
            
        return (loss, dsm2_output) if return_outputs else loss

    trainer.compute_loss = compute_loss

    print('Initial Evaluation')
    metrics = trainer.evaluate(test_dataset)
    print('Initial Metrics: \n', metrics)

    print('Training')
    trainer.train()
    
    print('Final Evaluation')
    metrics = trainer.evaluate(test_dataset)
    print('Final Metrics: \n', metrics)
    
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
        args.batch_size = 4
        args.patch_size = 2
        args.start_max_length = 16
        args.end_max_length = 64
        args.len_interval = 16
        args.save_every = 10
        args.max_steps = 20
        args.student_hidden_size = 256
        args.student_expansion_ratio = 2.0
        args.teacher_model_path = "Synthyra/ESM2-8M"
        args.ema_start_percent = 0.5

    main(args)
