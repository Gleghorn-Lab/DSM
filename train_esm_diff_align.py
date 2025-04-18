#! /usr/bin/env python3
# py -m trainers.train_esm_diff_align
import os
import argparse
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm, trange
from transformers import get_cosine_schedule_with_warmup, EvalPrediction
from huggingface_hub import login, hf_hub_download
from datasets import load_dataset, Dataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

from metrics.regression import compute_metrics_regression as compute_alignment_metrics # takes EvalPrediction object
from data.dataset_classes import IterableDatasetFromHF, SequenceDatasetFromList
from models.modeling_esm_diff import ESM_Diff
from models.modeling_nw_transformer import AlignmentModule, NWTransformerConfig
from data.data_collators import SequenceCollator
from models.alignment_helpers import GetAlignmentScoreFromLogits

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


def get_eval_data():
    """Download and prepare evaluation data"""
    local_file = hf_hub_download(
        repo_id="Synthyra/omg_prot50",
        filename=f"data/valid-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=42).select(range(1000))
    print(data)
    valid_seqs = data['sequence']
    local_file = hf_hub_download(
        repo_id="Synthyra/omg_prot50",
        filename=f"data/test-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=42).select(range(1000))
    print(data)
    test_seqs = data['sequence']
    return valid_seqs, test_seqs


def compute_model_metrics(logits, labels, input_ids):
    """Compute metrics similar to train_esm_diff.py"""
    metrics = {}
    scores = GetAlignmentScoreFromLogits()(logits, input_ids)
    
    # Calculate cross entropy loss
    cross_entropy_loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]), 
        labels.view(-1),
        ignore_index=-100
    )

    metrics['cross_entropy_loss'] = cross_entropy_loss.item()
    metrics['alignment_score'] = scores.mean().item()

    # Convert tensors to numpy for sklearn metrics
    y_pred = logits.argmax(dim=-1).flatten().cpu().numpy()
    y_true = labels.flatten().cpu().numpy()
    valid_indices = y_true != -100
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]
    
    # Calculate classification metrics
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

    return metrics


def training_step(
        args,
        model,
        alignment_module,
        old_alignment_module,
        batch,
        model_optimizer,
        alignment_optimizer,
        model_scheduler,
        alignment_scheduler,
        device,
        train_together=False,
    ):
    """
    Perform a single training step for both model and alignment module.
    When train_together is True, the model is trained with feedback from the alignment module.
    """
    model.train()
    alignment_module.train()
    old_alignment_module.eval()  # Always in eval mode

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Reset optimizers
    model_optimizer.zero_grad()
    alignment_optimizer.zero_grad()

    if train_together:
        # Joint training with alignment feedback
        # Get model outputs
        model_output = model(**batch)
        model_logits, model_labels = model_output.logits
        
        # Get alignment prediction from old module (fixed reference)
        with torch.no_grad():
            alignment_pred = old_alignment_module.get_logits(
                input_ids, model_logits, attention_mask, attention_mask
            )
        
        # Compute model loss based on alignment score
        ideal_labels = torch.ones_like(alignment_pred)
        model_loss = F.l1_loss(alignment_pred.view(-1), ideal_labels.view(-1))
        
        # Backward pass for model
        model_loss.backward()
        model_optimizer.step()
        model_scheduler.step()
        
        # Train alignment module on detached outputs
        alignment_output = alignment_module(
            input_ids.detach(),
            model_logits.detach(), 
            attention_mask.detach(),
            attention_mask.detach()
        )
        alignment_loss = alignment_output.loss
        alignment_pred = alignment_output.logits
        alignment_labels = alignment_output.labels
        
        # Backward pass for alignment
        alignment_loss.backward()
        alignment_optimizer.step()
        alignment_scheduler.step()
    else:
        # Separate training
        # Train model
        model_output = model(**batch)
        model_loss = model_output.loss
        model_logits, model_labels = model_output.logits
        
        # Backward pass for model
        model_loss.backward()
        model_optimizer.step()
        model_scheduler.step()
        
        # Train alignment module on detached outputs
        alignment_output = alignment_module(
            input_ids.detach(),
            model_logits.detach(),
            attention_mask.detach(),
            attention_mask.detach()
        )
        alignment_loss = alignment_output.loss
        alignment_pred = alignment_output.logits
        alignment_labels = alignment_output.labels
        
        # Backward pass for alignment
        alignment_loss.backward()
        alignment_optimizer.step()
        alignment_scheduler.step()

    return model_loss, model_logits, model_labels, alignment_loss, alignment_pred, alignment_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--model_path", type=str, default="Synthyra/ESM2-150M", help="Path to the model to train")
    parser.add_argument("--save_path", type=str, default="Synthyra/test-esm-diff", help="Path to save the model and report to wandb")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of steps to train for")
    parser.add_argument("--wandb_project", type=str, default="ESM-Diff", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of sequences fed to the model")
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in terms of evaluation rounds")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    args = parser.parse_args()
    return args


def evaluate(model, dataset, data_collator, device, batch_size=32, alignment_module=None):
    """Evaluate the model and optionally the alignment module on a dataset"""
    # Set models to evaluation mode
    model.eval()
    if alignment_module is not None:
        alignment_module.eval()
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Initialize collectors for all predictions and labels
    all_model_logits = []
    all_model_labels = []
    all_input_ids = []
    all_alignment_preds = []
    all_alignment_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Process batch
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            # Run model forward pass
            outputs = model(**batch)
            logits, labels = outputs.logits
            
            # Collect predictions and labels
            all_model_logits.append(logits.cpu())
            all_model_labels.append(labels.cpu())
            all_input_ids.append(input_ids.cpu())
            
            # Collect alignment predictions if module provided
            if alignment_module is not None:
                # Run alignment module forward pass
                alignment_output = alignment_module(input_ids, logits, attention_mask, attention_mask)
                all_alignment_preds.append(alignment_output.logits.cpu())
                all_alignment_labels.append(alignment_output.labels.cpu())

    # Concatenate all tensors
    model_logits = torch.cat(all_model_logits, dim=0)
    model_labels = torch.cat(all_model_labels, dim=0)
    input_ids = torch.cat(all_input_ids, dim=0)
    
    # Compute metrics on entire dataset
    metrics = compute_model_metrics(model_logits, model_labels, input_ids)
    
    # Compute alignment metrics if module provided
    if alignment_module is not None and all_alignment_preds:
        alignment_preds = torch.cat(all_alignment_preds, dim=0)
        alignment_labels = torch.cat(all_alignment_labels, dim=0)
        
        # Prepare data for metrics computation
        eval_pred = EvalPrediction(
            predictions=alignment_preds.numpy(),
            label_ids=alignment_labels.numpy()
        )
        
        # Add alignment metrics
        alignment_metrics = compute_alignment_metrics(eval_pred)
        for key, value in alignment_metrics.items():
            metrics[f"alignment_{key}"] = value
    
    # Reset models to training mode
    model.train()
    if alignment_module is not None:
        alignment_module.train()
        
    return metrics


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and alignment modules
    model = ESM_Diff.from_pretrained(args.model_path)
    alignment_module = AlignmentModule(NWTransformerConfig())
    old_alignment_module = copy.deepcopy(alignment_module)
    
    # Move models to device
    model.to(device)
    alignment_module.to(device)
    old_alignment_module.to(device)
    tokenizer = model.tokenizer
    summary(model)
    
    # Load datasets
    train_dataset = load_dataset("Synthyra/omg_prot50", split="train", streaming=True)
    valid_seqs, test_seqs = get_eval_data()
    if args.bugfix:
        valid_seqs = valid_seqs[:10]
        test_seqs = test_seqs[:10]
    
    valid_dataset = SequenceDatasetFromList(valid_seqs)
    test_dataset = SequenceDatasetFromList(test_seqs)
    data_collator = SequenceCollator(tokenizer, args.max_length)
    
    # Create train dataloader
    train_dataset = IterableDatasetFromHF(train_dataset, "sequence")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=4 if not args.bugfix else 0,
        prefetch_factor=10 if not args.bugfix else None,
    )
    
    # Setup optimizers with different learning rates
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    alignment_optimizer = torch.optim.AdamW(alignment_module.parameters(), lr=args.lr * 10)
    
    # Evaluate initial metrics
    print("Evaluating initial metrics...")
    initial_metrics = evaluate(model, test_dataset, data_collator, device, args.batch_size, alignment_module)
    print('Initial Metrics: \n', initial_metrics)
    
    if WANDB_AVAILABLE:
        wandb.log(initial_metrics, step=0)
    
    # Setup schedulers
    num_warmup_steps = args.save_every // 10
    model_scheduler = get_cosine_schedule_with_warmup(
        model_optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_steps
    )
    
    alignment_scheduler = get_cosine_schedule_with_warmup(
        alignment_optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_steps
    )
    
    # Training loop
    log_every = 100
    model.train()
    alignment_module.train()
    global_step = 0
    best_score = float('-inf')
    patience_counter = 0
    train_together = False
    model_loss_tally, alignment_loss_tally = 0, 0
    
    train_iter = iter(train_dataloader)
    progress_bar = trange(args.max_steps, desc="Training")
    
    for step in progress_bar:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        
        # Run training step
        model_loss, model_logits, model_labels, alignment_loss, alignment_pred, alignment_labels = training_step(
            args,
            model,
            alignment_module,
            old_alignment_module,
            batch,
            model_optimizer,
            alignment_optimizer,
            model_scheduler,
            alignment_scheduler,
            device,
            train_together=train_together
        )

        model_loss_tally += model_loss.item()
        alignment_loss_tally += alignment_loss.item()

        # Copy alignment module to old_alignment_module every 100 steps after train_together is True
        if train_together and global_step % 100 == 0 and global_step > 0:
            old_alignment_module.load_state_dict(alignment_module.state_dict())
            print(f"Updated old_alignment_module at step {global_step}")
        
        # Log metrics
        if WANDB_AVAILABLE and step % 100 == 0:
            wandb.log({
                "train/model_loss": model_loss.item(),
                "train/alignment_loss": alignment_loss.item(),
                "train/spearman_rho": spearman_rho,
                "train/train_together": int(train_together)
            }, step=global_step)
        
        # Update progress bar with concise info
        if step % log_every == 0:
            progress_bar.set_postfix(
                m_loss=f"{model_loss_tally / log_every:.4f}",
                a_loss=f"{alignment_loss_tally / log_every:.4f}",
                joint=train_together
            )
            model_loss_tally, alignment_loss_tally = 0, 0
        
        # Evaluate and save models at regular intervals
        if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
            # Run evaluation
            print(f"\nEvaluating at step {global_step}...")
            eval_metrics = evaluate(model, valid_dataset, data_collator, device, args.batch_size, alignment_module)
            print(f"Validation Metrics at step {global_step}: \n", eval_metrics)
            
            spearman_rho = eval_metrics.get("alignment_spearman_rho", 0)

            # Check if we should start training together
            if not train_together and spearman_rho > 0.9:
                train_together = True
                print(f"\nStarting joint training at step {global_step} with Spearman's rho: {spearman_rho}")
                old_alignment_module.load_state_dict(alignment_module.state_dict())

            if WANDB_AVAILABLE:
                wandb.log({"val/" + k: v for k, v in eval_metrics.items()}, step=global_step)
            
            # Save checkpoint
            output_dir = os.path.join(args.save_path.split('/')[-1], f"checkpoint-{global_step}")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            torch.save(alignment_module.state_dict(), os.path.join(output_dir, "alignment_module.pt"))
            
            # Check for best model and early stopping
            current_score = eval_metrics.get("alignment_score", 0) + eval_metrics.get("acc", 0)
            if current_score > best_score:
                best_score = current_score
                patience_counter = 0
                
                # Save best model
                best_dir = os.path.join(args.save_path.split('/')[-1], "best")
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                torch.save(alignment_module.state_dict(), os.path.join(best_dir, "alignment_module.pt"))
                print(f"New best model saved with score: {current_score:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} evaluations")
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {patience_counter} evaluations without improvement")
                    break
        
        global_step += 1
    
    # Final evaluation
    print("\nEvaluating final metrics...")
    final_metrics = evaluate(model, test_dataset, data_collator, device, args.batch_size, alignment_module)
    print('Final Metrics: \n', final_metrics)
    
    if WANDB_AVAILABLE:
        wandb.log({"test/" + k: v for k, v in final_metrics.items()}, step=global_step)
    
    # Push model to hub
    model.push_to_hub(args.save_path, private=True)
    
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    # py -m trainers.train_esm_diff_rl
    args = parse_args()

    if WANDB_AVAILABLE:
        run_name = args.save_path.split('/')[-1]
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if args.token is not None:
        login(args.token)    

    if args.bugfix:
        args.batch_size = 2
        args.max_length = 32
        args.save_every = 1000

    main(args)
