import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, login
from tqdm.auto import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef
)
from transformers import AutoModelForMaskedLM
from glob import glob

from data.dataset_classes import SequenceDatasetFromList
from data.data_collators import SequenceCollator_mask
from models.modeling_esm_diff import ESM_Diff
from models.alignment_helpers import AlignmentScorer
from evaluation.plot_mask_fill_results import generate_comparison_plot
from .utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mask_rates', nargs='+', type=float, default=[0.05, 0.15, 0.30, 0.50, 0.70, 0.90])
    parser.add_argument('--data_splits', nargs='+', type=str, default=['valid', 'test', 'ppi'])
    parser.add_argument('--max_length', type=int, default=1022)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--generate_comparison_plot', action='store_true', 
                       help='Generate a plot comparing all models across all mask rates')
    parser.add_argument('--plot_output', type=str, default='results/mask_rate_comparison.png',
                       help='Path to save comparison plot')
    return parser.parse_args()


def main():
    # py -m evaluation.mask_filling
    args = parse_args()
    
    metrics = ['loss', 'f1', 'alignment_score']
    # If only generating the comparison plot, skip evaluation
    if args.generate_comparison_plot and len(glob(os.path.join(args.results_dir, 'mask_fill_benchmark_*_mask*.csv'))) > 0:
        print("Generating comparison plot from existing results...")
        generate_comparison_plot(args.results_dir, metrics, args.plot_output,
                                 exclude_models=[
                                     'ESMdiff-650M-80k',
                                     'ESMdiff-650M-40k',
                                     'ESMC-600M',
                                     'ESMC-300M',
                                     
        ])
        return
    else:
        print("No existing results found. Running evaluation...")
    
    batch_size = args.batch_size
    mask_rates = args.mask_rates
    max_length = args.max_length

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Login once if token is provided
    if args.token is not None:
        login(args.token)
    
    # Initialize components that don't need to be recreated for each model or dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scorer = AlignmentScorer()
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    
    # Define models once
    model_names = {
        'Synthyra/ESM2-8M': 'ESM2-8M',
        'Synthyra/ESM2-35M': 'ESM2-35M',
        'Synthyra/ESM2-150M': 'ESM2-150M',
        'GleghornLab/eval_diff_150': 'ESMdiff-150M',
        'Synthyra/ESMplusplus_small': 'ESMC-300M',
        'Synthyra/ESMplusplus_large': 'ESMC-600M',
        'Synthyra/ESM2-650M': 'ESM2-650M',
        'lhallee/esm_diff_650_40000': 'ESMdiff-650M-40k',
        'lhallee/esm_diff_650_80000': 'ESMdiff-650M-80k',
        'GleghornLab/ESM_diff_650': 'ESMdiff-650M',
        'Synthyra/ESM2-3B': 'ESM2-3B'
    }

    all_results = {}
    for mask_rate in mask_rates:
        for type in args.data_splits:
            if type == 'ppi':
                data = load_dataset("lhallee/string_model_org_90_90_split", split='test')
                sequences = data['SeqB']
            else:
                local_file = hf_hub_download(
                    repo_id="Synthyra/omg_prot50",
                    filename=f"data/{type}-00000-of-00001.parquet",
                    repo_type="dataset"
                )
                data = Dataset.from_parquet(local_file)
                print(data)
                sequences = data['sequence']
            sequences = sorted(sequences, key=len, reverse=True)
            #sequences = sequences[:10]
            print(sequences[-1])

            results = []
            for model_name, nickname in model_names.items():
                set_seed(42)
                if 'diff' in model_name.lower():
                    model = ESM_Diff.from_pretrained(model_name).to(device).eval()
                    tokenizer = model.tokenizer
                    diff = True
                else:
                    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
                    tokenizer = model.tokenizer
                    diff = False
                
                dataset = SequenceDatasetFromList(sequences)
                collator = SequenceCollator_mask(tokenizer, max_length, mask_rate)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4 if os.cpu_count() > 8 else 0,
                    collate_fn=collator
                )

                vocab_size = model.config.vocab_size
                total_loss, count = 0.0, 0
                all_true, all_pred, pred_seqs, true_seqs = [], [], [], []
                
                with torch.no_grad():
                    for batch in tqdm(dataloader, desc=f'Evaluating {model_name}'):
                        batch = {k: v.to(device) for k, v in batch.items()}

                        if diff:
                            logits = model._get_logits(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                            )
                        else:
                            logits = model(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                            ).logits

                        original_ids = batch['original_ids']
                        labels = batch['labels']
                        loss = ce_loss(logits.view(-1, vocab_size), labels.view(-1))
                        preds = logits.argmax(dim=-1)

                        all_true.extend(labels.cpu().numpy().flatten())
                        all_pred.extend(preds.cpu().numpy().flatten())
                        for ids, pred in zip(original_ids, preds):
                            pred_seqs.append(tokenizer.decode(pred.flatten().tolist(), skip_special_tokens=True))
                            true_seqs.append(tokenizer.decode(ids.flatten().tolist(), skip_special_tokens=True))
                        total_loss += loss.item()
                        count += 1

                average_loss = total_loss / count
                perplexity = torch.exp(torch.tensor(average_loss)).item()
                
                alignment_scores = []

                for pred, true in zip(pred_seqs, true_seqs):
                    alignment_scores.append(scorer(true, pred))

                average_alignment_score = sum(alignment_scores) / len(alignment_scores)

                # Calculate metrics
                all_true = np.array(all_true).flatten()
                all_pred = np.array(all_pred).flatten()
                mask = (all_true != -100)
                all_true = all_true[mask]
                all_pred = all_pred[mask]
                precision = precision_score(all_true, all_pred, average='weighted', zero_division=0)
                recall = recall_score(all_true, all_pred, average='weighted', zero_division=0)
                f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)
                accuracy = accuracy_score(all_true, all_pred)
                mcc = matthews_corrcoef(all_true, all_pred)
                
                print(f'Results for {nickname}:')
                print(f'Loss: {average_loss}')
                print(f'Perplexity: {perplexity:.4f}')
                print(f'Precision: {precision:.4f}')
                print(f'Recall: {recall:.4f}')
                print(f'F1: {f1:.4f}')
                print(f'Accuracy: {accuracy:.4f}')
                print(f'MCC: {mcc:.4f}')
                print(f'Alignment score: {average_alignment_score:.4f}')
                
                results.append({
                    'model': nickname,
                    'loss': round(average_loss, 4),
                    'perplexity': round(perplexity, 4),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1': round(f1, 4),
                    'accuracy': round(accuracy, 4),
                    'mcc': round(mcc, 4),
                    'alignment_score': round(average_alignment_score, 4)
                })
                
                model.cpu()
                del model
                torch.cuda.empty_cache()

            # Save results to CSV
            results_df = pd.DataFrame(results)
            csv_path = os.path.join(args.results_dir, f'mask_fill_benchmark_{type}_mask{int(mask_rate*100)}.csv')
            results_df.to_csv(csv_path, index=False)
            all_results[type] = results_df
            
            # Create bar graphs for loss, MCC, and alignment score
            plt.figure(figsize=(14, 15))
            
            # Loss subplot (lower is better)
            plt.subplot(3, 1, 1)
            plt.bar(results_df['model'], results_df['loss'], color='skyblue')
            plt.title(f'Loss by Model ({type} dataset) - Mask rate {mask_rate}')
            plt.xlabel('Model')
            plt.ylabel('Loss (↓ lower is better)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # MCC subplot (higher is better)
            plt.subplot(3, 1, 2)
            plt.bar(results_df['model'], results_df['mcc'], color='lightgreen')
            plt.title(f'Matthews Correlation Coefficient by Model ({type} dataset) - Mask rate {mask_rate}')
            plt.xlabel('Model')
            plt.ylabel('MCC (↑ higher is better)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Alignment Score subplot (higher is better)
            plt.subplot(3, 1, 3)
            plt.bar(results_df['model'], results_df['alignment_score'], color='lightcoral')
            plt.title(f'Alignment Score by Model ({type} dataset) - Mask rate {mask_rate}')
            plt.xlabel('Model')
            plt.ylabel('Alignment Score (↑ higher is better)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plot_path = os.path.join(args.results_dir, f'mask_fill_metrics_{type}_mask{int(mask_rate*100)}.png')
            plt.savefig(plot_path)
            plt.close()
        
        # Create combined comparison graph if both datasets exist
        if len(all_results) == 2:
            fig, axes = plt.subplots(3, 2, figsize=(18, 18))
            
            metrics = ['loss', 'mcc', 'alignment_score']
            better_indicators = {'loss': '↓ lower is better', 'mcc': '↑ higher is better', 'alignment_score': '↑ higher is better'}
            colors = {'loss': 'skyblue', 'mcc': 'lightgreen', 'alignment_score': 'lightcoral'}
            
            for i, metric in enumerate(metrics):
                for j, dataset_type in enumerate(['valid', 'test']):
                    df = all_results[dataset_type]
                    axes[i, j].bar(df['model'], df[metric], color=colors[metric])
                    axes[i, j].set_title(f'{metric.upper()} by Model ({dataset_type} dataset) - Mask rate {mask_rate}')
                    axes[i, j].set_xlabel('Model')
                    axes[i, j].set_ylabel(f'{metric.upper()} ({better_indicators[metric]})')
                    axes[i, j].tick_params(axis='x', rotation=45)
                    axes[i, j].grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            comparison_path = os.path.join(args.results_dir, f'mask_fill_metrics_comparison_mask{int(mask_rate*100)}.png')
            plt.savefig(comparison_path)
            plt.close()
    
    # Generate the comprehensive plot across all mask rates if requested
    if args.generate_comparison_plot:
        generate_comparison_plot(args.results_dir, metrics, args.plot_output)


if __name__ == '__main__':
    main()
