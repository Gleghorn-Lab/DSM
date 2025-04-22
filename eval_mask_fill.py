import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskedLM
from datasets import Dataset
from huggingface_hub import hf_hub_download, login
from tqdm.auto import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef
)
from models.modeling_esm_diff import ESM_Diff
from utils import ProteinMasker


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    return parser.parse_args()


def main():
    # py -m eval_mask_fill
    args = parse_args()

    if args.token is not None:
        login(args.token)

    all_results = {}
    for type in ['valid', 'test']:
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
        total_tokens = sum(len(seq) for seq in sequences)
        print(f"Total tokens: {total_tokens}")

        # Load the ESM tokenizer and model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_names = {
            'Synthyra/ESM2-8M': 'ESM2-8M',
            'Synthyra/ESM2-35M': 'ESM2-35M',
            'Synthyra/ESM2-150M': 'ESM2-150M',
            'GleghornLab/eval_diff_150': 'ESMdiff-150M',
            #'Synthyra/ESMplusplus_small': 'ESMC-300M',
            #'Synthyra/ESMplusplus_large': 'ESMC-600M',
            'Synthyra/ESM2-650M': 'ESM2-650M'
        }

        ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

        mask_rate, batch_size = 0.15, 4
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
            
            vocab_size = model.config.vocab_size
            masker = ProteinMasker(tokenizer)
            total_loss, count = 0.0, 0
            all_true, all_pred = [], []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(sequences), batch_size), desc=f'Evaluating {model_name}'):
                    batch = sequences[i:i+batch_size]
                    encoding = tokenizer(batch, return_tensors='pt', padding=True, add_special_tokens=True, truncation=True, max_length=1024)
                    input_ids, attention_mask = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
                    masked_input_ids, labels = masker(input_ids, attention_mask)

                    if diff:
                        logits = model._get_logits(masked_input_ids, attention_mask)
                    else:
                        logits = model(input_ids=masked_input_ids, attention_mask=attention_mask).logits

                    loss = ce_loss(logits.view(-1, vocab_size), labels.view(-1))
                    pred = logits.argmax(dim=-1)

                    all_true.extend(labels.cpu().numpy().flatten())
                    all_pred.extend(pred.cpu().numpy().flatten())
                    total_loss += loss.item()
                    count += 1

            average_loss = total_loss / count
            perplexity = torch.exp(torch.tensor(average_loss)).item()
            
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
            
            results.append({
                'model': nickname,
                'loss': round(average_loss, 4),
                'perplexity': round(perplexity, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'accuracy': round(accuracy, 4),
                'mcc': round(mcc, 4)
            })
            
            model.cpu()
            del model
            torch.cuda.empty_cache()

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'mask_fill_benchmark_{type}.csv', index=False)
        all_results[type] = results_df
        
        # Create bar graphs for loss and MCC
        plt.figure(figsize=(14, 10))
        
        # Loss subplot (lower is better)
        plt.subplot(2, 1, 1)
        plt.bar(results_df['model'], results_df['loss'], color='skyblue')
        plt.title(f'Loss by Model ({type} dataset)')
        plt.xlabel('Model')
        plt.ylabel('Loss (↓ lower is better)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # MCC subplot (higher is better)
        plt.subplot(2, 1, 2)
        plt.bar(results_df['model'], results_df['mcc'], color='lightgreen')
        plt.title(f'Matthews Correlation Coefficient by Model ({type} dataset)')
        plt.xlabel('Model')
        plt.ylabel('MCC (↑ higher is better)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'mask_fill_metrics_{type}.png')
        plt.close()
    
    # Create combined comparison graph if both datasets exist
    if len(all_results) == 2:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        for i, metric in enumerate(['loss', 'mcc']):
            better_indicator = '↓ lower is better' if metric == 'loss' else '↑ higher is better'
            color = 'skyblue' if metric == 'loss' else 'lightgreen'
            
            for j, dataset_type in enumerate(['valid', 'test']):
                df = all_results[dataset_type]
                axes[i, j].bar(df['model'], df[metric], color=color)
                axes[i, j].set_title(f'{metric.upper()} by Model ({dataset_type} dataset)')
                axes[i, j].set_xlabel('Model')
                axes[i, j].set_ylabel(f'{metric.upper()} ({better_indicator})')
                axes[i, j].tick_params(axis='x', rotation=45)
                axes[i, j].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('mask_fill_metrics_comparison.png')
        plt.close()


if __name__ == '__main__':
    main()
