#!/usr/bin/env python3
"""
Minimal script to compute and compare ESMfold plDDT scores between natural and generated protein sequences.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from transformers import EsmForProteinFolding, AutoTokenizer
import re


torch.backends.cuda.matmul.allow_tf32 = True


def clean_protein_sequence(sequence):
    """
    Clean protein sequence by replacing invalid characters with 'A'.
    
    Args:
        sequence: Input protein sequence string
        
    Returns:
        Cleaned sequence with only standard amino acids (invalids replaced by 'A')
    """
    # Standard amino acid alphabet
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Replace any non-amino acid character with 'A' and convert to uppercase
    cleaned = ''.join(c.upper() if c.upper() in valid_aa else 'A' for c in sequence)
    
    return cleaned


def compute_esmfold_plddt(sequences, batch_size=1, max_length=400):
    """
    Compute ESMfold plDDT scores for a list of protein sequences.
    
    Args:
        sequences: List of protein sequences
        batch_size: Number of sequences to process at once
        max_length: Maximum sequence length to process
    
    Returns:
        List of average plDDT scores for each sequence
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clean sequences to remove invalid characters
    print("Cleaning protein sequences...")
    cleaned_sequences = []
    for seq in sequences:
        cleaned = clean_protein_sequence(seq)
        if len(cleaned) > 0:  # Only keep non-empty cleaned sequences
            cleaned_sequences.append(cleaned)
        else:
            print(f"Warning: Sequence '{seq[:50]}...' became empty after cleaning")
    
    print(f"Cleaned {len(sequences)} sequences to {len(cleaned_sequences)} valid sequences")
    
    # Load ESMFold model
    print("Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)
    model.esm = model.esm.half()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    
    plddt_scores = []
    
    # Filter sequences by length
    valid_sequences = [seq for seq in cleaned_sequences if len(seq) <= max_length and len(seq) > 0]
    print(f"Processing {len(valid_sequences)} sequences (filtered from {len(cleaned_sequences)} by max_length={max_length})")
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(valid_sequences), batch_size), desc="Computing plDDT"):
            batch_seqs = valid_sequences[i:i+batch_size]
            
            try:
                # Tokenize sequences
                tokenized = tokenizer(
                    batch_seqs, 
                    return_tensors="pt", 
                    padding=False, 
                    truncation=False,
                    add_special_tokens=False
                )
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                
                # Forward pass
                output = model(tokenized["input_ids"])
                
                # Extract plDDT scores
                plddt = output["plddt"]  # Shape: (batch_size, seq_len)
                
                # Compute average plDDT for each sequence in the batch
                for j, seq in enumerate(batch_seqs):
                    seq_len = len(seq)
                    # Use actual sequence length, not padded length
                    avg_plddt = plddt[j, :seq_len].mean().item()
                    plddt_scores.append(avg_plddt)
                    
            except Exception as e:
                print(f"Error processing sequences {i}-{i+len(batch_seqs)}: {e}")
                # Add NaN for failed sequences
                plddt_scores.extend([np.nan] * len(batch_seqs))
    
    return plddt_scores


def compare_distributions(natural_scores, generated_scores, output_dir="results"):
    """
    Compare plDDT score distributions and create visualizations.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Remove NaN values
    natural_clean = [x for x in natural_scores if not np.isnan(x)]
    generated_clean = [x for x in generated_scores if not np.isnan(x)]
    
    print(f"\nResults:")
    print(f"Natural sequences: {len(natural_clean)} valid scores (avg: {np.mean(natural_clean):.3f} ± {np.std(natural_clean):.3f})")
    print(f"Generated sequences: {len(generated_clean)} valid scores (avg: {np.mean(generated_clean):.3f} ± {np.std(generated_clean):.3f})")
    
    # Statistical test
    if len(natural_clean) > 0 and len(generated_clean) > 0:
        t_stat, p_value = stats.ttest_ind(natural_clean, generated_clean)
        ks_stat, ks_p = stats.ks_2samp(natural_clean, generated_clean)
        
        print(f"\nStatistical Tests:")
        print(f"T-test: t={t_stat:.3f}, p={p_value:.3e}")
        print(f"KS-test: D={ks_stat:.3f}, p={ks_p:.3e}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 5))
    
    # Histogram comparison
    plt.subplot(1, 2, 1)
    plt.hist(natural_clean, bins=30, alpha=0.7, label='Natural', density=True)
    plt.hist(generated_clean, bins=30, alpha=0.7, label='Generated', density=True)
    plt.xlabel('Average plDDT Score')
    plt.ylabel('Density')
    plt.title('plDDT Score Distribution Comparison')
    plt.legend()
    
    # Box plot comparison
    plt.subplot(1, 2, 2)
    data_to_plot = [natural_clean, generated_clean]
    labels = ['Natural', 'Generated']
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Average plDDT Score')
    plt.title('plDDT Score Box Plot Comparison')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plddt_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'sequence_type': ['Natural'] * len(natural_clean) + ['Generated'] * len(generated_clean),
        'plddt_score': natural_clean + generated_clean
    })
    results_df.to_csv(f"{output_dir}/plddt_scores.csv", index=False)
    print(f"\nResults saved to {output_dir}/")


def main():
    # py -m evaluation.compute_plddt_comparison
    parser = argparse.ArgumentParser(description="Compare ESMfold plDDT scores between natural and generated sequences")
    parser.add_argument("--input_csv", type=str, default="evaluation/comparisons/unconditional_generation_seqs.csv",
                       help="CSV file with natural and generated sequences")
    parser.add_argument("--max_sequences", type=int, default=100,
                       help="Maximum number of sequences to process from each type")
    parser.add_argument("--max_length", type=int, default=400,
                       help="Maximum sequence length to process")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for ESMfold computation")
    parser.add_argument("--output_dir", type=str, default="plddt_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load sequences
    print(f"Loading sequences from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Sample sequences if requested
    if args.max_sequences > 0:
        df = df.head(args.max_sequences)
    
    natural_seqs = df['natural'].dropna().tolist()
    generated_seqs = df['generated'].dropna().tolist()
    
    print(f"Loaded {len(natural_seqs)} natural and {len(generated_seqs)} generated sequences")
    
    # Compute plDDT scores
    print("\nComputing plDDT scores for natural sequences...")
    natural_plddt = compute_esmfold_plddt(natural_seqs, args.batch_size, args.max_length)
    
    print("\nComputing plDDT scores for generated sequences...")
    generated_plddt = compute_esmfold_plddt(generated_seqs, args.batch_size, args.max_length)
    
    # Compare distributions
    print("\nComparing distributions...")
    compare_distributions(natural_plddt, generated_plddt, args.output_dir)


if __name__ == "__main__":
    main() 