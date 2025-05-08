import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from huggingface_hub import login, hf_hub_download
from IPython.display import display

from models.modeling_esm_diff import ESM_Diff
from evaluation.compare_distributions import compare_corpora_kmers


MODEL_PATH = 'GleghornLab/ESM_diff_650'
PREVIEW = False
SLOW = False
REMASKING = 'random'


def get_eval_data(num_samples):
    local_file = hf_hub_download(
        repo_id="Synthyra/omg_prot50",
        filename=f"data/valid-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=888).select(range(num_samples))
    data = data.filter(lambda x: len(x['sequence']) > 20 and len(x['sequence']) < 2048)
    print(data)
    valid_seqs = sorted(data['sequence'], key=len, reverse=True)
    return valid_seqs


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--sweep_step', action='store_true', help='Sweep through step divisors (fixed temperature)')
    parser.add_argument('--sweep_temp', action='store_true', help='Sweep through temperatures (fixed step divisor)')
    parser.add_argument('--step_divisor', type=int, default=25, help='Step divisor (used when sweep_temp=True)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature (used when sweep_step=True)')
    return parser.parse_args()


if __name__ == '__main__':
    # py -m evaluation.unconditional_generation_tuning
    args = arg_parser()
    if args.token is not None:
        login(args.token)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM_Diff.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer

    natural_seqs = get_eval_data(args.num_samples)
    
    # Define ranges for step divisor and temperature
    step_divisors = [1, 2, 5, 10, 25, 50, 100]
    temperatures = [0.1, 0.25, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    results = []
    
    # Determine which parameter to sweep
    if args.sweep_step and args.sweep_temp:
        print("Error: Cannot sweep both step divisor and temperature at the same time.")
        print("Please use either --sweep_step or --sweep_temp, not both.")
        exit(1)
    elif args.sweep_step:
        # Sweep through step divisors with fixed temperature
        parameter_combinations = [(step, args.temperature) for step in step_divisors]
        print(f"Sweeping through step divisors with fixed temperature = {args.temperature}")
    elif args.sweep_temp:
        # Sweep through temperatures with fixed step divisor
        parameter_combinations = [(args.step_divisor, temp) for temp in temperatures]
        print(f"Sweeping through temperatures with fixed step divisor = {args.step_divisor}")
    else:
        # Default: use the provided values for a single run
        parameter_combinations = [(args.step_divisor, args.temperature)]
        print(f"Running single evaluation with step_divisor = {args.step_divisor}, temperature = {args.temperature}")
    
    # For progress tracking
    total_combinations = len(parameter_combinations)
    combination_count = 0
    
    # Loop through parameter combinations
    for step_divisor, temperature in parameter_combinations:
        combination_count += 1
        print(f"\n[{combination_count}/{total_combinations}] Testing STEP_DIVISOR={step_divisor}, TEMPERATURE={temperature}")
        
        generated_seqs = []
        
        # Generate sequences with current parameters
        for seq in tqdm(natural_seqs):
            output_tokens = model.mask_diffusion_generate(
                length=len(seq),
                block_wise=False,
                batch_size=args.batch_size,
                steps=len(seq) // step_divisor,
                temperature=temperature,
                remasking=REMASKING,
                preview=PREVIEW,
                slow=SLOW,
                start_with_methionine=False
            )
            for gen_seq in output_tokens:
                generated_seqs.append(model._decode_seq(gen_seq))
        
        # Compare distributions and collect stats
        stats = compare_corpora_kmers(natural_seqs, generated_seqs)
        
        # Store results for each k-mer
        for k, res in stats.items():
            chi_p = res["p"]
            jsd = res["js"]
            results.append({
                'step_divisor': step_divisor,
                'temperature': temperature,
                'k': k,
                'p_value': chi_p,
                'jsd': jsd
            })
    
    # Create and display dataframe with all results
    results_df = pd.DataFrame(results)
    
    # Display results grouped by k-mer
    for k in sorted(results_df['k'].unique()):
        print(f"\nResults for {k}-mer:")
        k_results = results_df[results_df['k'] == k].sort_values('p_value', ascending=True)
        display(k_results)
    
    # Display overall best results sorted by 3-mer p-value
    print("\nAll results sorted by 3-mer p-value (lower is better):")
    
    # Create pivot table
    pivot_df = results_df.pivot_table(
        index=['step_divisor', 'temperature'], 
        columns='k', 
        values=['p_value', 'jsd']
    )
    
    # Fix for the merging error - directly sort the pivot table
    # Sort by JSD first (smaller is better), then by p-value (smaller is better) for 3-mer
    sorted_indices = pivot_df.sort_values(by=[('jsd', 3), ('p_value', 3)], ascending=[True, True]).index
    
    # Sort the entire pivot table by the sorted indices
    final_results = pivot_df.loc[sorted_indices]
    display(final_results)
    
    # Print best combination
    best_combo = sorted_indices[0]
    print(f"\nBest combination: step_divisor={best_combo[0]}, temperature={best_combo[1]}")

    # Save results to CSV
    results_df.to_csv(f'unconditional_generation_tuning_results_{args.sweep_step}_{args.sweep_temp}.csv', index=False)
