import torch
import argparse
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from IPython.display import display

from models.modeling_dsm import DSM
from evaluation.compare_distributions import CorpusComparator, AA20
from .utils import get_eval_data


MODEL_PATH = 'GleghornLab/DSM_650'
PREVIEW = False
SLOW = False
STEP_DIVISOR = 10
TEMPERATURE = 1.0


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    return parser.parse_args()


if __name__ == '__main__':
    # py -m evaluation.unconditional_generation_tuning_remasking
    args = arg_parser()
    if args.token is not None:
        login(args.token)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSM.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer
    mask_token = tokenizer.mask_token
    comparator = CorpusComparator(vocabulary=AA20)

    natural_seqs = get_eval_data(args.num_samples, max_length=128)
    
    remaskings = ['dual', 'random', 'low_confidence']
    
    results = []

    for remasking in remaskings:
        print(f"\nTesting REMASKING={remasking}")
        
        generated_seqs = []
        
        # Generate sequences with current parameters
        for seq in tqdm(natural_seqs):
            template = ''.join([mask_token] * len(seq))
            template_tokens = tokenizer.encode(template, add_special_tokens=True, return_tensors='pt').to(device)
            attention_mask = torch.ones_like(template_tokens)

            output_tokens = model.mask_diffusion_generate(
                tokenizer=tokenizer,
                input_tokens=template_tokens,
                step_divisor=STEP_DIVISOR,
                temperature=TEMPERATURE,
                remasking=remasking,
                preview=PREVIEW,
                slow=SLOW,
                start_with_methionine=False
            )
            generated_seqs.extend(model.decode_output(output_tokens, attention_mask))
        
        # Compare distributions and collect stats
        stats = comparator.compare_corpora_kmers(natural_seqs, generated_seqs, ks=(1, 2, 3, 4, 5))
        
        # Store results for each k-mer
        for k, res in stats.items():
            chi_p = res["p"]
            jsd = res["js"]
            results.append({
                'remasking': remasking,
                'k': k,
                'p_value': chi_p,
                'jsd': jsd
            })
    
    # Create and display dataframe with all results
    results_df = pd.DataFrame(results)
    
    # Display results grouped by k-mer
    for k in sorted(results_df['k'].unique()):
        print(f"\nResults for {k}-mer:")
        k_results = results_df[results_df['k'] == k].sort_values('jsd', ascending=True)
        display(k_results)
    
    # Display overall best results sorted by 3-mer p-value
    print("\nAll results sorted by 3-mer JSD (lower is better):")
    
    # Create pivot table
    pivot_df = results_df.pivot_table(
        index=['remasking'], 
        columns='k', 
        values=['p_value', 'jsd']
    )
    
    # Fix for the merging error - directly sort the pivot table
    # Sort by JSD first (smaller is better), then by p-value (smaller is better) for 3-mer
    sorted_indices = pivot_df.sort_values(by=[('jsd', 3)], ascending=[True]).index
    
    # Sort the entire pivot table by the sorted indices
    final_results = pivot_df.loc[sorted_indices]
    display(final_results)
    
    # Print best combination
    best_combo = sorted_indices[0]
    print(f"\nBest combination: remasking={best_combo[0]}")

    # Save results to CSV
    results_df.to_csv(f'unconditional_generation_tuning_results_remasking.csv', index=False)
