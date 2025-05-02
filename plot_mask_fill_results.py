import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a plot of loss across all mask rates')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory containing evaluation result CSV files')
    parser.add_argument('--output_file', type=str, default='plots/mask_rate_comparison.png', help='Path to save the output plot')
    parser.add_argument('--metric', type=str, default='loss', choices=['loss', 'perplexity', 'precision', 'recall', 'f1', 'accuracy', 'mcc', 'alignment_score'], 
                        help='Metric to plot on y-axis')
    parser.add_argument('--dataset_type', type=str, default='test', choices=['valid', 'test', 'both'], 
                        help='Dataset type to use for plotting')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Find all CSV files in the results directory
    csv_files = glob(os.path.join(args.results_dir, f'mask_fill_benchmark_{args.dataset_type}_mask*.csv'))
    
    # If dataset_type is 'both', get files for both valid and test
    if args.dataset_type == 'both':
        csv_files = glob(os.path.join(args.results_dir, 'mask_fill_benchmark_*_mask*.csv'))
    
    if not csv_files:
        # If no results exist yet, run the evaluation script to generate them
        print("No result files found. Running the evaluation script to generate results...")
        # We'll use the models and mask_rates from the evaluation script
        from eval_mask_fill import main as run_eval
        run_eval()
        
        # Try to find the CSV files again
        csv_files = glob(os.path.join(args.results_dir, f'mask_fill_benchmark_{args.dataset_type}_mask*.csv'))
        if args.dataset_type == 'both':
            csv_files = glob(os.path.join(args.results_dir, 'mask_fill_benchmark_*_mask*.csv'))
            
        if not csv_files:
            print("No result files found even after running evaluation. Please check the evaluation script.")
            return
    
    # Dictionary to store all results
    results_data = {}
    
    # Process all CSV files
    for csv_file in csv_files:
        # Extract mask rate and dataset type from filename
        filename = os.path.basename(csv_file)
        parts = filename.replace('.csv', '').split('_')
        dataset_type = parts[2]
        mask_rate = int(parts[-1].replace('mask', '')) / 100
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Extract the models and their metrics
        for _, row in df.iterrows():
            model_name = row['model']
            metric_value = row[args.metric]
            
            if model_name not in results_data:
                results_data[model_name] = {'mask_rates': [], 'metrics': [], 'dataset_types': []}
            
            results_data[model_name]['mask_rates'].append(mask_rate)
            results_data[model_name]['metrics'].append(metric_value)
            results_data[model_name]['dataset_types'].append(dataset_type)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define a colormap for different models
    cmap = plt.cm.get_cmap('tab10', len(results_data))
    
    # Plot lines for each model
    for i, (model_name, data) in enumerate(results_data.items()):
        # Create a DataFrame for easier manipulation
        model_df = pd.DataFrame({
            'mask_rate': data['mask_rates'],
            'metric': data['metrics'],
            'dataset_type': data['dataset_types']
        })
        
        if args.dataset_type != 'both':
            # If specific dataset type, plot a single line
            mask_rates_sorted = sorted(model_df['mask_rate'].unique())
            metrics = [model_df[model_df['mask_rate'] == rate]['metric'].values[0] for rate in mask_rates_sorted]
            plt.plot(mask_rates_sorted, metrics, marker='o', label=model_name, color=cmap(i))
        else:
            # If both dataset types, plot separate lines for valid and test
            for dataset in ['valid', 'test']:
                dataset_df = model_df[model_df['dataset_type'] == dataset]
                if not dataset_df.empty:
                    mask_rates_sorted = sorted(dataset_df['mask_rate'].unique())
                    metrics = [dataset_df[dataset_df['mask_rate'] == rate]['metric'].values[0] for rate in mask_rates_sorted]
                    plt.plot(mask_rates_sorted, metrics, marker='o', label=f"{model_name} ({dataset})", 
                             color=cmap(i), linestyle='solid' if dataset == 'test' else 'dashed')
    
    # Add labels and title
    plt.xlabel('Mask Rate')
    plt.ylabel(args.metric.capitalize())
    plt.title(f'Model Performance ({args.metric}) vs Mask Rate')
    
    # Format x-axis to show mask rates as percentages
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output_file}")
    
    # Show the plot
    plt.show()


if __name__ == '__main__':
    main() 