import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob


# ['loss', 'perplexity', 'precision', 'recall', 'f1', 'accuracy', 'mcc', 'alignment_score']
pretty_metric_names = {
    'loss': 'Loss',
    'perplexity': 'Perplexity',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1 Score',
    'accuracy': 'Accuracy',
    'mcc': 'MCC',
    'alignment_score': 'Alignment Score'
}

metric_direction = {
    'loss': 'down',
    'perplexity': 'up',
    'precision': 'up',
    'recall': 'up',
    'f1': 'up',
    'accuracy': 'up',
    'mcc': 'up',
    'alignment_score': 'up'
}

# Define arrows for metric directions
direction_arrows = {
    'up': '↑',  # Arrow pointing up for metrics where higher is better
    'down': '↓'  # Arrow pointing down for metrics where lower is better
}

model_name_to_nickname = {
    'ESMdiff-650M': r'DSM$_{650M}$',
    'ESMdiff-650M-80k': r'DSM$_{650M-80k}$',
    'ESMdiff-650M-40k': r'DSM$_{650M-40k}$',
    'ESMdiff-150M': r'DSM$_{150M}$',
    'ESM2-3B': r'ESM2$_{3B}$',
    'ESM2-650M': r'ESM2$_{650M}$',
    'ESMC-600M': r'ESMC$_{600M}$',
    'ESM2-150M': r'ESM2$_{150M}$',
    'ESMC-300M': r'ESMC$_{300M}$',
    'ESM2-35M': r'ESM2$_{35M}$',
    'ESM2-8M': r'ESM2$_{8M}$',
    'ESM2-3B': r'ESM2$_{3B}$',
    'DPLM-650M': r'DPLM$_{650M}$',
    'DPLM-150M': r'DPLM$_{150M}$'
}

"""
model_name_to_color = {
    # ESMdiff models - blue family (distinguishable for most color blindness types)
    'ESMdiff-650M-80k': '#0072B2',  # dark blue
    'ESMdiff-650M-40k': '#56B4E9',  # light blue
    'ESMdiff-650M': '#009E73',      # teal
    'ESMdiff-150M': '#00BFC4',      # cyan
    
    # ESMC models - orange/brown family
    'ESMC-600M': '#E69F00',         # orange
    'ESMC-300M': '#D55E00',         # dark orange/brown
    
    # ESM2 models - grayscale + purple
    'ESM2-650M': '#999999',         # medium gray
    'ESM2-150M': '#CC79A7',         # pink/purple
    'ESM2-35M': '#F0E442',          # yellow
    'ESM2-8M': '#666666',           # dark gray
    'ESM2-3B': '#000000'            # black
}
"""


model_name_to_color = {
    # ESMdiff models - red/orange family
    'ESMdiff-650M-80k': 'crimson',
    'ESMdiff-650M-40k': 'firebrick',
    'ESMdiff-650M': 'tomato',
    'ESMdiff-150M': 'coral',
    
    # ESMC models - blue/purple family
    'ESMC-600M': 'royalblue',
    'ESMC-300M': 'mediumslateblue',
    
    # ESM2 models - green/teal family
    'ESM2-650M': 'forestgreen',
    'ESM2-150M': 'mediumseagreen',
    'ESM2-35M': 'darkturquoise',
    'ESM2-8M': 'teal',
    'ESM2-3B': 'darkgreen',
    
    # DPLM models - gold/silver family
    'DPLM-650M': 'gold',
    'DPLM-150M': 'silver'
}


def generate_comparison_plot(results_dir, metrics=None, output_file='results/mask_rate_comparison.png', exclude_models=None):
    """
    Generate plots comparing all models across all mask rates for multiple metrics.
    Dataset types (valid, test) are in separate rows, metrics in separate columns.
    Each plot has its own y-axis scale to properly display metrics with different value ranges.
    
    Args:
        results_dir: Directory containing the evaluation results
        metrics: List of metrics to use for comparison (e.g., ['loss', 'precision', 'recall'])
                 If None, uses all available metrics
        output_file: Path to save the plot
        exclude_models: List of model names to exclude from the plot
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find all CSV files in the results directory
    csv_files = glob(os.path.join(results_dir, 'mask_fill_benchmark_*_mask*.csv'))
    
    if not csv_files:
        print("No result files found. Run the evaluation first.")
        return
    
    # Use default metrics if none provided
    if metrics is None:
        metrics = ['loss', 'perplexity', 'precision', 'recall', 'f1', 'accuracy', 'mcc', 'alignment_score']
    
    # Initialize exclude_models if None
    if exclude_models is None:
        exclude_models = []
    
    # Dictionary to store all results
    results_data = {}
    
    # Process all CSV files
    for csv_file in csv_files:
        # Extract mask rate and dataset type from filename
        filename = os.path.basename(csv_file)
        parts = filename.replace('.csv', '').split('_')
        dataset_type = parts[3]
        mask_rate = int(parts[-1].replace('mask', '')) / 100
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check which metrics are available in the data
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            print(f"None of the requested metrics found in {csv_file}")
            continue
        
        # Extract the models and their metrics
        for _, row in df.iterrows():
            model_name = row['model']
            
            # Skip models that should be excluded
            if model_name in exclude_models:
                continue
            
            if model_name not in results_data:
                results_data[model_name] = {
                    'mask_rates': [], 
                    'dataset_types': [], 
                    **{metric: [] for metric in available_metrics}
                }
            
            # Check for invalid metric values
            skip_datapoint = False
            if 'loss' in available_metrics and row['loss'] == -100:
                skip_datapoint = True
            if 'perplexity' in available_metrics and row['perplexity'] == 0:
                skip_datapoint = True
                
            if skip_datapoint:
                continue
                
            results_data[model_name]['mask_rates'].append(mask_rate)
            results_data[model_name]['dataset_types'].append(dataset_type)
            
            for metric in available_metrics:
                metric_value = row[metric]
                results_data[model_name][metric].append(metric_value)
    
    if not results_data:
        print("No data found for the requested metrics.")
        return
    
    # Calculate the grid size for subplots - metrics in columns, dataset types in rows
    n_metrics = len(metrics)
    n_cols = n_metrics
    n_rows = 2  # One for valid, one for test
    
    # Create a figure with subplots for each metric and dataset type
    # Note: Removed sharey='row' to allow each plot to have its own y-scale
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 8), squeeze=False)
    
    # Add a main title
    fig.suptitle('Model Performance vs Mask Rate', fontsize=16)
    
    dataset_types = ['valid', 'test']
    dataset_titles = {'valid': 'Validation Dataset', 'test': 'Test Dataset'}
    
    # For each metric (columns) and dataset type (rows)
    for col_idx, metric in enumerate(metrics):
        if metric not in next(iter(results_data.values())):
            continue
            
        metric_name = pretty_metric_names.get(metric, metric)
        
        for row_idx, dataset_type in enumerate(dataset_types):
            ax = axes[row_idx, col_idx]
            
            # Plot lines for each model
            for model_name, data in results_data.items():
                if metric not in data:
                    continue
                    
                # Create a DataFrame for easier manipulation
                model_df = pd.DataFrame({
                    'mask_rate': data['mask_rates'],
                    'metric': data[metric],
                    'dataset_type': data['dataset_types']
                })
                
                # Filter for current dataset type
                dataset_df = model_df[model_df['dataset_type'] == dataset_type]
                if not dataset_df.empty:
                    # Group by mask_rate and calculate mean to handle potential duplicates
                    grouped = dataset_df.groupby('mask_rate')['metric'].mean().reset_index()
                    mask_rates_sorted = sorted(grouped['mask_rate'])
                    metrics_values = [grouped[grouped['mask_rate'] == rate]['metric'].values[0] for rate in mask_rates_sorted]
                    
                    # Use nickname and color from the mappings
                    nickname = model_name_to_nickname.get(model_name, model_name)
                    color = model_name_to_color.get(model_name, None)  # Use None to fall back to default color cycle if not found
                    
                    ax.plot(mask_rates_sorted, metrics_values, marker='o', label=nickname, color=color)
            
            # Set titles and labels
            if row_idx == 0:
                # Add direction arrow based on metric_direction dictionary
                arrow = direction_arrows.get(metric_direction.get(metric, 'up'), '')
                ax.set_title(f'{metric_name} {arrow}')
            
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_titles[dataset_type]}\n{metric_name}")
            else:
                ax.set_ylabel(metric_name)
                
            # Only add x-label on the bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel('Mask Rate')
            else:
                ax.set_xlabel(' ')
                
            ax.grid(True, linestyle='--', alpha=0.7)
            # Format x-axis to show mask rates as percentages
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
            
            # Set y-axis ticks to a reasonable number to avoid overcrowding
            ax.locator_params(axis='y', nbins=6)
    
    # Hide any unused subplots
    for col_idx in range(len(metrics), n_cols):
        for row_idx in range(n_rows):
            axes[row_idx, col_idx].set_visible(False)
    
    # Add a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Reorder handles and labels according to model_name_to_nickname order
    ordered_handles_labels = []
    for model_name in model_name_to_nickname.keys():
        # Find this model name in the existing labels
        nickname = model_name_to_nickname[model_name]
        if nickname in labels:
            idx = labels.index(nickname)
            ordered_handles_labels.append((handles[idx], labels[idx]))
    
    # Use only the models that were actually plotted
    if ordered_handles_labels:
        ordered_handles, ordered_labels = zip(*ordered_handles_labels)
        fig.legend(ordered_handles, ordered_labels, loc='upper right', bbox_to_anchor=(1.05, 0.5))
    else:
        # Fallback to original order if none of the expected models were found
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 0.5))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-metric comparison plot saved to {output_file}")