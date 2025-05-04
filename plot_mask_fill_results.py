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


def generate_comparison_plot(results_dir, metric='loss', output_file='results/mask_rate_comparison.png'):
    """
    Generate a plot comparing all models across all mask rates for a specific metric.
    
    Args:
        results_dir: Directory containing the evaluation results
        metric: Metric to use for comparison (loss, perplexity, etc.)
        output_file: Path to save the plot
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find all CSV files in the results directory
    csv_files = glob(os.path.join(results_dir, 'mask_fill_benchmark_*_mask*.csv'))
    
    if not csv_files:
        print("No result files found. Run the evaluation first.")
        return
    
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
        
        # Extract the models and their metrics
        for _, row in df.iterrows():
            model_name = row['model']
            metric_value = row[metric]
            
            if model_name not in results_data:
                results_data[model_name] = {'mask_rates': [], 'metrics': [], 'dataset_types': []}
            
            results_data[model_name]['mask_rates'].append(mask_rate)
            results_data[model_name]['metrics'].append(metric_value)
            results_data[model_name]['dataset_types'].append(dataset_type)
    
    # Define a colormap for different models
    colors = plt.cm.tab10.colors[:len(results_data)]
    
    # Prepare separate subplots for valid and test datasets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    metric_name = pretty_metric_names[metric]

    # Add a main title
    fig.suptitle(f'Model Performance ({metric_name}) vs Mask Rate', fontsize=16)
    
    # Plot lines for each model on both subplots
    for i, (model_name, data) in enumerate(results_data.items()):
        # Create a DataFrame for easier manipulation
        model_df = pd.DataFrame({
            'mask_rate': data['mask_rates'],
            'metric': data['metrics'],
            'dataset_type': data['dataset_types']
        })
        
        # Plot for valid dataset
        valid_df = model_df[model_df['dataset_type'] == 'valid']
        if not valid_df.empty:
            mask_rates_sorted = sorted(valid_df['mask_rate'].unique())
            metrics = [valid_df[valid_df['mask_rate'] == rate]['metric'].values[0] for rate in mask_rates_sorted]
            ax1.plot(mask_rates_sorted, metrics, marker='o', label=model_name, color=colors[i])
        
        # Plot for test dataset
        test_df = model_df[model_df['dataset_type'] == 'test']
        if not test_df.empty:
            mask_rates_sorted = sorted(test_df['mask_rate'].unique())
            metrics = [test_df[test_df['mask_rate'] == rate]['metric'].values[0] for rate in mask_rates_sorted]
            ax2.plot(mask_rates_sorted, metrics, marker='o', label=model_name, color=colors[i])
    
    # Set titles and labels
    ax1.set_title('Validation Dataset')
    ax2.set_title('Test Dataset')
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('Mask Rate')
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle='--', alpha=0.7)
        # Format x-axis to show mask rates as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
    
    # Add legend to the right of the second subplot with minimal spacing
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5))
    
    # Adjust layout to minimize white space
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Reduced right margin
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")
    
    # Also create a single plot with all datasets for quick comparison
    plt.figure(figsize=(12, 8))
    
    # Plot lines for each model and dataset
    for i, (model_name, data) in enumerate(results_data.items()):
        # Create a DataFrame for easier manipulation
        model_df = pd.DataFrame({
            'mask_rate': data['mask_rates'],
            'metric': data['metrics'],
            'dataset_type': data['dataset_types']
        })
        
        # Plot for each dataset type
        for dataset, linestyle in [('valid', 'dashed'), ('test', 'solid')]:
            dataset_df = model_df[model_df['dataset_type'] == dataset]
            if not dataset_df.empty:
                mask_rates_sorted = sorted(dataset_df['mask_rate'].unique())
                metrics = [dataset_df[dataset_df['mask_rate'] == rate]['metric'].values[0] for rate in mask_rates_sorted]
                plt.plot(mask_rates_sorted, metrics, marker='o', linestyle=linestyle,
                         label=f"{model_name} ({dataset})", color=colors[i])
    
    # Add labels and title
    plt.xlabel('Mask Rate')
    plt.ylabel(metric_name)
    plt.title(f'Model Performance ({metric_name}) vs Mask Rate')
    
    # Format x-axis to show mask rates as percentages
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    combined_output = output_file.replace('.png', '_combined.png')
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    print(f"Combined comparison plot saved to {combined_output}")