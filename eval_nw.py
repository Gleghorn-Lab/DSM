import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import EsmTokenizer, EvalPrediction
from scipy import stats
from tqdm import tqdm

from metrics.regression import compute_metrics_regression
from models.modeling_nw_transformer import NWTransformerCross
from data.dataset_classes import NWDatasetEval
from data.data_collators import NWCollatorCross


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GleghornLab/AlignmentTransformer", help="Path to the model to evaluate")
    parser.add_argument("--dataset_name", type=str, default="Synthyra/bernett_gold_ppi", help="Name of the dataset to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for evaluation")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum length of sequences fed to the model")
    parser.add_argument("--output_plot", type=str, default="prediction_plot.png", help="Path to save the visualization")
    return parser.parse_args()


def plot_predictions(predictions, labels, metrics, output_path):
    """
    Create a visualization of predicted vs. actual values with metrics and confidence intervals
    
    Args:
        predictions (np.array): Model predictions
        labels (np.array): Ground truth labels
        metrics (dict): Dictionary of computed metrics
        output_path (str): Path to save the visualization
    """
    # Set up the figure with a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot of predicted vs. actual
    scatter = ax.scatter(labels, predictions, alpha=0.6, color='#3498db', edgecolor='#2980b9', s=80)
    
    # Add line of perfect prediction
    min_val = min(np.min(predictions), np.min(labels))
    max_val = max(np.max(predictions), np.max(labels))
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'k--', alpha=0.5)
    
    # Calculate confidence interval for Spearman correlation
    rho = metrics['spearman_rho']
    n = len(predictions)
    # Fisher z-transformation for confidence intervals
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1 / np.sqrt(n - 3)
    z_lower = z - 1.96 * se
    z_upper = z + 1.96 * se
    # Transform back to correlation
    lower_rho = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    upper_rho = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    # Add regression line with confidence interval
    slope, intercept, _, _, _ = stats.linregress(labels, predictions)
    x_line = np.linspace(min_val-margin, max_val+margin, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='#e74c3c', linewidth=2, label=f'Regression Line')
    
    # Add confidence interval for regression
    ax.fill_between(
        x_line, 
        y_line - 1.96 * np.std(predictions - (slope * labels + intercept)),
        y_line + 1.96 * np.std(predictions - (slope * labels + intercept)),
        color='#e74c3c', alpha=0.2, label='95% Confidence Interval'
    )
    
    # Create metrics text
    metrics_text = "\n".join([
        f"Spearman ρ: {metrics['spearman_rho']:.3f} (95% CI: [{lower_rho:.3f}, {upper_rho:.3f}])",
        f"Pearson r: {metrics['pearson_rho']:.3f} (p-value: {metrics['pear_pval']:.2e})",
        f"R²: {metrics['r_squared']:.3f}",
        f"RMSE: {metrics['rmse']:.3f}",
        f"MAE: {metrics['mae']:.3f}"
    ])
    
    # Add text box with metrics
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#7f8c8d')
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Add labels and title
    ax.set_xlabel('True Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)
    ax.set_title('Prediction Performance', fontsize=16, fontweight='bold')
    
    # Set equal axis limits
    ax.set_xlim([min_val-margin, max_val+margin])
    ax.set_ylim([min_val-margin, max_val+margin])
    
    # Create legend
    legend_elements = [
        Patch(facecolor='#3498db', label='Data Points'),
        Patch(facecolor='#e74c3c', label='Regression Line'),
        Patch(facecolor='#e74c3c', alpha=0.2, label='95% Confidence Interval')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add grid and tight layout
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_dataset(args.dataset_name, split='test').filter(
        lambda x: len(x['SeqA']) + len(x['SeqB']) <= args.max_length
    )
    data = data.shuffle(seed=42).select(range(1000))
    print(data)

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")   
    model = NWTransformerCross.from_pretrained(args.model_path)
    model.eval()
    model.to(device)

    eval_dataset = NWDatasetEval(data)
    eval_collator = NWCollatorCross(tokenizer, args.max_length)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_collator
    )

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Concatenate all predictions and labels
    all_preds = np.stack(all_preds).flatten()
    all_labels = np.stack(all_labels).flatten()
    
    # Compute evaluation metrics
    metrics = compute_metrics_regression(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Create and save visualization
    plot_predictions(all_preds, all_labels, metrics, args.output_plot)


if __name__ == "__main__":
    args = parse_args()
    main(args)