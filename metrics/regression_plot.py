import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score


def plot_predictions(predictions, labels, metrics, output_path):
    """
    Create a visualization of predicted vs. actual values with metrics and confidence intervals
    
    Args:
        predictions (np.array): Model predictions
        labels (np.array): Ground truth labels
        metrics (dict): Dictionary of computed metrics
        output_path (str): Path to save the visualization
    """
    # Flatten arrays and remove NaN values
    predictions, labels = predictions.flatten(), labels.flatten()
    mask = np.isfinite(labels) & np.isfinite(predictions)
    labels, predictions = labels[mask], predictions[mask]
    
    # Create scatter plot and regression line with 95% CI
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=labels, y=predictions, ax=ax)
    sns.regplot(
        x=labels, y=predictions,
        ci=95, ax=ax, scatter=False,
        line_kws={'color': 'red'}
    )
    
    # Set labels and title
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Regression Plot with 95% Confidence Interval')
    
    # Get metrics from the metrics dictionary or calculate if not present
    r2 = metrics.get('r_squared', r2_score(labels, predictions))
    r_s = metrics.get('spearman_rho', spearmanr(labels, predictions)[0])
    p_s = metrics.get('spear_pval', spearmanr(labels, predictions)[1])
    r_p = metrics.get('pearson_rho', pearsonr(labels, predictions)[0])
    p_p = metrics.get('pear_pval', pearsonr(labels, predictions)[1])
    
    # Annotate statistics on the plot
    stats_text = (
        f"$R^2$ = {r2:.2f}\n"
        f"Spearman $\\rho$ = {r_s:.2f}  (p = {p_s:.2e})\n"
        f"Pearson $\\rho$ = {r_p:.2f}  (p = {p_p:.2e})"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close(fig)