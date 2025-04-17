import numpy as np
from transformers import EvalPrediction
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def spearman(probs, correct_preds):
    """
    Calculate and print the Spearman correlation between probabilities and correct predictions.

    Args:
        probs (np.array): Predicted probabilities.
        correct_preds (np.array): Correct predictions (ground truth).
    Prints:
        The Spearman correlation coefficient and its p-value.
    """
    correlation, pval = spearmanr(probs.flatten(), correct_preds.flatten())
    print(f'Spearman: {correlation:.3f}, {pval:.3E}')


def get_residuals(preds, labels):
    """
    Calculate the absolute residuals between predictions and labels.

    Args:
        preds (np.array): Predicted values.
        labels (np.array): True labels.

    Returns:
        np.array: Absolute residuals
    """
    return np.abs(labels.flatten() - preds.flatten())


def compute_metrics_regression(p: EvalPrediction):
    """
    Compute various regression metrics for model evaluation.

    Args:
        (p: EvalPrediction): An object containing predictions and label ids.

    Returns:
        dict: A dictionary containing the following metrics:
            - r_squared: Coefficient of determination
            - spearman_rho: Spearman's rank correlation coefficient
            - spear_pval: p-value for Spearman's correlation
            - pearson_rho: Pearson correlation coefficient
            - pear_pval: p-value for Pearson's correlation
            - mse: Mean Squared Error
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    logits = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    r2 = r2_score(labels, logits)
    spearman_rho, spear_pval = spearmanr(logits, labels)
    pearson_rho, pear_pval = pearsonr(logits, labels)
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    rmse = np.sqrt(mse)

    return {
        'r_squared': round(r2, 5),
        'spearman_rho': round(spearman_rho, 5),
        'spear_pval': round(spear_pval, 5),
        'pearson_rho': round(pearson_rho, 5),
        'pear_pval': round(pear_pval, 5),
        'mse': round(mse, 5),
        'mae': round(mae, 5),
        'rmse': round(rmse, 5),
    }

# Test code
if __name__ == "__main__":
    print("Testing regression metric functions:")
    
    # Generate random predictions and labels
    np.random.seed(42)  # for reproducibility
    num_samples = 1000
    true_labels = np.random.rand(num_samples) * 10  # Random labels between 0 and 10
    predictions = true_labels + np.random.normal(0, 1, num_samples)  # Predictions with some noise
    
    # Test spearman function
    print("\nTesting spearman function:")
    spearman(predictions, true_labels)
    
    # Test get_residuals function
    print("\nTesting get_residuals function:")
    residuals = get_residuals(predictions, true_labels)
    print(f"Mean residual: {np.mean(residuals):.4f}")
    print(f"Max residual: {np.max(residuals):.4f}")
    
    # Test compute_metrics_regression function
    print("\nTesting compute_metrics_regression function:")
    eval_pred = EvalPrediction(predictions=predictions, label_ids=(None, true_labels))
    metrics = compute_metrics_regression(eval_pred)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Test with perfect predictions
    print("\nTesting with perfect predictions:")
    perfect_eval_pred = EvalPrediction(predictions=true_labels, label_ids=(None, true_labels))
    perfect_metrics = compute_metrics_regression(perfect_eval_pred)
    
    for metric, value in perfect_metrics.items():
        print(f"{metric}: {value}")
    
    # Test with completely incorrect predictions
    print("\nTesting with completely incorrect predictions:")
    incorrect_predictions = 10 - true_labels  # Invert the predictions
    incorrect_eval_pred = EvalPrediction(predictions=incorrect_predictions, label_ids=(None, true_labels))
    incorrect_metrics = compute_metrics_regression(incorrect_eval_pred)
    
    for metric, value in incorrect_metrics.items():
        print(f"{metric}: {value}")
