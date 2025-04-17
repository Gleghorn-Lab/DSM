import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
try:
    from .max_metrics import max_metrics
except:
    from max_metrics import max_metrics


def compute_metrics_multi_label_classification(p: EvalPrediction):
    '''
        Compute evaluation metrics for multi-label classification tasks.

    This function calculates various performance metrics for multi-label classification,
    including accuracy, F1 score, precision, recall, hamming loss, threshold,
    Matthews Correlation Coefficient (MCC), and Area Under the ROC Curve (AUC).

    Args:
        (p: EvalPrediction): An object containing predictions and label ids.

    Returns:
        dict: A dictionary containing the computed metrics:
        - 'accuracy': The proportion of correct predictions.
        - 'f1': The F1 score
        - 'precision': The proportion of true positive predictions among all positive predictions.
        - 'recall': The proportion of true positive predictions among all actual positive instances.
        - 'hamming_loss': The proportion of wrong labels to the total number of labels.
        - 'threshold': The optimal threshold for classification.
        - 'mcc': Matthews Correlation Coefficient, a balanced measure for binary and multiclass classification.
        - 'auc': Area Under the ROC Curve, measuring the ability to distinguish between classes.
    '''
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    preds = np.array(preds)
    labels = np.array(labels)

    preds = torch.tensor(preds)
    y_true = torch.tensor(labels, dtype=torch.int)

    probs = F.softmax(preds, dim=-1)
    y_pred = (probs > 0.5).int()

    f1, prec, recall, thres = max_metrics(probs, y_true)
    y_pred, y_true = y_pred.flatten(), y_true.flatten()
    accuracy = accuracy_score(y_pred, y_true)
    hamming = hamming_loss(y_pred, y_true)
    
    # Calculate MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calculate AUC
    auc = roc_auc_score(y_true, probs.flatten(), average='macro', multi_class='ovr')

    return {
        'accuracy': round(accuracy, 5),
        'f1': round(f1, 5),
        'precision': round(prec, 5),
        'recall': round(recall, 5),
        'hamming_loss': round(hamming, 5),
        'threshold': round(thres, 5),
        'mcc': round(mcc, 5),
        'auc': round(auc, 5)
    }



def compute_metrics_single_label_classification(p: EvalPrediction):
    """
    Compute evaluation metrics for single-label classification tasks.

    This function calculates various performance metrics for single-label classification,
    and prints a confusion matrix.

    Args:
        (p: EvalPrediction): An object containing predictions and label ids.

    Returns:
        dict: A dictionary containing the computed metrics:
            - 'f1': The F1 score.
            - 'precision': The proportion of true positive predictions among all positive predictions.
            - 'recall': The proportion of true positive predictions among all actual positive instances.
            - 'accuracy': The proportion of correct predictions.
            - 'mcc': Matthews Correlation Coefficient, a balanced measure for binary and multiclass classification.
            - 'auc': Area Under the ROC Curve, measuring the ability to distinguish between classes.

    Note:
        The function handles cases where some labels are marked as -100 (ignored).
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    preds = torch.tensor(np.array(preds))
    y_true = torch.tensor(np.array(labels), dtype=torch.int)

    if y_true.size() == preds.size():
        y_pred = (preds > 0.5).int().flatten()
        n_classes = 2
        probs = None
    else:
        n_classes = preds.shape[-1]
        y_pred = preds.argmax(dim=-1).flatten()
        probs = y_pred.clone()

    all_classes = np.arange(n_classes)

    y_true = y_true.flatten()

    valid_indices = y_true != -100

    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]

    # Convert to numpy for sklearn metrics
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    cm = confusion_matrix(y_true_np, y_pred_np, labels=all_classes)
    print("\nConfusion Matrix:")
    print(cm)

    f1 = f1_score(y_true_np, y_pred_np, average='weighted', labels=all_classes, zero_division=0)
    precision = precision_score(y_true_np, y_pred_np, average='weighted', labels=all_classes, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, average='weighted', labels=all_classes, zero_division=0)
    accuracy = accuracy_score(y_true_np, y_pred_np)
    mcc = matthews_corrcoef(y_true_np, y_pred_np)
    
    # AUC calculation
    if probs != None:
        probs_np = probs.numpy()
        if n_classes == 2:  # Binary classification
            try:
                auc = roc_auc_score(y_true_np, y_pred_np)
            except:
                auc = -100
        else:  # Multiclass classification
            # Binarize the true labels
            y_true_bin = label_binarize(y_true_np, classes=all_classes)
            # Compute AUC for each class
            try:
                auc = roc_auc_score(y_true_bin, probs_np, average='weighted', multi_class='ovr')
            except:
                auc = -100
    else:
        auc = -100

    return {
        'f1': round(f1, 5),
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'accuracy': round(accuracy, 5),
        'mcc': round(mcc, 5),
        'auc': round(auc, 5)
    }


if __name__ == "__main__":
    print("Testing multi-label classification metrics:")
    num_samples, num_classes = 1000, 5
    preds = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, 2, (num_samples, num_classes))
    
    eval_pred = EvalPrediction(predictions=preds.numpy(), label_ids=(None, labels.numpy()))
    metrics = compute_metrics_multi_label_classification(eval_pred)
    print(metrics)
    
    print("\nTesting single-label classification metrics:")
    num_samples, num_classes = 1000, 5
    preds = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    eval_pred = EvalPrediction(predictions=preds.numpy(), label_ids=(None, labels.numpy()))
    metrics = compute_metrics_single_label_classification(eval_pred)
    print(metrics)
    
    print("\nTesting single-label classification metrics with -100 labels:")
    labels[::10] = -100  # Set every 10th label to -100
    eval_pred = EvalPrediction(predictions=preds.numpy(), label_ids=(None, labels.numpy()))
    metrics = compute_metrics_single_label_classification(eval_pred)
    print(metrics)