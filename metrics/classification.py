import numpy as np
import torch
from transformers import EvalPrediction
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


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
