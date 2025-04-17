from transformers import EvalPrediction
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef


def compute_autoencoder_metrics(eval_preds: EvalPrediction):
    predictions, labels = eval_preds.predictions, eval_preds.label_ids
    logits, mse_loss, lm_loss, div_loss = predictions
    # Compute f1 score
    y_pred = logits.argmax(axis=-1).flatten()
    y_true = labels.flatten()
    valid_indices = y_true != -100
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {
        "mse_loss": mse_loss.mean().item(),
        "lm_loss": lm_loss.mean().item(),
        "div_loss": div_loss.mean().item(),
        "f1": f1,
    }


def compute_lm_metrics_with_logits(eval_preds: EvalPrediction):
    logits = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
    labels = eval_preds.label_ids
    # Compute f1 score
    y_pred = logits.argmax(axis=-1).flatten()
    y_true = labels.flatten()
    valid_indices = y_true != -100
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "f1": f1,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "mcc": mcc,
    }


def compute_lm_metrics_with_preds(eval_preds: EvalPrediction):
    logits = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
    labels = eval_preds.label_ids
    # Compute f1 score
    y_pred = logits.flatten()
    y_true = labels.flatten()
    valid_indices = y_true != -100
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "f1": f1,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "mcc": mcc,
    }