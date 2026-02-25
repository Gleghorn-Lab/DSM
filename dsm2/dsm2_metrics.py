import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from transformers import EvalPrediction

from models.alignment_helpers import GetAlignmentScoreFromLogits


class ComputeDSM2Metrics:
    def __init__(self, tokenizer):
        self.alignment_scorer = GetAlignmentScoreFromLogits(tokenizer)
        self.expected_vocab_size = tokenizer.vocab_size

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _extract_prediction_tensors(self, predictions):
        if isinstance(predictions, (tuple, list)):
            candidate_values = list(predictions)
        else:
            candidate_values = [predictions]

        tensors = []
        for value in candidate_values:
            if isinstance(value, (torch.Tensor, np.ndarray)):
                tensors.append(value)
        return tensors

    def _select_logits_and_mask_labels(self, predictions):
        tensors = self._extract_prediction_tensors(predictions)

        lm_logits = None
        mask_labels = None

        for tensor in tensors:
            if len(tensor.shape) == 3 and tensor.shape[-1] == self.expected_vocab_size:
                lm_logits = tensor
            if len(tensor.shape) == 2 and mask_labels is None:
                mask_labels = tensor

        if lm_logits is None:
            for tensor in tensors:
                if len(tensor.shape) == 3:
                    lm_logits = tensor
                    break

        assert lm_logits is not None, "Could not extract 3D language-model logits from eval predictions."
        return lm_logits, mask_labels

    def __call__(self, eval_preds: EvalPrediction):
        metrics = {}
        lm_logits, mask_labels = self._select_logits_and_mask_labels(eval_preds.predictions)

        if isinstance(eval_preds.label_ids, (tuple, list)):
            input_ids = eval_preds.label_ids[0]
        else:
            input_ids = eval_preds.label_ids

        labels_to_use = input_ids if mask_labels is None else mask_labels

        lm_logits_np = self._to_numpy(lm_logits)
        input_ids_np = self._to_numpy(input_ids)
        labels_np = self._to_numpy(labels_to_use)
        scores = self.alignment_scorer.batched_call(lm_logits_np, input_ids_np)

        lm_logits_torch = torch.as_tensor(lm_logits_np)
        labels_torch = torch.as_tensor(labels_np).long()

        cross_entropy_loss = F.cross_entropy(
            lm_logits_torch.view(-1, lm_logits_torch.shape[-1]),
            labels_torch.view(-1),
            ignore_index=-100,
        )

        metrics["cross_entropy_loss"] = float(cross_entropy_loss.item())
        metrics["alignment_score"] = float(scores.mean())

        y_pred = lm_logits_np.argmax(axis=-1).flatten()
        y_true = labels_np.flatten()
        valid_indices = y_true != -100

        if valid_indices.any():
            y_pred = y_pred[valid_indices]
            y_true = y_true[valid_indices]
            metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["prec"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["rec"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["acc"] = float(accuracy_score(y_true, y_pred))
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        else:
            metrics["f1"] = 0.0
            metrics["prec"] = 0.0
            metrics["rec"] = 0.0
            metrics["acc"] = 0.0
            metrics["mcc"] = 0.0

        return metrics
