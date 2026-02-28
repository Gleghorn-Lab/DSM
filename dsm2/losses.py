import torch
from typing import Tuple

from models.FastPLMs.embedding_mixin import Pooler


def pool_states(hidden_states: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pools a tuple of hidden states using mean and var pooling over valid tokens.
    Returns stacked pooled states of shape (num_layers, b, 2d).
    """
    assert len(hidden_states) > 0, "hidden_states must contain at least one layer."
    assert attention_mask.ndim == 2, (
        f"attention_mask must have shape (batch_size, seq_len), got {attention_mask.shape}."
    )

    pooler = Pooler(pooling_types=["mean", "var"])
    stacked = torch.stack(hidden_states).float()
    mask = attention_mask.float()
    pooled = []
    for layer in stacked:
        pooled.append(pooler(layer, attention_mask=mask))  # (b, 2d)
    return torch.stack(pooled)  # (num_layers, b, 2d)


def contrastive_loss_from_pooled(
    s_pooled: torch.Tensor,
    t_pooled: torch.Tensor,
) -> torch.Tensor:
    """
    Computes depth-weighted contrastive loss from pre-pooled student and teacher representations.
    s_pooled, t_pooled: (num_layers, b, 2d)
    """
    assert s_pooled.shape == t_pooled.shape, (
        f"Student and teacher pooled representations must match; got {s_pooled.shape} vs {t_pooled.shape}."
    )
    num_layers, _, _ = s_pooled.shape
    assert num_layers > 0, "Pooled representations must contain at least one layer."
    s_pooled = s_pooled.float()
    t_pooled = t_pooled.float()

    # (num_layers, b, b)
    intra_student_reps = torch.bmm(s_pooled, s_pooled.transpose(1, 2)).softmax(dim=-1)
    intra_teacher_reps = torch.bmm(t_pooled, t_pooled.transpose(1, 2)).softmax(dim=-1)

    # (num_layers, b, b)
    squared_diff = (intra_student_reps - intra_teacher_reps) ** 2

    # Weight by depth
    depth_weights = torch.arange(1, num_layers + 1, device=squared_diff.device, dtype=squared_diff.dtype) / num_layers
    # (num_layers, 1, 1)
    depth_weights = depth_weights.view(num_layers, 1, 1)

    weighted_squared_diff = squared_diff * depth_weights

    # Average over layers and batch pairs
    layer_batch_loss = weighted_squared_diff.mean(dim=0).mean()  # scalar

    return layer_batch_loss


def contrastive_loss(
    student_hidden_states: Tuple[torch.Tensor, ...],
    teacher_hidden_states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a depth-weighted contrastive loss mapping student representations
    to teacher representations over valid (non-padding) tokens.
    """
    assert len(student_hidden_states) == len(teacher_hidden_states), (
        "Student and teacher hidden states must have the same number of layers."
    )
    s_pooled = pool_states(student_hidden_states, attention_mask=attention_mask)
    t_pooled = pool_states(teacher_hidden_states, attention_mask=attention_mask)
    return contrastive_loss_from_pooled(s_pooled, t_pooled)


def jepa_loss(
    student_hidden_states: Tuple[torch.Tensor, ...],
    teacher_hidden_states: Tuple[torch.Tensor, ...],
    distill_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes depth-weighted MSE between student and teacher hidden states
    over an explicit distillation token mask.
    """
    assert len(student_hidden_states) == len(teacher_hidden_states), (
        "Student and teacher hidden states must have the same number of layers."
    )
    num_layers = len(student_hidden_states)
    assert num_layers > 0, "student_hidden_states must contain at least one layer."
    assert distill_mask.ndim == 2, (
        f"distill_mask must have shape (batch_size, seq_len), got {distill_mask.shape}."
    )

    batch_size, seq_len, _ = student_hidden_states[0].shape
    assert distill_mask.shape == (batch_size, seq_len), (
        f"distill_mask shape {distill_mask.shape} must match hidden-state shape {(batch_size, seq_len)}."
    )
    mask = distill_mask.bool()
    assert mask.any(), "distill_mask must include at least one token."
    for layer_idx in range(num_layers):
        student_shape = student_hidden_states[layer_idx].shape
        teacher_shape = teacher_hidden_states[layer_idx].shape
        assert student_shape == teacher_shape, (
            f"Layer {layer_idx} shape mismatch between student and teacher: {student_shape} vs {teacher_shape}."
        )

    # Stack to (num_layers, b, seq_len, d) in fp32 for stability.
    s_stacked = torch.stack(student_hidden_states).float()
    t_stacked = torch.stack(teacher_hidden_states).float()

    # MSE per token, (num_layers, b, seq_len, d)
    squared_diff = (s_stacked - t_stacked) ** 2

    # Mean over hidden dimension, (num_layers, b, seq_len)
    mse_per_token = squared_diff.mean(dim=-1)

    # Weight by layer depth, (num_layers, 1, 1)
    depth_weights = torch.arange(1, num_layers + 1, device=squared_diff.device, dtype=squared_diff.dtype) / num_layers
    depth_weights = depth_weights.view(num_layers, 1, 1)

    weighted_mse = mse_per_token * depth_weights

    # Filter out padding tokens and sum up
    valid_mse = weighted_mse[:, mask]

    # Average over valid tokens and layers
    return valid_mse.mean()
