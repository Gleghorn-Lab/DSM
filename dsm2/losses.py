import torch
from typing import Any, Tuple

from models.FastPLMs.embedding_mixin import Pooler


def pool_states(hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Pools a tuple of hidden states natively using mean and var pooling.
    Returns stacked pooled states of shape (num_layers, b, 2d).
    """
    pooler = Pooler(pooling_types=["mean", "var"])
    stacked = torch.stack(hidden_states)
    pooled = []
    for layer in stacked:
        pooled.append(pooler(layer))  # (b, 2d)
    return torch.stack(pooled)  # (num_layers, b, 2d)


def contrastive_loss_from_pooled(
    s_pooled: torch.Tensor,
    t_pooled: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Computes depth-weighted contrastive loss from pre-pooled student and teacher representations.
    s_pooled, t_pooled: (num_layers, b, 2d)
    """
    num_layers, _, _ = s_pooled.shape

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
    **kwargs: Any,
) -> torch.Tensor:
    """
    Computes a depth-weighted contrastive loss mapping student representations
    to teacher representations, scaled by the inverse of the mask rate.
    """
    assert len(student_hidden_states) == len(teacher_hidden_states), "Student and teacher hidden states must have the same number of layers"
    s_pooled = pool_states(student_hidden_states)
    t_pooled = pool_states(teacher_hidden_states)
    return contrastive_loss_from_pooled(s_pooled, t_pooled)


def jepa_loss(
    student_hidden_states: Tuple[torch.Tensor, ...],
    teacher_hidden_states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Computes depth-weighted MSE between student and teacher hidden states for unmasked tokens,
    scaled by inverse mask rate.
    """
    assert len(student_hidden_states) == len(teacher_hidden_states), "Student and teacher hidden states must have the same number of layers"
    num_layers = len(student_hidden_states)

    mask = attention_mask.bool()  # (b, seq_len)

    # Stack to (num_layers, b, seq_len, d)
    s_stacked = torch.stack(student_hidden_states)
    t_stacked = torch.stack(teacher_hidden_states)

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
