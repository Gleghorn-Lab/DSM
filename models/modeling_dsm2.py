import torch
import torch.nn.functional as F
from typing import Tuple

from FastPLMs.embedding_mixin import Pooler


def contrastive_loss(
    student_hidden_states: Tuple[torch.Tensor, ...],
    teacher_hidden_states: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    assert len(student_hidden_states) == len(teacher_hidden_states), "Student and teacher hidden states must have the same number of layers"

    pooler = Pooler(pooling_types=["mean", "var"])

    loss = 0
    for depth, (student_hidden_state, teacher_hidden_state) in enumerate(zip(student_hidden_states, teacher_hidden_states)):
        s = pooler(student_hidden_state) # (b, d)
        t = pooler(teacher_hidden_state) # (b, d)
        intra_student_reps = s.matmul(s.T) # (b, b)
        intra_teacher_reps = t.matmul(t.T) # (b, b)

        layer_loss = F.mse_loss(intra_student_reps, intra_teacher_reps) 
        # We weigh the later layers more heavily
        layer_loss *= (depth + 1) / len(student_hidden_states)
        loss += layer_loss

    return loss


def jepa_loss(
    student_hidden_states: Tuple[torch.Tensor, ...],
    teacher_hidden_states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    assert len(student_hidden_states) == len(teacher_hidden_states), "Student and teacher hidden states must have the same number of layers"

    