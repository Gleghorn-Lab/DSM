import copy
import torch

from accelerate.utils import extract_model_from_parallel
from dataclasses import dataclass


@dataclass
class TrainerCallbackState:
    global_step: int
    max_steps: int
    epoch: int


class DSM2TrainerCallback:
    def on_train_begin(self, state: TrainerCallbackState, model: torch.nn.Module):
        return

    def on_step_begin(self, state: TrainerCallbackState, model: torch.nn.Module):
        return

    def on_step_end(self, state: TrainerCallbackState, model: torch.nn.Module):
        return

    def on_train_end(self, state: TrainerCallbackState, model: torch.nn.Module):
        return


class EMATeacherCallback(DSM2TrainerCallback):
    def __init__(self, total_steps: int, ema_start_percent: float, ema_decay: float):
        self.total_steps = total_steps
        self.ema_start_percent = ema_start_percent
        self.ema_decay = ema_decay
        self.ema_active = False

    def on_step_begin(self, state: TrainerCallbackState, model: torch.nn.Module):
        start_step = int(self.total_steps * self.ema_start_percent)
        if (state.global_step >= start_step) and (not self.ema_active):
            self.ema_active = True
            print(f"Initializing EMA Teacher at step {state.global_step}")

            unwrapped_model = extract_model_from_parallel(model)
            ema_teacher = copy.deepcopy(unwrapped_model)
            for param in ema_teacher.parameters():
                param.requires_grad = False
            ema_teacher.eval()
            unwrapped_model.ema_teacher = ema_teacher

    def on_step_end(self, state: TrainerCallbackState, model: torch.nn.Module):
        if self.ema_active:
            unwrapped_model = extract_model_from_parallel(model)
            ema_teacher = unwrapped_model.ema_teacher
            assert ema_teacher is not None, "EMA teacher should be initialized before EMA updates."
            with torch.no_grad():
                for student_param, teacher_param in zip(unwrapped_model.parameters(), ema_teacher.parameters()):
                    teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1.0 - self.ema_decay)
    