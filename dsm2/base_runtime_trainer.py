import math
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from typing import Sequence

from dsm2.dsm2_callbacks import DSM2TrainerCallback, TrainerCallbackState
from dsm2.dsm2_config import DSM2OptimizationConfig, DSM2RuntimeConfig
from dsm2.model_utils import extract_model_from_parallel
from dsm2.trainer_utils import infer_rank_world_size_local_rank


class BaseRuntimeTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimization_config: DSM2OptimizationConfig,
        runtime_config: DSM2RuntimeConfig,
        callbacks: Sequence[DSM2TrainerCallback],
        wandb_enabled: bool,
        wandb_module,
    ):
        self.model = model
        self.optimization_config = optimization_config
        self.runtime_config = runtime_config
        self.callbacks = list(callbacks)
        self.wandb_enabled = wandb_enabled
        self.wandb_module = wandb_module

        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_distributed = False
        self.is_main_process = True
        self.device = torch.device("cpu")

        self.global_step = 0
        self.epoch = 0
        self.optimizer = None
        self.scheduler = None
        self._owns_process_group = False
        self._is_prepared = False

        self.output_dir = self.runtime_config.save_path.split("/")[-1]

    def create_optimizer(self):
        raise NotImplementedError("Subclasses must implement create_optimizer().")

    def train_step(self, patches):
        raise NotImplementedError("Subclasses must implement train_step().")

    def eval_step(self, patches):
        raise NotImplementedError("Subclasses must implement eval_step().")

    def _callback_state(self) -> TrainerCallbackState:
        return TrainerCallbackState(
            global_step=self.global_step,
            max_steps=self.optimization_config.max_steps,
            epoch=self.epoch,
        )

    def _dispatch_on_train_begin(self):
        state = self._callback_state()
        for callback in self.callbacks:
            callback.on_train_begin(state=state, model=self.model)

    def _dispatch_on_step_begin(self):
        state = self._callback_state()
        for callback in self.callbacks:
            callback.on_step_begin(state=state, model=self.model)

    def _dispatch_on_step_end(self):
        state = self._callback_state()
        for callback in self.callbacks:
            callback.on_step_end(state=state, model=self.model)

    def _dispatch_on_train_end(self):
        state = self._callback_state()
        for callback in self.callbacks:
            callback.on_train_end(state=state, model=self.model)

    def _configure_distributed(self):
        env_rank, env_world_size, env_local_rank = infer_rank_world_size_local_rank()

        if self.runtime_config.init_distributed and (env_world_size > 1) and (not dist.is_initialized()):
            dist.init_process_group(backend=self.runtime_config.distributed_backend, init_method="env://")
            self._owns_process_group = True

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = env_local_rank
            self.is_distributed = self.world_size > 1
        else:
            self.rank = env_rank
            self.world_size = env_world_size
            self.local_rank = env_local_rank
            self.is_distributed = False

        self.is_main_process = self.rank == 0

        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _prepare_model(self):
        self.model = self.model.to(self.device)
        self.model = torch.compile(self.model)

        if self.is_distributed:
            if self.device.type == "cuda":
                self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                self.model = DistributedDataParallel(self.model)

    def _create_scheduler(self):
        max_steps = self.optimization_config.max_steps
        assert max_steps > 0, "max_steps must be > 0."
        warmup_steps = max(1, int(max_steps * 0.01))
        plateau_steps = max(1, int(max_steps * 0.49))
        cooldown_steps = max(1, int(max_steps * 0.30))
        total_phased_steps = warmup_steps + plateau_steps + cooldown_steps
        assert total_phased_steps <= max_steps, (
            f"Scheduler phases exceed max_steps: warmup={warmup_steps}, plateau={plateau_steps}, "
            f"cooldown={cooldown_steps}, max_steps={max_steps}."
        )
        cooldown_start = warmup_steps + plateau_steps
        cooldown_end = cooldown_start + cooldown_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            if current_step < cooldown_start:
                return 1.0
            if current_step < cooldown_end:
                cooldown_progress = float(current_step - cooldown_start + 1) / float(cooldown_steps)
                return 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
            if current_step >= max_steps:
                return 0.0
            return 0.0

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _barrier(self):
        if self.is_distributed:
            dist.barrier()

    def _unwrap_model(self) -> torch.nn.Module:
        return extract_model_from_parallel(self.model, keep_torch_compile=False)

    def _save_checkpoint(self, global_step: int):
        if self.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-step-{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            saveable_model = self._unwrap_model()
            saveable_model.save_pretrained(checkpoint_dir)
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
            torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

        self._barrier()

    def prep_for_training(self):
        self._configure_distributed()
        self._prepare_model()
        self.optimizer = self.create_optimizer()
        self._create_scheduler()
        self.optimizer.zero_grad(set_to_none=True)
        self._is_prepared = True

    def _log_prefixed_metrics(self, prefix: str, metrics: dict[str, float]):
        if self.wandb_enabled and self.is_main_process:
            prefixed_metrics = {}
            for metric_name in metrics:
                prefixed_metrics[f"{prefix}/{metric_name}"] = metrics[metric_name]
            self.wandb_module.log(prefixed_metrics, step=self.global_step)

    def shutdown(self):
        if self._owns_process_group and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
