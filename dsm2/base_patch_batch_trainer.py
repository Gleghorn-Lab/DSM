import numpy as np
import torch

from contextlib import nullcontext
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from typing import Any, Dict, List, Sequence

from dsm2.base_runtime_trainer import BaseRuntimeTrainer
from dsm2.dsm2_callbacks import DSM2TrainerCallback
from dsm2.dsm2_config import DSM2OptimizationConfig, DSM2RuntimeConfig
from dsm2.trainer_utils import gather_object_across_ranks, reduce_loss_sum_and_count, reduce_mean_float


class BasePatchBatchTrainer(BaseRuntimeTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        compute_metrics,
        optimization_config: DSM2OptimizationConfig,
        runtime_config: DSM2RuntimeConfig,
        callbacks: Sequence[DSM2TrainerCallback],
        wandb_enabled: bool,
        wandb_module,
    ):
        super().__init__(
            model=model,
            optimization_config=optimization_config,
            runtime_config=runtime_config,
            callbacks=callbacks,
            wandb_enabled=wandb_enabled,
            wandb_module=wandb_module,
        )

        assert patch_size > 0, "patch_size must be > 0."
        assert optimization_config.patch_accum > 0, "patch_accum must be > 0."
        assert optimization_config.grad_accum > 0, "grad_accum must be > 0."
        assert optimization_config.logging_steps > 0, "logging_steps must be > 0."
        assert optimization_config.save_every > 0, "save_every must be > 0."

        self.patch_size = patch_size
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.use_amp = runtime_config.use_amp and torch.cuda.is_available()

    def _accumulate_patch_group(self, data_iter) -> tuple[List[Dict[str, torch.Tensor]], bool]:
        patches: List[Dict[str, torch.Tensor]] = []
        exhausted = False

        for _ in range(self.optimization_config.patch_accum):
            try:
                patch = next(data_iter)
            except StopIteration:
                exhausted = True
                break
            patches.append(patch)

        return patches, exhausted

    def _create_eval_loader(self, eval_dataset: Dataset) -> DataLoader:
        if self.is_distributed:
            sampler = DistributedSampler(
                eval_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = RandomSampler(eval_dataset)

        loader_kwargs = {
            "sampler": sampler,
            "batch_size": self.patch_size,
            "drop_last": False,
            "num_workers": self.optimization_config.dataloader_num_workers,
            "pin_memory": self.runtime_config.pin_memory and torch.cuda.is_available(),
            "collate_fn": self.train_loader.collate_fn,
        }
        if self.optimization_config.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.optimization_config.dataloader_prefetch_factor

        return DataLoader(eval_dataset, **loader_kwargs)

    def _reduce_train_metrics(self, metric_values: Dict[str, float]) -> Dict[str, float]:
        reduced_metrics: Dict[str, float] = {}
        for metric_name in metric_values:
            reduced_metrics[metric_name] = reduce_mean_float(
                metric_values[metric_name],
                device=self.device,
                is_distributed=self.is_distributed,
                world_size=self.world_size,
            )
        return reduced_metrics

    def train(self):
        assert self._is_prepared, "prep_for_training() must be called before train()."

        eval_every = self.optimization_config.eval_every
        if eval_every <= 0:
            eval_every = self.optimization_config.save_every

        train_metric_windows: Dict[str, List[float]] = {}
        train_loss_window: List[float] = []
        self._dispatch_on_train_begin()

        while self.global_step < self.optimization_config.max_steps:
            self.epoch += 1

            if self.is_distributed and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(self.epoch)

            self.model.train()
            train_iter = iter(self.train_loader)
            exhausted = False
            micro_step_idx = 0

            while (not exhausted) and (self.global_step < self.optimization_config.max_steps):
                patches, exhausted = self._accumulate_patch_group(train_iter)
                if len(patches) == 0:
                    break

                micro_step_idx += 1
                is_sync_step = (micro_step_idx % self.optimization_config.grad_accum == 0) or exhausted
                if is_sync_step:
                    self._dispatch_on_step_begin()

                sync_context = nullcontext()
                if self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and (not is_sync_step):
                    sync_context = self.model.no_sync()

                amp_context = nullcontext()
                if self.use_amp:
                    amp_context = autocast("cuda", dtype=torch.bfloat16)

                with sync_context:
                    with amp_context:
                        loss, train_metrics = self.train_step(patches)
                        scaled_loss = loss / float(self.optimization_config.grad_accum)
                    scaled_loss.backward()

                train_loss_window.append(float(loss.detach().item()))
                for metric_name in train_metrics:
                    if metric_name not in train_metric_windows:
                        train_metric_windows[metric_name] = []
                    train_metric_windows[metric_name].append(float(train_metrics[metric_name]))

                if is_sync_step:
                    if self.optimization_config.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optimization_config.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    self._dispatch_on_step_end()

                    if self.global_step % self.optimization_config.logging_steps == 0:
                        mean_metrics: Dict[str, float] = {}
                        mean_metrics["loss"] = float(sum(train_loss_window) / len(train_loss_window))
                        for metric_name in train_metric_windows:
                            metric_history = train_metric_windows[metric_name]
                            mean_metrics[metric_name] = float(sum(metric_history) / len(metric_history))

                        reduced_metrics = self._reduce_train_metrics(mean_metrics)
                        if self.is_main_process:
                            print(f"Train step {self.global_step}: {reduced_metrics}")
                        self._log_prefixed_metrics("train", reduced_metrics)

                        train_loss_window = []
                        train_metric_windows = {}

                    if self.global_step % eval_every == 0:
                        valid_metrics = self.evaluate(eval_dataset=self.valid_dataset, prefix="valid")
                        if self.is_main_process:
                            print(f"Validation at step {self.global_step}: {valid_metrics}")

                    if self.global_step % self.optimization_config.save_every == 0:
                        self._save_checkpoint(self.global_step)

        self._dispatch_on_train_end()
        self._barrier()
        return self._unwrap_model()

    @torch.no_grad()
    def evaluate(self, eval_dataset: Dataset = None, prefix: str = "valid"):
        assert self._is_prepared, "prep_for_training() must be called before evaluate()."

        loader = self.valid_loader
        if eval_dataset is not None:
            if eval_dataset is self.valid_dataset:
                loader = self.valid_loader
            elif eval_dataset is self.test_dataset:
                loader = self.test_loader
            else:
                loader = self._create_eval_loader(eval_dataset)
        elif prefix == "test":
            loader = self.test_loader

        was_training = self.model.training
        self.model.eval()

        local_loss_sum = 0.0
        local_loss_count = 0
        local_logits: List[torch.Tensor] = []
        local_mask_labels: List[torch.Tensor] = []
        local_input_ids: List[torch.Tensor] = []

        for patch in loader:
            amp_context = nullcontext()
            if self.use_amp:
                amp_context = autocast("cuda", dtype=torch.bfloat16)

            with amp_context:
                loss, eval_payload = self.eval_step([patch])

            local_loss_sum += float(loss.item())
            local_loss_count += 1
            local_logits.append(eval_payload["logits"].detach().cpu())
            local_mask_labels.append(eval_payload["mask_labels"].detach().cpu())
            local_input_ids.append(eval_payload["input_ids"].detach().cpu())

        global_loss_sum, global_loss_count = reduce_loss_sum_and_count(
            local_loss_sum=local_loss_sum,
            local_loss_count=local_loss_count,
            device=self.device,
            is_distributed=self.is_distributed,
        )
        total_loss = global_loss_sum / float(max(1, global_loss_count))

        if len(local_logits) > 0:
            local_eval_object = {
                "logits": torch.cat(local_logits, dim=0).numpy(),
                "mask_labels": torch.cat(local_mask_labels, dim=0).numpy(),
                "input_ids": torch.cat(local_input_ids, dim=0).numpy(),
            }
        else:
            local_eval_object = {
                "logits": np.zeros((0, 1, 1), dtype=np.float32),
                "mask_labels": np.zeros((0, 1), dtype=np.int64),
                "input_ids": np.zeros((0, 1), dtype=np.int64),
            }

        gathered_eval_objects = gather_object_across_ranks(
            local_object=local_eval_object,
            is_distributed=self.is_distributed,
            world_size=self.world_size,
        )

        metrics: Dict[str, float] = {}
        if self.is_main_process:
            gathered_logits: List[torch.Tensor] = []
            gathered_mask_labels: List[torch.Tensor] = []
            gathered_input_ids: List[torch.Tensor] = []

            for eval_object in gathered_eval_objects:
                if eval_object["logits"].shape[0] > 0:
                    gathered_logits.append(torch.from_numpy(eval_object["logits"]))
                    gathered_mask_labels.append(torch.from_numpy(eval_object["mask_labels"]))
                    gathered_input_ids.append(torch.from_numpy(eval_object["input_ids"]))

            if len(gathered_logits) > 0:
                global_logits = torch.cat(gathered_logits, dim=0)
                global_mask_labels = torch.cat(gathered_mask_labels, dim=0)
                global_input_ids = torch.cat(gathered_input_ids, dim=0)
                metrics = self.compute_metrics.from_custom_outputs(
                    logits=global_logits,
                    mask_labels=global_mask_labels,
                    input_ids=global_input_ids,
                )
            else:
                metrics = self.compute_metrics.zero_metrics()

            metrics["loss"] = float(total_loss)
            self._log_prefixed_metrics(prefix, metrics)
            print(f"{prefix} metrics: {metrics}")

        if was_training:
            self.model.train()

        self._barrier()
        return metrics
