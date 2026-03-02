import os
import torch
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from typing import Dict, List, Sequence
from tqdm.auto import tqdm

from dsm2.base_runtime_trainer import BaseRuntimeTrainer
from dsm2.callbacks import DSM2TrainerCallback
from dsm2.config import DSM2OptimizationConfig, DSM2RuntimeConfig
from dsm2.trainer_utils import gather_object_across_ranks, reduce_loss_sum_and_count, reduce_mean_float


class AsyncPatchGroupPrefetcher:
    def __init__(self, fetch_fn):
        self.fetch_fn = fetch_fn
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_future: Future | None = None
        self.is_shutdown = False

    def submit(self):
        assert not self.is_shutdown, "Cannot submit to a shutdown prefetcher."
        assert self.pending_future is None, "Prefetch already in flight; consume it before submitting another."
        self.pending_future = self.executor.submit(self.fetch_fn)

    def get(self):
        assert self.pending_future is not None, "No prefetched patch-group is available to consume."
        prefetched = self.pending_future.result()
        self.pending_future = None
        return prefetched

    def shutdown(self):
        if self.is_shutdown:
            return
        self.is_shutdown = True
        self.executor.shutdown(wait=True)


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

    def _build_progress_bar(self, total: int, desc: str):
        return tqdm(
            total=total,
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            leave=True,
            disable=not self.is_main_process,
        )

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

        is_windows = os.name == "nt"
        eval_num_workers = 0 if is_windows else self.optimization_config.dataloader_num_workers
        loader_kwargs = {
            "sampler": sampler,
            "batch_size": self.patch_size,
            "drop_last": False,
            "num_workers": eval_num_workers,
            "pin_memory": self.runtime_config.pin_memory and torch.cuda.is_available(),
            "collate_fn": self.train_loader.collate_fn,
        }
        if eval_num_workers > 0:
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
        train_progress = self._build_progress_bar(
            total=self.optimization_config.max_steps,
            desc="Training optimizer steps",
        )
        if self.is_main_process and self.global_step > 0:
            train_progress.update(self.global_step)

        try:
            while self.global_step < self.optimization_config.max_steps:
                self.epoch += 1

                if self.is_distributed and isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(self.epoch)

                self.model.train()
                train_iter = iter(self.train_loader)
                exhausted = False
                micro_step_idx = 0
                prefetcher = AsyncPatchGroupPrefetcher(lambda: self._accumulate_patch_group(train_iter))
                patches, exhausted = self._accumulate_patch_group(train_iter)
                try:
                    while (len(patches) > 0) and (self.global_step < self.optimization_config.max_steps):
                        if not exhausted:
                            prefetcher.submit()

                        micro_step_idx += 1
                        is_sync_step = (micro_step_idx % self.optimization_config.grad_accum == 0) or exhausted
                        if is_sync_step:
                            self._dispatch_on_step_begin()

                        sync_context = nullcontext()
                        if self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and (not is_sync_step):
                            sync_context = self.model.no_sync()

                        with sync_context:
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
                            train_progress.update(1)

                            if self.global_step % self.optimization_config.logging_steps == 0:
                                mean_metrics: Dict[str, float] = {}
                                mean_metrics["loss"] = float(sum(train_loss_window) / len(train_loss_window))
                                for metric_name in train_metric_windows:
                                    metric_history = train_metric_windows[metric_name]
                                    mean_metrics[metric_name] = float(sum(metric_history) / len(metric_history))

                                reduced_metrics = self._reduce_train_metrics(mean_metrics)
                                if self.is_main_process:
                                    learning_rate = float(self.scheduler.get_last_lr()[0])
                                    print(f"Train step {self.global_step}: {reduced_metrics}")
                                    train_progress.set_postfix(
                                        loss=f"{reduced_metrics['loss']:.4f}",
                                        lr=f"{learning_rate:.2e}",
                                    )
                                self._log_prefixed_metrics("train", reduced_metrics)

                                train_loss_window = []
                                train_metric_windows = {}

                            if self.global_step % eval_every == 0:
                                valid_metrics = self.evaluate(eval_dataset=self.valid_dataset, prefix="valid")
                                if self.is_main_process:
                                    print(f"Validation at step {self.global_step}: {valid_metrics}")

                            if self.global_step % self.optimization_config.save_every == 0:
                                self._save_checkpoint(self.global_step)

                        if self.global_step >= self.optimization_config.max_steps:
                            break
                        if exhausted:
                            patches = []
                        else:
                            patches, exhausted = prefetcher.get()
                finally:
                    prefetcher.shutdown()
        finally:
            train_progress.close()

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
        eval_progress = tqdm(
            total=len(loader),
            desc=f"{prefix} eval patches",
            unit="patch",
            dynamic_ncols=True,
            leave=False,
            disable=not self.is_main_process,
        )

        try:
            for patch in loader:
                loss, eval_payload = self.eval_step([patch])
                local_loss_sum += float(loss.item())
                local_loss_count += 1
                local_logits.append(eval_payload["logits"].detach().cpu())
                local_mask_labels.append(eval_payload["mask_labels"].detach().cpu())
                local_input_ids.append(eval_payload["input_ids"].detach().cpu())
                eval_progress.update(1)
                if self.is_main_process:
                    running_loss = local_loss_sum / float(local_loss_count)
                    eval_progress.set_postfix(loss=f"{running_loss:.4f}")
        finally:
            eval_progress.close()

        global_loss_sum, global_loss_count = reduce_loss_sum_and_count(
            local_loss_sum=local_loss_sum,
            local_loss_count=local_loss_count,
            device=self.device,
            is_distributed=self.is_distributed,
        )
        total_loss = global_loss_sum / float(max(1, global_loss_count))

        assert len(local_logits) > 0, "local_logits must contain at least one tensor."
        assert len(local_logits) == len(local_mask_labels) == len(local_input_ids), "local_logits, local_mask_labels, and local_input_ids must have the same length."
    
        local_eval_object = {
            "logits": torch.cat(local_logits, dim=0).float().numpy(),
            "mask_labels": torch.cat(local_mask_labels, dim=0).int().numpy(),
            "input_ids": torch.cat(local_input_ids, dim=0).int().numpy(),
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
