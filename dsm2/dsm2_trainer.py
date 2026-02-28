import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Sequence

from dsm2.base_patch_batch_trainer import BasePatchBatchTrainer
from dsm2.dsm2_callbacks import DSM2TrainerCallback
from dsm2.dsm2_config import DSM2LossConfig, DSM2OptimizationConfig, DSM2RuntimeConfig
from dsm2.losses import contrastive_loss_from_pooled, pool_states
from dsm2.model_utils import extract_model_from_parallel
from dsm2.dsm2_optim import MuonAdamWWrapper, create_muonclip_optimizer, partition_dsm2_parameters


class DSM2Trainer(BasePatchBatchTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        loss_config: DSM2LossConfig,
        optimization_config: DSM2OptimizationConfig,
        runtime_config: DSM2RuntimeConfig,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        compute_metrics,
        callbacks: Sequence[DSM2TrainerCallback],
        wandb_enabled: bool,
        wandb_module,
    ):
        assert loss_config.patch_size > 0, "DSM2 custom trainer requires patch_size > 0."
        assert 0.0 <= loss_config.teacher_free_percent <= 1.0, (
            f"teacher_free_percent must be in [0.0, 1.0], got {loss_config.teacher_free_percent}."
        )
        assert 0.0 <= loss_config.aux_loss_warmup_percent <= 1.0, (
            f"aux_loss_warmup_percent must be in [0.0, 1.0], got {loss_config.aux_loss_warmup_percent}."
        )
        assert loss_config.max_aux_to_ce_ratio >= 0.0, (
            f"max_aux_to_ce_ratio must be >= 0.0, got {loss_config.max_aux_to_ce_ratio}."
        )
        self.teacher_model = teacher_model
        self.loss_config = loss_config
        self.teacher_free_steps = int(optimization_config.max_steps * loss_config.teacher_free_percent)
        self.aux_loss_warmup_steps = int(optimization_config.max_steps * loss_config.aux_loss_warmup_percent)
        self.ema_cleanup_complete = False

        super().__init__(
            model=model,
            patch_size=loss_config.patch_size,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            compute_metrics=compute_metrics,
            optimization_config=optimization_config,
            runtime_config=runtime_config,
            callbacks=callbacks,
            wandb_enabled=wandb_enabled,
            wandb_module=wandb_module,
        )

    def prep_for_training(self):
        super().prep_for_training()
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

        unwrapped_model = extract_model_from_parallel(self.model)
        unwrapped_model.ema_teacher = None

    def create_optimizer(self):
        if self.optimizer is None:
            unwrapped_model = extract_model_from_parallel(self.model, keep_torch_compile=False)
            if self.optimization_config.use_muon:
                muon_params, adamw_params, attention_params = partition_dsm2_parameters(unwrapped_model)
                muonclip = create_muonclip_optimizer(
                    model=unwrapped_model,
                    muon_params=muon_params,
                    attention_params=attention_params,
                    muon_lr=self.optimization_config.muon_lr,
                    muon_tau=self.optimization_config.muon_tau,
                )
                adamw = torch.optim.AdamW(
                    adamw_params,
                    lr=self.optimization_config.learning_rate,
                    weight_decay=0.0,
                )
                self.optimizer = MuonAdamWWrapper(muonclip, adamw)
            else:
                self.optimizer = torch.optim.AdamW(
                    unwrapped_model.parameters(),
                    lr=self.optimization_config.learning_rate,
                    weight_decay=0.0,
                )

        return self.optimizer

    def _select_active_teacher(self, unwrapped_model):
        if unwrapped_model.ema_teacher is None:
            return self.teacher_model
        self._cleanup_after_ema_start(unwrapped_model)
        return unwrapped_model.ema_teacher

    def _cleanup_after_ema_start(self, unwrapped_model):
        if self.ema_cleanup_complete:
            return

        assert self.teacher_model is not None, "teacher_model must exist before EMA cleanup."
        print("EMA active: dropping teacher projections and original teacher model.")

        student_base_model = extract_model_from_parallel(self.model, keep_torch_compile=False)
        if student_base_model.teacher_projections is not None:
            projection_params = list(student_base_model.teacher_projections.parameters())
            if isinstance(self.optimizer, MuonAdamWWrapper):
                self.optimizer.remove_params(projection_params)
            student_base_model.teacher_projections = None
            print("EMA cleanup: removed student teacher_projections.")
        else:
            print("EMA cleanup: student teacher_projections already absent (pretrained path).")

        ema_teacher = unwrapped_model.ema_teacher
        assert ema_teacher is not None, "ema_teacher must exist before cleanup."
        ema_base_model = extract_model_from_parallel(ema_teacher, keep_torch_compile=False)
        if ema_base_model.teacher_projections is not None:
            ema_base_model.teacher_projections = None
            print("EMA cleanup: removed EMA teacher_projections.")
        else:
            print("EMA cleanup: EMA teacher_projections already absent.")

        self.teacher_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.ema_cleanup_complete = True

    def _forward_teacher_patch(self, active_teacher, patch_input_ids: torch.Tensor, patch_attention_mask: torch.Tensor):
        with torch.no_grad():
            if active_teacher is self.teacher_model:
                teacher_outputs = active_teacher(
                    input_ids=patch_input_ids,
                    attention_mask=patch_attention_mask,
                    output_hidden_states=True,
                    output_attentions=False,
                )
                teacher_hidden_states = teacher_outputs.hidden_states
                assert teacher_hidden_states is not None, "Teacher must return hidden states for DSM2 training."
                teacher_hidden_states = tuple(teacher_hidden_states)
                if len(teacher_hidden_states) > self.teacher_model.config.num_hidden_layers:
                    teacher_hidden_states = teacher_hidden_states[1:]
                return tuple(hidden_state.detach() for hidden_state in teacher_hidden_states)

            teacher_outputs = active_teacher(
                input_ids=patch_input_ids,
                attention_mask=patch_attention_mask,
                alpha_ce=0.0,
                alpha_jepa=0.0,
                alpha_contrastive=0.0,
            )
            teacher_hidden_states = teacher_outputs.student_hidden_states
            assert teacher_hidden_states is not None, "EMA teacher must return student_hidden_states for DSM2 training."
            teacher_hidden_states = tuple(teacher_hidden_states)
            return tuple(hidden_state.detach() for hidden_state in teacher_hidden_states)

    def _aggregate_s_max(self, all_s_max_patches):
        assert len(all_s_max_patches) > 0, "Cannot aggregate s_max from an empty patch list."
        num_layers = len(all_s_max_patches[0])
        assert num_layers > 0, "s_max must contain at least one transformer layer."
        num_heads = len(all_s_max_patches[0][0])
        assert num_heads > 0, "s_max must contain at least one attention head."

        for patch_idx in range(len(all_s_max_patches)):
            patch_s_max = all_s_max_patches[patch_idx]
            assert len(patch_s_max) == num_layers, (
                f"s_max patch {patch_idx} has {len(patch_s_max)} layers but expected {num_layers}."
            )
            for layer_idx in range(num_layers):
                assert len(patch_s_max[layer_idx]) == num_heads, (
                    f"s_max patch {patch_idx}, layer {layer_idx} has {len(patch_s_max[layer_idx])} heads "
                    f"but expected {num_heads}."
                )

        reduced_s_max = []
        for layer_idx in range(num_layers):
            layer_head_maxes = []
            for head_idx in range(num_heads):
                head_values = [patch[layer_idx][head_idx] for patch in all_s_max_patches]
                layer_head_maxes.append(torch.stack(head_values).max())
            reduced_s_max.append(layer_head_maxes)
        return reduced_s_max

    def _set_optimizer_s_max(self, reduced_s_max):
        assert isinstance(self.optimizer, MuonAdamWWrapper), f"Expected MuonAdamWWrapper, got {type(self.optimizer)}."
        self.optimizer.last_s_max = reduced_s_max

    def _aux_loss_warmup_scale(self, training: bool, use_teacher: bool) -> float:
        if (not use_teacher) and training:
            return 0.0
        if not training:
            return 1.0 if use_teacher else 0.0
        if self.aux_loss_warmup_steps <= 0:
            return 1.0
        progress = float(self.global_step - self.teacher_free_steps) / float(self.aux_loss_warmup_steps)
        return float(max(0.0, min(1.0, progress)))

    def _run_patch_group(self, patches: List[Dict[str, torch.Tensor]], training: bool):
        batch_size = 0
        for patch in patches:
            batch_size += int(patch["input_ids"].size(0))
        assert batch_size > 0, "Patch group must contain at least one sample."

        total_ce_loss = 0.0
        total_jepa_loss = 0.0
        total_contrastive_loss = 0.0

        all_teacher_pooled = []
        all_student_pooled = []
        all_s_max_patches = []
        all_logits = []
        all_mask_labels = []
        all_input_ids = []

        use_teacher = (not training) or (self.global_step >= self.teacher_free_steps)
        aux_loss_scale = self._aux_loss_warmup_scale(training=training, use_teacher=use_teacher)
        scaled_alpha_jepa = self.loss_config.alpha_jepa * aux_loss_scale
        scaled_alpha_contrastive = self.loss_config.alpha_contrastive * aux_loss_scale
        needs_teacher_states = use_teacher and ((scaled_alpha_jepa > 0.0) or (scaled_alpha_contrastive > 0.0))
        active_teacher = None
        if needs_teacher_states:
            unwrapped_model = extract_model_from_parallel(self.model)
            active_teacher = self._select_active_teacher(unwrapped_model)

        for patch in patches:
            patch_input_ids = patch["input_ids"].to(self.device, non_blocking=True)
            patch_attention_mask = patch["attention_mask"].to(self.device, non_blocking=True)
            current_patch_size = int(patch_input_ids.size(0))

            teacher_hidden_states = None
            alpha_jepa = 0.0
            if needs_teacher_states:
                assert active_teacher is not None, "active_teacher must be set when teacher losses are enabled."
                teacher_hidden_states = self._forward_teacher_patch(active_teacher, patch_input_ids, patch_attention_mask)
                alpha_jepa = scaled_alpha_jepa
            dsm2_patch_output = self.model(
                input_ids=patch_input_ids,
                attention_mask=patch_attention_mask,
                teacher_hidden_states=teacher_hidden_states,
                alpha_ce=self.loss_config.alpha_ce,
                alpha_jepa=alpha_jepa,
                alpha_contrastive=0.0,
                output_s_max=training and self.optimization_config.use_muon,
            )

            weight = float(current_patch_size) / float(batch_size)
            if dsm2_patch_output.ce_loss is not None:
                total_ce_loss += dsm2_patch_output.ce_loss * weight
            if dsm2_patch_output.jepa_loss is not None:
                total_jepa_loss += dsm2_patch_output.jepa_loss * weight

            if needs_teacher_states and (scaled_alpha_contrastive > 0.0):
                assert teacher_hidden_states is not None, "teacher_hidden_states must be set when contrastive loss is enabled."
                with torch.no_grad():
                    teacher_pooled = pool_states(teacher_hidden_states, attention_mask=patch_attention_mask)
                student_hidden_states = dsm2_patch_output.student_hidden_states
                assert student_hidden_states is not None, "DSM2 output must contain student_hidden_states for contrastive loss."
                student_pooled = pool_states(student_hidden_states, attention_mask=patch_attention_mask)
                all_teacher_pooled.append(teacher_pooled)
                all_student_pooled.append(student_pooled)

            if training and self.optimization_config.use_muon:
                assert dsm2_patch_output.s_max is not None, "Training step requires s_max for MuonClip."
                all_s_max_patches.append(dsm2_patch_output.s_max)

            assert dsm2_patch_output.logits is not None, "DSM2 output must contain logits."
            assert dsm2_patch_output.mask_labels is not None, "DSM2 output must contain mask labels."
            all_logits.append(dsm2_patch_output.logits)
            all_mask_labels.append(dsm2_patch_output.mask_labels)
            all_input_ids.append(patch_input_ids)

        if needs_teacher_states and (scaled_alpha_contrastive > 0.0) and (len(all_teacher_pooled) > 0):
            stacked_teacher_pooled = torch.cat(all_teacher_pooled, dim=1)
            stacked_student_pooled = torch.cat(all_student_pooled, dim=1)
            total_contrastive_loss = contrastive_loss_from_pooled(
                s_pooled=stacked_student_pooled,
                t_pooled=stacked_teacher_pooled,
            )

        if training and self.optimization_config.use_muon:
            reduced_s_max = self._aggregate_s_max(all_s_max_patches)
            self._set_optimizer_s_max(reduced_s_max)

        if not isinstance(total_ce_loss, torch.Tensor):
            total_ce_loss = torch.tensor(total_ce_loss, device=self.device, dtype=torch.float32)
        total_jepa_loss = torch.as_tensor(total_jepa_loss, device=self.device, dtype=total_ce_loss.dtype)
        total_contrastive_loss = torch.as_tensor(total_contrastive_loss, device=self.device, dtype=total_ce_loss.dtype)

        weighted_ce_loss = self.loss_config.alpha_ce * total_ce_loss
        weighted_jepa_loss = scaled_alpha_jepa * total_jepa_loss
        weighted_contrastive_loss = scaled_alpha_contrastive * total_contrastive_loss
        weighted_aux_loss_pre_guard = weighted_jepa_loss + weighted_contrastive_loss
        weighted_aux_loss = weighted_aux_loss_pre_guard
        aux_guard_scale = 1.0
        if training and (self.loss_config.max_aux_to_ce_ratio > 0.0):
            ce_term_magnitude = float(torch.abs(weighted_ce_loss.detach()).item())
            aux_term_magnitude = float(torch.abs(weighted_aux_loss.detach()).item())
            max_aux_magnitude = ce_term_magnitude * self.loss_config.max_aux_to_ce_ratio
            if (aux_term_magnitude > max_aux_magnitude) and (aux_term_magnitude > 0.0):
                aux_guard_scale = max_aux_magnitude / aux_term_magnitude
                weighted_aux_loss = weighted_aux_loss * aux_guard_scale

        loss = weighted_ce_loss + weighted_aux_loss

        metrics = {
            "ce_loss": float(total_ce_loss.detach().item()),
            "jepa_loss": float(total_jepa_loss.detach().item()),
            "contrastive_loss": float(total_contrastive_loss.detach().item()),
            "weighted_ce_loss": float(weighted_ce_loss.detach().item()),
            "weighted_jepa_loss": float(weighted_jepa_loss.detach().item()),
            "weighted_contrastive_loss": float(weighted_contrastive_loss.detach().item()),
            "weighted_aux_loss_pre_guard": float(weighted_aux_loss_pre_guard.detach().item()),
            "weighted_aux_loss": float(weighted_aux_loss.detach().item()),
            "aux_loss_scale": float(aux_loss_scale),
            "aux_guard_scale": float(aux_guard_scale),
        }

        eval_payload = {
            "logits": torch.cat(all_logits, dim=0),
            "mask_labels": torch.cat(all_mask_labels, dim=0),
            "input_ids": torch.cat(all_input_ids, dim=0),
        }
        return loss, metrics, eval_payload

    def train_step(self, patches):
        loss, train_metrics, _ = self._run_patch_group(patches=patches, training=True)
        return loss, train_metrics

    def eval_step(self, patches):
        loss, _, eval_payload = self._run_patch_group(patches=patches, training=False)
        return loss, eval_payload
