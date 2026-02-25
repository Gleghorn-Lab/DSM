import torch

from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import extract_model_from_parallel
from transformers import Trainer

from models.modeling_dsm2 import contrastive_loss_from_pooled, pool_states
from dsm2.dsm2_config import DSM2LossConfig, DSM2OptimizationConfig
from dsm2.dsm2_optim import MuonAdamWWrapper, create_muonclip_optimizer, partition_dsm2_parameters


class DSM2Trainer(Trainer):
    def __init__(
        self,
        teacher_model,
        loss_config: DSM2LossConfig,
        optimization_config: DSM2OptimizationConfig,
        wandb_enabled: bool,
        wandb_module,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss_config = loss_config
        self.optimization_config = optimization_config
        self.wandb_enabled = wandb_enabled
        self.wandb_module = wandb_module

        unwrapped_model = extract_model_from_parallel(self.model)
        unwrapped_model.ema_teacher = None

    def create_optimizer(self):
        if self.optimizer is None:
            unwrapped_model = extract_model_from_parallel(self.model)
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
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            self.optimizer = MuonAdamWWrapper(muonclip, adamw)
        return self.optimizer

    def _select_active_teacher(self, unwrapped_model):
        if unwrapped_model.ema_teacher is None:
            return self.teacher_model
        return unwrapped_model.ema_teacher

    def _forward_teacher_patch(self, active_teacher, patch_input_ids: torch.Tensor, patch_attention_mask: torch.Tensor):
        if active_teacher is self.teacher_model:
            teacher_outputs = active_teacher(
                input_ids=patch_input_ids,
                attention_mask=patch_attention_mask,
                output_hidden_states=True,
                output_attentions=False,
            )
            teacher_hidden_states = teacher_outputs.hidden_states
            assert teacher_hidden_states is not None, "Teacher must return hidden states for DSM2 training."
            if len(teacher_hidden_states) > self.teacher_model.config.num_hidden_layers:
                teacher_hidden_states = teacher_hidden_states[1:]
            return teacher_hidden_states

        teacher_outputs = active_teacher(
            input_ids=patch_input_ids,
            attention_mask=patch_attention_mask,
            alpha_ce=0.0,
            alpha_jepa=0.0,
            alpha_contrastive=0.0,
        )
        teacher_hidden_states = teacher_outputs.student_hidden_states
        assert teacher_hidden_states is not None, "EMA teacher must return student_hidden_states for DSM2 training."
        return teacher_hidden_states

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
        optimizer_for_s_max = self.optimizer
        if isinstance(optimizer_for_s_max, AcceleratedOptimizer):
            optimizer_for_s_max = optimizer_for_s_max.optimizer

        assert isinstance(optimizer_for_s_max, MuonAdamWWrapper), (
            f"Expected MuonAdamWWrapper, got {type(optimizer_for_s_max)}."
        )
        optimizer_for_s_max.last_s_max = reduced_s_max

    def _log_train_components(self, ce_loss, jepa_loss, contrastive_loss):
        logs = {
            "train/ce_loss": ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            "train/jepa_loss": jepa_loss.item() if isinstance(jepa_loss, torch.Tensor) else jepa_loss,
            "train/contrastive_loss": contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
        }
        if self.wandb_enabled:
            self.wandb_module.log(logs, step=self.state.global_step)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        batch_size = input_ids.size(0)
        patch_size = self.loss_config.patch_size if self.loss_config.patch_size > 0 else batch_size

        total_ce_loss = 0.0
        total_jepa_loss = 0.0
        total_contrastive_loss = 0.0

        all_teacher_pooled = []
        all_student_pooled = []
        all_s_max_patches = []

        unwrapped_model = extract_model_from_parallel(model)
        active_teacher = self._select_active_teacher(unwrapped_model)
        dsm2_output = None

        for start_idx in range(0, batch_size, patch_size):
            end_idx = min(start_idx + patch_size, batch_size)
            patch_input_ids = input_ids[start_idx:end_idx]
            patch_attention_mask = attention_mask[start_idx:end_idx]
            current_patch_size = end_idx - start_idx

            teacher_hidden_states = self._forward_teacher_patch(active_teacher, patch_input_ids, patch_attention_mask)
            dsm2_patch_output = model(
                input_ids=patch_input_ids,
                attention_mask=patch_attention_mask,
                teacher_hidden_states=teacher_hidden_states,
                alpha_ce=self.loss_config.alpha_ce,
                alpha_jepa=self.loss_config.alpha_jepa,
                alpha_contrastive=0.0,
            )

            weight = current_patch_size / batch_size
            if dsm2_patch_output.ce_loss is not None:
                total_ce_loss += dsm2_patch_output.ce_loss * weight
            if dsm2_patch_output.jepa_loss is not None:
                total_jepa_loss += dsm2_patch_output.jepa_loss * weight

            dsm2_output = dsm2_patch_output

            if self.loss_config.alpha_contrastive > 0.0:
                with torch.no_grad():
                    teacher_pooled = pool_states(teacher_hidden_states)
                student_pooled = pool_states(dsm2_patch_output.student_hidden_states)
                all_teacher_pooled.append(teacher_pooled)
                all_student_pooled.append(student_pooled)

            if dsm2_patch_output.s_max is not None:
                all_s_max_patches.append(dsm2_patch_output.s_max)

        assert dsm2_output is not None, "DSM2 patch output should always be populated after patch loop."

        if (self.loss_config.alpha_contrastive > 0.0) and (len(all_teacher_pooled) > 0):
            stacked_teacher_pooled = torch.cat(all_teacher_pooled, dim=1)
            stacked_student_pooled = torch.cat(all_student_pooled, dim=1)
            total_contrastive_loss = contrastive_loss_from_pooled(
                s_pooled=stacked_student_pooled,
                t_pooled=stacked_teacher_pooled,
            )

        if model.training:
            assert len(all_s_max_patches) > 0, "No s_max values were collected during training; MuonClip cannot run."
            reduced_s_max = self._aggregate_s_max(all_s_max_patches)
            self._set_optimizer_s_max(reduced_s_max)

        loss = (
            (self.loss_config.alpha_ce * total_ce_loss)
            + (self.loss_config.alpha_jepa * total_jepa_loss)
            + (self.loss_config.alpha_contrastive * total_contrastive_loss)
        )

        if (self.state.global_step % self.args.logging_steps == 0) and self.is_world_process_zero():
            self._log_train_components(total_ce_loss, total_jepa_loss, total_contrastive_loss)

        return (loss, dsm2_output) if return_outputs else loss
