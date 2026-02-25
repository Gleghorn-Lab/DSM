#! /usr/bin/env python3
# py -m training.train_dsm2
import argparse
import os
from dataclasses import asdict

import entrypoint_setup
import torch

from huggingface_hub import login
from torchinfo import summary
from transformers import TrainingArguments

from models.modeling_dsm2 import DSM2, DSM2Config
from dsm2.dsm2_callbacks import EMATeacherCallback
from dsm2.dsm2_config import (
    DSM2DataConfig,
    DSM2EMAConfig,
    DSM2LossConfig,
    DSM2ModelConfig,
    DSM2OptimizationConfig,
    DSM2RuntimeConfig,
    DSM2TrainConfigBundle,
    apply_bugfix_profile,
)
from dsm2.dsm2_data import build_dsm2_data_bundle
from dsm2.dsm2_metrics import ComputeDSM2Metrics
from dsm2.dsm2_teacher import load_teacher_model
from dsm2.dsm2_trainer import DSM2Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="DSM2 Trainer")
    parser.add_argument("--hf_token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--wandb_token", type=str, default=None, help="Wandb token")
    parser.add_argument("--wandb_project", type=str, default="DSM2", help="Wandb project name")
    parser.add_argument("--teacher_model_path", type=str, default="Synthyra/DPLM-3B", help="Path to initialize the teacher model from")
    parser.add_argument("--student_hidden_size", type=int, default=256, help="Hidden size for the student model")
    parser.add_argument("--student_expansion_ratio", type=float, default=4.0, help="FFN expansion ratio for the student model")
    parser.add_argument("--data_path", type=str, default="Synthyra/uniref50", help="Dataset repository containing train/valid/test splits")
    parser.add_argument("--sequence_column", type=str, default="sequence", help="Name of the sequence column in dataset splits")
    parser.add_argument("--save_path", type=str, default="GleghornLab/DSM2_600", help="Path to save the model and report to wandb")
    parser.add_argument("--alpha_ce", type=float, default=1.0, help="Weight for CE loss")
    parser.add_argument("--alpha_jepa", type=float, default=1.0, help="Weight for JEPA loss")
    parser.add_argument("--alpha_contrastive", type=float, default=1.0, help="Weight for contrastive loss")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=8, help="Max patch size processed per forward pass")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum tokenized sequence length")
    parser.add_argument("--sliding_window_size", type=int, default=512, help="Sliding-window size for flex attention")
    parser.add_argument("--dilation", type=int, default=16, help="Dilation factor for flex attention")
    parser.add_argument("--save_every", type=int, default=1000, help="Save and evaluate every N steps")
    parser.add_argument("--ema_start_percent", type=float, default=0.25, help="Fraction of steps before EMA teacher starts")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay factor")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="Muon optimizer learning rate")
    parser.add_argument("--muon_tau", type=float, default=100.0, help="QK-Clip tau threshold")
    parser.add_argument("--train_limit", type=int, default=100000, help="Maximum train samples to load (<=0 uses full split)")
    parser.add_argument("--valid_limit", type=int, default=1000, help="Maximum validation samples to load (<=0 uses full split)")
    parser.add_argument("--test_limit", type=int, default=1000, help="Maximum test samples to load (<=0 uses full split)")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed used for dataset shuffling")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Gradient clipping norm")
    parser.add_argument("--no_compile_teacher", action="store_true", help="Disable torch.compile for the teacher model")
    parser.add_argument("--no_compile_student", action="store_true", help="Disable torch.compile for the student model")
    parser.add_argument("--bugfix", action="store_true", help="Use a tiny debug configuration")
    return parser.parse_args()


def build_config_bundle(args) -> DSM2TrainConfigBundle:
    runtime_config = DSM2RuntimeConfig(
        hf_token=args.hf_token,
        wandb_token=args.wandb_token,
        wandb_project=args.wandb_project,
        save_path=args.save_path,
        bugfix=args.bugfix,
        compile_teacher=not args.no_compile_teacher,
        compile_student=not args.no_compile_student,
    )
    model_config = DSM2ModelConfig(
        teacher_model_path=args.teacher_model_path,
        student_hidden_size=args.student_hidden_size,
        student_expansion_ratio=args.student_expansion_ratio,
        sliding_window_size=args.sliding_window_size,
        dilation=args.dilation,
    )
    optimization_config = DSM2OptimizationConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        save_every=args.save_every,
        max_grad_norm=args.max_grad_norm,
        muon_lr=args.muon_lr,
        muon_tau=args.muon_tau,
    )
    loss_config = DSM2LossConfig(
        alpha_ce=args.alpha_ce,
        alpha_jepa=args.alpha_jepa,
        alpha_contrastive=args.alpha_contrastive,
        patch_size=args.patch_size,
    )
    data_config = DSM2DataConfig(
        data_path=args.data_path,
        max_length=args.max_length,
        train_limit=args.train_limit,
        valid_limit=args.valid_limit,
        test_limit=args.test_limit,
        shuffle_seed=args.shuffle_seed,
        sequence_column=args.sequence_column,
    )
    ema_config = DSM2EMAConfig(
        start_percent=args.ema_start_percent,
        decay=args.ema_decay,
    )
    return DSM2TrainConfigBundle(
        runtime=runtime_config,
        model=model_config,
        optimization=optimization_config,
        loss=loss_config,
        data=data_config,
        ema=ema_config,
    )


def maybe_compile_model(model, model_name: str, should_compile: bool):
    if not should_compile:
        return model

    print(f"Compiling {model_name} model...")
    try:
        return torch.compile(model)
    except Exception as compile_error:
        print(f"Warning: {model_name} torch.compile() failed: {compile_error}")
        return model


def build_training_arguments(config: DSM2TrainConfigBundle, wandb_enabled: bool):
    return TrainingArguments(
        output_dir=config.runtime.save_path.split("/")[-1],
        per_device_train_batch_size=config.optimization.batch_size,
        per_device_eval_batch_size=config.optimization.batch_size,
        max_steps=config.optimization.max_steps,
        gradient_accumulation_steps=config.optimization.grad_accum,
        logging_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=config.optimization.save_every,
        eval_steps=config.optimization.save_every,
        warmup_steps=config.optimization.save_every,
        learning_rate=config.optimization.learning_rate,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        report_to="wandb" if wandb_enabled else "none",
        save_total_limit=3,
        max_grad_norm=config.optimization.max_grad_norm,
        label_names=["input_ids"],
        remove_unused_columns=False,
    )


def initialize_wandb(config: DSM2TrainConfigBundle):
    wandb_available = os.environ["WANDB_AVAILABLE"] == "true"
    if wandb_available and (config.runtime.wandb_token is not None):
        import wandb

        wandb.login(config.runtime.wandb_token)
        run_name = config.runtime.save_path.split("/")[-1]
        wandb.init(project=config.runtime.wandb_project, name=run_name, config=asdict(config))
        return True, wandb
    return False, None


def build_student_model(config: DSM2TrainConfigBundle, teacher_config):
    student_config = DSM2Config(
        vocab_size=teacher_config.vocab_size,
        hidden_size=config.model.student_hidden_size,
        num_attention_heads=config.model.student_hidden_size // 64,
        num_hidden_layers=teacher_config.num_hidden_layers,
        teacher_hidden_size=teacher_config.hidden_size,
        expansion_ratio=config.model.student_expansion_ratio,
        attn_backend="flex",
        sliding_window_size=config.model.sliding_window_size,
        dilation=config.model.dilation,
    )
    student_model = DSM2(student_config).to(torch.bfloat16)
    student_model.attn_backend = "flex"
    return student_model


def main(config: DSM2TrainConfigBundle, wandb_enabled: bool, wandb_module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model = load_teacher_model(config.model.teacher_model_path, device=device)
    tokenizer = teacher_model.tokenizer
    summary(teacher_model)

    print("Initializing Student DSM2 from scratch")
    student_model = build_student_model(config, teacher_model.config)
    summary(student_model)

    teacher_model = maybe_compile_model(teacher_model, "teacher", config.runtime.compile_teacher)
    student_model = maybe_compile_model(student_model, "student", config.runtime.compile_student)

    data_bundle = build_dsm2_data_bundle(config.data, tokenizer)
    training_args = build_training_arguments(config, wandb_enabled)

    trainer = DSM2Trainer(
        model=student_model,
        args=training_args,
        train_dataset=data_bundle.train_dataset,
        eval_dataset=data_bundle.valid_dataset,
        data_collator=data_bundle.data_collator,
        compute_metrics=ComputeDSM2Metrics(tokenizer),
        callbacks=[
            EMATeacherCallback(
                total_steps=config.optimization.max_steps,
                ema_start_percent=config.ema.start_percent,
                ema_decay=config.ema.decay,
            )
        ],
        teacher_model=teacher_model,
        loss_config=config.loss,
        optimization_config=config.optimization,
        wandb_enabled=wandb_enabled,
        wandb_module=wandb_module,
    )

    print("Trainer initialized")
    print("Initial Evaluation")
    initial_metrics = trainer.evaluate(data_bundle.test_dataset)
    print("Initial Metrics:\n", initial_metrics)

    print("Training")
    trainer.train()

    print("Final Evaluation")
    final_metrics = trainer.evaluate(data_bundle.test_dataset)
    print("Final Metrics:\n", final_metrics)

    if not args.no_compile_student:
        trainer.model._orig_mod.push_to_hub(config.runtime.save_path, private=True)
    else:
        trainer.model.push_to_hub(config.runtime.save_path, private=True)

    if wandb_enabled:
        wandb_module.finish()


if __name__ == "__main__":
    args = parse_args()
    config_bundle = build_config_bundle(args)

    if config_bundle.runtime.bugfix:
        apply_bugfix_profile(config_bundle)

    if config_bundle.runtime.hf_token is not None:
        login(config_bundle.runtime.hf_token)

    wandb_enabled_flag, wandb_module_ref = initialize_wandb(config_bundle)
    main(config_bundle, wandb_enabled_flag, wandb_module_ref)
