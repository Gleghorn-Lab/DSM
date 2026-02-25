#! /usr/bin/env python3
# py -m dsm2.train_dsm2
import entrypoint_setup

import argparse
import os
import time
import torch
from dataclasses import asdict
from huggingface_hub import login
from tqdm.auto import tqdm

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
from dsm2.dsm2_data import build_dsm2_data_bundle, build_dsm2_dataloaders
from dsm2.trainer_utils import infer_rank_world_size_local_rank


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
    parser.add_argument("--alpha_jepa", type=float, default=0.1, help="Weight for JEPA loss")
    parser.add_argument("--alpha_contrastive", type=float, default=10.0, help="Weight for contrastive loss")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (effective size metadata)")
    parser.add_argument("--patch_size", type=int, default=16, help="Micro-batch size processed per forward pass")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps over patch groups")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum optimizer steps")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum tokenized sequence length")
    parser.add_argument("--sliding_window_size", type=int, default=512, help="Sliding-window size for flex attention")
    parser.add_argument("--dilation", type=int, default=16, help="Dilation factor for flex attention")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N optimizer steps")
    parser.add_argument("--eval_every", type=int, default=0, help="Evaluate every N optimizer steps (<=0 uses save_every)")
    parser.add_argument("--logging_steps", type=int, default=100, help="Train metric logging frequency in optimizer steps")
    parser.add_argument("--ema_start_percent", type=float, default=0.4, help="Fraction of steps before EMA teacher starts")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay factor")
    parser.add_argument("--muon_lr", type=float, default=0.001, help="Muon optimizer learning rate")
    parser.add_argument("--muon_tau", type=float, default=100.0, help="QK-Clip tau threshold")
    parser.add_argument("--train_limit", type=int, default=100000, help="Maximum train samples to load (<=0 uses full split)")
    parser.add_argument("--valid_limit", type=int, default=1000, help="Maximum validation samples to load (<=0 uses full split)")
    parser.add_argument("--test_limit", type=int, default=1000, help="Maximum test samples to load (<=0 uses full split)")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed used for dataset shuffling")
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Gradient clipping norm, 0 disables")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="Dataloader prefetch factor when workers > 0")
    parser.add_argument("--distributed_backend", type=str, default="gloo", help="Torch distributed backend")
    parser.add_argument("--no_init_distributed", action="store_true", help="Do not initialize process groups in the trainer")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pinned-memory dataloaders")
    parser.add_argument("--bugfix", action="store_true", help="Use a tiny debug configuration")
    return parser.parse_args()


def build_config_bundle(args) -> DSM2TrainConfigBundle:
    eval_every = args.eval_every
    if eval_every <= 0:
        eval_every = args.save_every

    runtime_config = DSM2RuntimeConfig(
        hf_token=args.hf_token,
        wandb_token=args.wandb_token,
        wandb_project=args.wandb_project,
        save_path=args.save_path,
        bugfix=args.bugfix,
        distributed_backend=args.distributed_backend,
        init_distributed=not args.no_init_distributed,
        pin_memory=not args.no_pin_memory,
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
        patch_accum=args.patch_accum,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        save_every=args.save_every,
        eval_every=eval_every,
        warmup_steps=0,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        muon_lr=args.muon_lr,
        muon_tau=args.muon_tau,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
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


def compile_model(model, model_name: str):
    print(f"Compiling {model_name} model...")
    return torch.compile(model)


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
    from models.modeling_dsm2 import DSM2, DSM2Config

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


def print_run_overview(config: DSM2TrainConfigBundle, data_bundle, data_loaders, world_size: int, is_main_process: bool):
    if not is_main_process:
        return

    effective_samples_per_optimizer_step = (
        config.loss.patch_size * config.optimization.patch_accum * config.optimization.grad_accum * world_size
    )
    eval_every = config.optimization.eval_every
    if eval_every <= 0:
        eval_every = config.optimization.save_every

    print("==== DSM2 Run Overview ====")
    print(f"Save path: {config.runtime.save_path}")
    print(f"Teacher model: {config.model.teacher_model_path}")
    print(f"Dataset: {config.data.data_path}")
    print(
        f"Split sizes loaded | train={len(data_bundle.train_dataset)}, "
        f"valid={len(data_bundle.valid_dataset)}, test={len(data_bundle.test_dataset)}"
    )
    print(
        f"Dataloader patches | train={len(data_loaders.train_loader)}, "
        f"valid={len(data_loaders.valid_loader)}, test={len(data_loaders.test_loader)}"
    )
    print(
        "Optimization | "
        f"max_steps={config.optimization.max_steps}, "
        f"lr={config.optimization.learning_rate:.2e}, "
        f"patch_size={config.loss.patch_size}, "
        f"patch_accum={config.optimization.patch_accum}, "
        f"grad_accum={config.optimization.grad_accum}"
    )
    print(
        f"Effective samples per optimizer step across all ranks: {effective_samples_per_optimizer_step} "
        f"(world_size={world_size})"
    )
    print(
        "Milestones | "
        f"log every {config.optimization.logging_steps} steps, "
        f"evaluate every {eval_every} steps, "
        f"save every {config.optimization.save_every} steps"
    )
    print("Progress bars | training optimizer steps + per-evaluation patch progress")
    print("============================")


def main(config: DSM2TrainConfigBundle, wandb_enabled: bool, wandb_module):
    from dsm2.dsm2_metrics import ComputeDSM2Metrics
    from dsm2.dsm2_teacher import load_teacher_model
    from dsm2.dsm2_trainer import DSM2Trainer
    from dsm2.model_utils import extract_model_from_parallel, patch_accelerate_extract_model_from_parallel

    patch_accelerate_extract_model_from_parallel()

    rank, world_size, local_rank = infer_rank_world_size_local_rank()
    is_main_process = rank == 0

    if torch.cuda.is_available():
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    teacher_model = load_teacher_model(config.model.teacher_model_path, device=device)
    tokenizer = teacher_model.tokenizer

    if is_main_process:
        print("Initializing Student DSM2 from scratch")
    student_model = build_student_model(config, teacher_model.config)

    teacher_model = compile_model(teacher_model, "teacher")

    data_bundle = build_dsm2_data_bundle(config.data, tokenizer)
    use_distributed = config.runtime.init_distributed and (world_size > 1)
    data_loaders = build_dsm2_dataloaders(
        data_bundle=data_bundle,
        patch_size=config.loss.patch_size,
        num_workers=config.optimization.dataloader_num_workers,
        prefetch_factor=config.optimization.dataloader_prefetch_factor,
        pin_memory=config.runtime.pin_memory,
        is_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
    )
    print_run_overview(
        config=config,
        data_bundle=data_bundle,
        data_loaders=data_loaders,
        world_size=world_size,
        is_main_process=is_main_process,
    )

    trainer = DSM2Trainer(
        model=student_model,
        teacher_model=teacher_model,
        loss_config=config.loss,
        optimization_config=config.optimization,
        runtime_config=config.runtime,
        train_loader=data_loaders.train_loader,
        valid_loader=data_loaders.valid_loader,
        test_loader=data_loaders.test_loader,
        valid_dataset=data_bundle.valid_dataset,
        test_dataset=data_bundle.test_dataset,
        compute_metrics=ComputeDSM2Metrics(tokenizer, student_model.config.vocab_size),
        callbacks=[
            EMATeacherCallback(
                total_steps=config.optimization.max_steps,
                ema_start_percent=config.ema.start_percent,
                ema_decay=config.ema.decay,
            )
        ],
        wandb_enabled=wandb_enabled,
        wandb_module=wandb_module,
    )

    phase_progress = tqdm(
        total=4,
        desc="DSM2 run phases",
        unit="phase",
        dynamic_ncols=True,
        leave=True,
        disable=not is_main_process,
    )
    run_start_time = time.perf_counter()

    try:
        phase_progress.set_description_str("DSM2 phase: prepare trainer")
        trainer.prep_for_training()
        phase_progress.update(1)

        if is_main_process:
            print("Trainer initialized")
            print("Initial Evaluation")
        phase_progress.set_description_str("DSM2 phase: initial test evaluation")
        initial_metrics = trainer.evaluate(eval_dataset=data_bundle.test_dataset, prefix="test")
        phase_progress.update(1)
        if is_main_process:
            print("Initial Metrics:\n", initial_metrics)

        if is_main_process:
            print("Training")
        phase_progress.set_description_str("DSM2 phase: training")
        trainer.train()
        phase_progress.update(1)

        if is_main_process:
            print("Final Evaluation")
        phase_progress.set_description_str("DSM2 phase: final test evaluation")
        final_metrics = trainer.evaluate(eval_dataset=data_bundle.test_dataset, prefix="test")
        phase_progress.update(1)
        if is_main_process:
            print("Final Metrics:\n", final_metrics)
            elapsed_minutes = (time.perf_counter() - run_start_time) / 60.0
            print(f"Total runtime: {elapsed_minutes:.2f} minutes")

        if is_main_process:
            if config.runtime.hf_token is None:
                print("Skipping push_to_hub because --hf_token was not provided.")
            else:
                saveable_model = extract_model_from_parallel(trainer.model, keep_torch_compile=False)
                saveable_model.push_to_hub(config.runtime.save_path, private=True)
    finally:
        phase_progress.close()
        trainer.shutdown()
        if wandb_enabled and is_main_process:
            wandb_module.finish()


if __name__ == "__main__":
    args = parse_args()
    args.patch_accum = args.batch_size // args.patch_size
    config_bundle = build_config_bundle(args)

    if config_bundle.runtime.bugfix:
        apply_bugfix_profile(config_bundle)

    if config_bundle.runtime.hf_token is not None:
        login(config_bundle.runtime.hf_token)

    rank, _, _ = infer_rank_world_size_local_rank()
    if rank == 0:
        wandb_enabled_flag, wandb_module_ref = initialize_wandb(config_bundle)
    else:
        wandb_enabled_flag = False
        wandb_module_ref = None

    main(config_bundle, wandb_enabled_flag, wandb_module_ref)
