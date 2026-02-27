from dataclasses import dataclass


@dataclass
class DSM2RuntimeConfig:
    hf_token: str | None
    wandb_token: str | None
    wandb_project: str
    save_path: str
    bugfix: bool
    distributed_backend: str
    init_distributed: bool
    pin_memory: bool


@dataclass
class DSM2ModelConfig:
    teacher_model_path: str
    pretrained_weights: str | None
    student_hidden_size: int
    student_expansion_ratio: float
    sliding_window_size: int
    dilation: int
    attn_backend: str


@dataclass
class DSM2OptimizationConfig:
    learning_rate: float
    batch_size: int
    patch_accum: int
    grad_accum: int
    max_steps: int
    save_every: int
    eval_every: int
    warmup_steps: int
    logging_steps: int
    max_grad_norm: float
    use_muon: bool
    muon_lr: float
    muon_tau: float
    dataloader_num_workers: int
    dataloader_prefetch_factor: int


@dataclass
class DSM2LossConfig:
    alpha_ce: float
    alpha_jepa: float
    alpha_contrastive: float
    teacher_free_percent: float
    patch_size: int


@dataclass
class DSM2DataConfig:
    data_path: str
    max_length: int
    train_limit: int
    valid_limit: int
    test_limit: int
    shuffle_seed: int
    sequence_column: str


@dataclass
class DSM2EMAConfig:
    start_percent: float
    decay: float


@dataclass
class DSM2TrainConfigBundle:
    runtime: DSM2RuntimeConfig
    model: DSM2ModelConfig
    optimization: DSM2OptimizationConfig
    loss: DSM2LossConfig
    data: DSM2DataConfig
    ema: DSM2EMAConfig


def apply_bugfix_profile(config: DSM2TrainConfigBundle):
    config.optimization.batch_size = 4
    config.loss.patch_size = 2
    config.optimization.patch_accum = 2
    config.optimization.grad_accum = 2
    config.optimization.save_every = 10
    config.optimization.eval_every = 10
    config.optimization.warmup_steps = 10
    config.optimization.logging_steps = 1
    config.optimization.max_steps = 20
    config.optimization.dataloader_num_workers = 0
    config.model.student_hidden_size = 256
    config.model.student_expansion_ratio = 2.0
    config.model.teacher_model_path = "Synthyra/ESM2-8M"
    config.ema.start_percent = 0.5
    config.data.train_limit = 100
    config.data.valid_limit = 10
    config.data.test_limit = 10
