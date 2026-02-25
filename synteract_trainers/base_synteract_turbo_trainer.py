import numpy as np
import os
import queue
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import create_repo, upload_folder
from concurrent.futures import ThreadPoolExecutor
from torch.amp import autocast, GradScaler
from torchinfo import summary
from torch.utils.data import DistributedSampler, RandomSampler, DataLoader, Dataset as TorchDataset
from typing import Dict, List, Tuple, Optional, Any, Iterator, Union, Set
from contextlib import nullcontext
from tqdm.auto import tqdm

from models.synteract.synteract4.model import Synteract4Model, Synteract4BaseModel
from models.synteract.synteract4.config import Synteract4Config
from models.synteract.synteract5.model import Synteract5Model, Synteract5BaseModel
from models.synteract.synteract5.config import Synteract5Config
from losses.softbce import SoftBCEWithLogitsLoss
from losses.contrastive import CosineEmbeddingLoss
from data_classes.torch_datasets import BasicPairedDataset, SingleSpeciesPairedDataset
from data_classes.collators import BasicPairedCollator, EmbeddingLookup, SQLEmbeddingLookup, TokenizingCollator
from data_classes.samplers import OneSpeciesBatchSampler, DistributedOneSpeciesBatchSampler
from utils.embedding import embed_dataset_with_cache, embed_dataset_distributed
from utils.metrics import calculate_classification_metrics, calculate_topk_metrics
from utils.exceptions import TrainingNaNError
from utils.violations import build_interaction_partners, find_violation_indices
from .base_trainer import BaseTrainer
from .loss_utils import mean_over_non_ignored_entries


if os.environ['WANDB_AVAILABLE'] == 'true':
    import wandb


class EmbedMode:
    """Constants for embedding modes."""
    PRECOMPUTED = 'precomputed'
    ON_THE_FLY = 'on_the_fly'


class AsyncEmbeddingResolver:
    """
    Asynchronous embedding resolver that prefetches embeddings in a background thread.
    
    This overlaps embedding lookup with GPU computation, hiding the latency of
    dictionary lookups and tensor operations for the next batch while the current
    batch is being processed on GPU.
    
    Usage:
        resolver = AsyncEmbeddingResolver(embedding_lookup)
        
        # Start prefetching first batch
        resolver.submit(first_batch)
        
        for batch in dataloader:
            # Get the prefetched result (blocks if not ready)
            current_batch = resolver.get()
            
            # Start prefetching next batch while processing current
            resolver.submit(batch)
            
            # Process current_batch on GPU...
        
        # Don't forget the last batch
        last_batch = resolver.get()
        
        resolver.shutdown()
    """
    
    def __init__(self, embedding_lookup, max_queue_size: int = 2, max_workers: int = 1):
        """
        Args:
            embedding_lookup: EmbeddingLookup instance with add_embeddings method
            max_queue_size: Maximum number of prefetched batches to queue
            max_workers: Number of worker threads for embedding resolution
        """
        assert max_queue_size > 0, "max_queue_size must be >= 1"
        assert max_workers > 0, "max_workers must be >= 1"
        self.embedding_lookup = embedding_lookup
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.max_queue_size = max_queue_size
        self.inflight = 0
        self._shutdown = False
    
    def submit(self, batch: Dict[str, Any]) -> None:
        """Submit a batch for async embedding resolution."""
        if self._shutdown:
            return
        assert self.inflight < self.max_queue_size, "Async embedding resolver queue is full"
        future = self.executor.submit(self.embedding_lookup.add_embeddings, batch)
        self.result_queue.put(future)
        self.inflight += 1
    
    def get(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get the next resolved batch. Blocks until ready.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Batch with embeddings added
        """
        future = self.result_queue.get(timeout=timeout)
        self.inflight -= 1
        return future.result()

    def pending_count(self) -> int:
        """Return number of in-flight batches."""
        return self.inflight
    
    def shutdown(self) -> None:
        """Shutdown the executor and clean up."""
        self._shutdown = True
        self.executor.shutdown(wait=True)


class BaseSynteractTurboTrainer(BaseTrainer):
    """
    Base trainer for protein-protein interaction models.
    
    Supports two embedding modes:
    1. PRECOMPUTED: Load embeddings once, pass embedding tensors to model
       - Model takes: a, b, a_mask, b_mask
       - Uses: SharedMemoryCollator or EmbeddingLookup
       
    2. ON_THE_FLY: Model embeds sequences during forward pass
       - Model takes: input_ids_a, input_ids_b, attention_mask_a, attention_mask_b
       - Uses: TokenizingCollator
    
    Child classes must set these attributes in __init__ BEFORE calling prep_for_training():
        - self.embed_mode: EmbedMode.PRECOMPUTED or EmbedMode.ON_THE_FLY
        - self.train_dataframes: Dict[str, pd.DataFrame]
        - self.valid_dataframes: Dict[str, pd.DataFrame] 
        - self.test_dataframes: Dict[str, pd.DataFrame]
        - self.seq_dict: Dict[str, str] - protein_id -> sequence (single dict)
        - self.seq_dict_a: Dict[str, str] - A-track id -> sequence (optional)
        - self.seq_dict_b: Dict[str, str] - B-track id -> sequence (optional)
        - self.interaction_set: Set[str] - known interactions for violation detection
        - self.skip_violations: bool - whether to skip violation detection
        - self.batch_size: int
        - self.patch_size: int
        - self.patch_accum: int
    """
    
    def __init__(self, args):
        """
        Initialize the PPI trainer.
        
        Args:
            args: Training arguments with:
                - a_encoder_path: Path to A-track encoder (precomputed mode)
                - b_encoder_path: Path to B-track encoder (precomputed mode)
                - max_length: Maximum sequence length
                - patch_size: Size of each patch (sub-batch)
                - grad_accum: Gradient accumulation steps
                - Other standard training args from BaseTrainer
        """
        super().__init__(args)        
        # These will be set by child classes
        self.a_embed_dict = None
        self.b_embed_dict = None
        self.a_embed_dim = None
        self.b_embed_dim = None
        # backward compatibility
        self.embed_dim = None 
        
        # Sequence dictionaries (single or dual-track)
        self.seq_dict = None
        self.seq_dict_a = None
        self.seq_dict_b = None
        self.is_single_organism = False
        self.single_organism_id = None
        self.interaction_set = None
        self.interaction_partners = {}
        self.skip_violations = False
        
        self.tokenizer = None
        self.tokenizer_a = None
        self.tokenizer_b = None
        self.embedding_lookup = None
        
        # Dual encoder support
        self.same_encoder = args.same_encoder
        self.swap_allowed = self.same_encoder  # Can only swap A/B when using same encoder
        
        # NaN detection - when enabled, raises TrainingNaNError on NaN/Inf loss
        self.halt_on_nan = getattr(args, 'halt_on_nan', False)
        
        # Batch sizing
        self.patch_size = args.patch_size
        self.patch_accum = args.patch_accum
        self.batch_size = args.batch_size

        # Async embedding resolution settings (precomputed mode)
        self.async_resolve_workers = args.async_resolve_workers
        self.async_resolve_queue = args.async_resolve_queue
        self.pin_embeddings = args.pin_embeddings
        assert self.async_resolve_workers > 0, "async_resolve_workers must be >= 1"
        assert self.async_resolve_queue > 0, "async_resolve_queue must be >= 1"
        self._resolve_executor = None
        if self.async_resolve_workers > 1 and args.memory_mode != 'sql':
            self._resolve_executor = ThreadPoolExecutor(max_workers=self.async_resolve_workers)
        
        # Mixed precision (AMP) setup - standardized on bfloat16
        self.use_amp = args.use_amp
        self.amp_dtype = torch.bfloat16
        if self.use_amp:
            self.scaler = GradScaler('cuda')
            self._print(f"Using Automatic Mixed Precision (AMP) with {self.amp_dtype}")
        else:
            self.scaler = None

        # === SET EMBEDDING MODE ===
        # SQL mode requires preembed mode (streaming precomputed embeddings from SQLite)
        if args.memory_mode == 'sql' and not args.preembed:
            self._print("WARNING: SQL mode requires preembed=True. Enabling preembed automatically.")
            args.preembed = True
        
        if args.preembed:
            # Precomputed mode requires last_hidden_state processing for both encoders
            assert args.a_encoder_layer_processing == 'last_hidden_state', (
                f"EmbedMode.PRECOMPUTED requires a_encoder_layer_processing='last_hidden_state', "
                f"but got '{args.a_encoder_layer_processing}'. "
                f"Linear/conv processing requires on-the-fly embedding to learn layer weights."
            )
            assert args.b_encoder_layer_processing == 'last_hidden_state', (
                f"EmbedMode.PRECOMPUTED requires b_encoder_layer_processing='last_hidden_state', "
                f"but got '{args.b_encoder_layer_processing}'. "
                f"Linear/conv processing requires on-the-fly embedding to learn layer weights."
            )
            self.embed_mode = EmbedMode.PRECOMPUTED
            if args.memory_mode == 'sql':
                self._print("Using PRECOMPUTED embedding mode with SQL streaming (memory-efficient)")
            else:
                self._print("Using PRECOMPUTED embedding mode")
        else:
            self.embed_mode = EmbedMode.ON_THE_FLY
            self._print(f"Using ON_THE_FLY embedding mode")
            self._print(f"  A-track layer processing: {args.a_encoder_layer_processing}")
            self._print(f"  B-track layer processing: {args.b_encoder_layer_processing}")

    def prep_for_training(self, memory_mode: str = 'basic'):
        """
        Prepare for training.
        
        Args:
            memory_mode: For precomputed mode, how to handle embeddings:
                - 'basic': Main process does embedding lookup
                - 'sql': Stream embeddings from SQLite database (for large datasets)
        """
        # Get embeddings if using precomputed mode
        if self.embed_mode == EmbedMode.PRECOMPUTED:
            use_sql = (memory_mode == 'sql')
            self.get_embeddings(use_sql=use_sql)
        
        # Build interaction partner lookup for fast violation detection
        self._build_interaction_partners()
        
        # Ensure GPU is clear before model initialization
        # (embedding generation may have used GPU temporarily)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get model (may provide tokenizer(s) for on-the-fly mode)
        self.model = self.get_model()
        
        # For on-the-fly mode, get tokenizer(s) from model
        if self.embed_mode == EmbedMode.ON_THE_FLY:
            if self.same_encoder:
                # Single encoder mode - use shared tokenizer
                if self.model.tokenizer is not None:
                    self.tokenizer = self.model.tokenizer
                else:
                    self.tokenizer = self.model.tokenizer_a
                self.tokenizer_a = self.tokenizer
                self.tokenizer_b = self.tokenizer
            else:
                # Dual encoder mode - use separate tokenizers
                self.tokenizer_a = self.model.tokenizer_a
                self.tokenizer_b = self.model.tokenizer_b
                self.tokenizer = None  # No single tokenizer in dual mode
        
        # Move to device and compile
        self.model = self.model.to(self.device)
        self.model = self._compile_model(self.model)
        
        # Wrap with DDP if distributed
        self.model = self._wrap_model_ddp(self.model)
        
        # Create datasets and loaders
        self.train_dataset, self.valid_dataset, self.test_dataset = self.get_datasets()
        self.train_loader, self.valid_loader, self.test_loader = self.get_data_loaders(memory_mode)
        
        self.optimizer, self.scheduler = self.get_optimizers()
        self.loss_fct = self.get_loss_fct()
        
        if self.is_main_process:
            self.setup_logging()

    def _embed_dataset(
        self,
        plm_path: str,
        ids: List[str],
        seq_dict: Dict[str, str],
        use_sql: bool,
        db_path_suffix: str,
        embed_path_suffix: str,
        max_len: int,
        encoder_precision: str,
        layer_processing: str,
        kernel_size: int,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[str], int]:
        """
        Helper to embed ids with an Encoder.
        
        Uses distributed embedding when multiple GPUs are available.
        
        Args:
            plm_path: Path to the encoder model
            ids: List of ids to embed
            seq_dict: Dict of id -> sequence
            use_sql: Whether to save to SQLite
            db_path_suffix: Suffix for SQL database path
            embed_path_suffix: Suffix for embedding file path
            max_len: Max sequence length for embedding
            encoder_precision: Precision string for encoder weights
            layer_processing: Layer processing strategy (must be last_hidden_state for precomputed mode)
            kernel_size: Kernel size for conv layer processing (unused when last_hidden_state)
            
        Returns:
            Tuple of (embed_dict or None, sql_db_path or None, embed_dim)
        """
        if self.is_distributed:
            return embed_dataset_distributed(
                plm_path=plm_path,
                ids=ids,
                seq_dict=seq_dict,
                use_sql=use_sql,
                db_path_suffix=db_path_suffix,
                embed_path_suffix=embed_path_suffix,
                max_len=max_len,
                encoder_precision=encoder_precision,
                layer_processing=layer_processing,
                kernel_size=kernel_size,
                device=self.device,
                patch_size=self.args.embedding_batch_size,
                bugfix=self.args.bugfix,
                use_compile=self.args.use_compile,
                print_fn=self._print,
                rank=self.rank,
                world_size=self.world_size,
            )
        else:
            return embed_dataset_with_cache(
                plm_path=plm_path,
                ids=ids,
                seq_dict=seq_dict,
                use_sql=use_sql,
                db_path_suffix=db_path_suffix,
                embed_path_suffix=embed_path_suffix,
                max_len=max_len,
                encoder_precision=encoder_precision,
                layer_processing=layer_processing,
                kernel_size=kernel_size,
                device=self.device,
                patch_size=self.args.embedding_batch_size,
                bugfix=self.args.bugfix,
                use_compile=self.args.use_compile,
                print_fn=self._print,
            )

    def get_embeddings(self, use_sql: bool = False):
        """
        Compute and cache embeddings for precomputed mode.
        Override in child classes if using a different PLM or embedding strategy.
        
        Supports both single encoder (same PLM for A and B) and dual encoder 
        (different PLMs for A and B tracks) modes.
        
        When distributed training is enabled, all GPUs participate in embedding
        for faster parallel processing.
        
        Args:
            use_sql: If True, save embeddings to SQLite database instead of memory.
                    This enables streaming mode for large datasets.
        """
        # Synchronize before starting
        self._synchronize()
        
        # Prepare sequence lists - all processes need these for distributed embedding
        assert self.seq_dict is not None or self.seq_dict_a is not None, (
            "Missing seq_dict/seq_dict_a for embedding."
        )
        seq_dict_a = self.seq_dict_a if self.seq_dict_a is not None else self.seq_dict
        seq_dict_b = self.seq_dict_b if self.seq_dict_b is not None else seq_dict_a
        
        if self.same_encoder:
            merged_seq_dict = dict(seq_dict_a)
            for seq_id, seq in seq_dict_b.items():
                if seq_id in merged_seq_dict:
                    assert merged_seq_dict[seq_id] == seq, f"Sequence mismatch for id '{seq_id}'"
                else:
                    merged_seq_dict[seq_id] = seq

            all_ids = list(set(merged_seq_dict.keys()))
            self._print(f"Embedding {len(all_ids)} unique ids (shared encoder)")
            self._print(f"Using same encoder for A and B tracks: {self.args.a_encoder_path}")
            
            # All processes participate in distributed embedding
            embed_dict, sql_db_path, embed_dim = self._embed_dataset(
                plm_path=self.args.a_encoder_path,
                ids=all_ids,
                seq_dict=merged_seq_dict,
                use_sql=use_sql,
                db_path_suffix='shared',
                embed_path_suffix='shared',
                max_len=max(self.args.max_length_a, self.args.max_length_b),
                encoder_precision=self.args.a_encoder_precision,
                layer_processing=self.args.a_encoder_layer_processing,
                kernel_size=self.args.kernel_size,
            )
            
            self.a_embed_dict = embed_dict
            self.b_embed_dict = embed_dict  # Same dict for both
            self.a_sql_db_path = sql_db_path
            self.b_sql_db_path = sql_db_path  # Same DB for both
            self.a_embed_dim = embed_dim
            self.b_embed_dim = embed_dim
            self.embed_dim = embed_dim  # backwards compat
        else:
            assert self.seq_dict is not None or (self.seq_dict_a is not None and self.seq_dict_b is not None), (
                "Dual-encoder embedding requires seq_dict or both seq_dict_a/seq_dict_b."
            )
            seq_dict_a = self.seq_dict_a if self.seq_dict_a is not None else self.seq_dict
            seq_dict_b = self.seq_dict_b if self.seq_dict_b is not None else self.seq_dict
            a_ids = list(set(seq_dict_a.keys()))
            b_ids = list(set(seq_dict_b.keys()))
            self._print(f"Embedding {len(a_ids)} unique A-track ids")
            self._print(f"Embedding {len(b_ids)} unique B-track ids")
            
            self._print(f"Using different encoders:")
            self._print(f"  A-track encoder: {self.args.a_encoder_path}")
            self._print(f"  B-track encoder: {self.args.b_encoder_path}")
            
            # Embed with A encoder - all processes participate
            a_embed_dict, a_sql_db_path, a_embed_dim = self._embed_dataset(
                plm_path=self.args.a_encoder_path,
                ids=a_ids,
                seq_dict=seq_dict_a,
                use_sql=use_sql,
                db_path_suffix='track_a',
                embed_path_suffix='track_a',
                max_len=self.args.max_length_a,
                encoder_precision=self.args.a_encoder_precision,
                layer_processing=self.args.a_encoder_layer_processing,
                kernel_size=self.args.kernel_size,
            )
            
            # Embed with B encoder - all processes participate
            b_embed_dict, b_sql_db_path, b_embed_dim = self._embed_dataset(
                plm_path=self.args.b_encoder_path,
                ids=b_ids,
                seq_dict=seq_dict_b,
                use_sql=use_sql,
                db_path_suffix='track_b',
                embed_path_suffix='track_b',
                max_len=self.args.max_length_b,
                encoder_precision=self.args.b_encoder_precision,
                layer_processing=self.args.b_encoder_layer_processing,
                kernel_size=self.args.kernel_size,
            )
            
            self.a_embed_dict = a_embed_dict
            self.b_embed_dict = b_embed_dict
            self.a_sql_db_path = a_sql_db_path
            self.b_sql_db_path = b_sql_db_path
            self.a_embed_dim = a_embed_dim
            self.b_embed_dim = b_embed_dim
            self.embed_dim = a_embed_dim  # Default to A
        
        self._synchronize()
        
    def get_datasets(self) -> Tuple[TorchDataset, TorchDataset, TorchDataset]:
        """Create basic datasets that work with both embedding modes."""
        assert self.seq_dict is not None or self.seq_dict_a is not None, (
            "Missing seq_dict/seq_dict_a for dataset construction."
        )
        seq_dict_a = self.seq_dict_a if self.seq_dict_a is not None else self.seq_dict
        seq_dict_b = self.seq_dict_b if self.seq_dict_b is not None else seq_dict_a
        dataset_cls = SingleSpeciesPairedDataset if self.is_single_organism else BasicPairedDataset
        train_dataset = dataset_cls(
            self.train_dataframes,
            seq_dict=None,
            seq_dict_a=seq_dict_a,
            seq_dict_b=seq_dict_b,
            eval_mode=False,
            swap_allowed=self.swap_allowed,
        )
        valid_dataset = dataset_cls(
            self.valid_dataframes,
            seq_dict=None,
            seq_dict_a=seq_dict_a,
            seq_dict_b=seq_dict_b,
            eval_mode=True,
            swap_allowed=self.swap_allowed,
        )
        test_dataset = dataset_cls(
            self.test_dataframes,
            seq_dict=None,
            seq_dict_a=seq_dict_a,
            seq_dict_b=seq_dict_b,
            eval_mode=True,
            swap_allowed=self.swap_allowed,
        )
        return train_dataset, valid_dataset, test_dataset

    def get_data_loaders(self, memory_mode: str = 'basic') -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders based on embedding mode.
        
        Training uses single-species batch sampling to ensure each batch contains
        only proteins from one species (org_a == org_b). This is critical for 
        contrastive learning with proper negative sampling.
        
        Validation and test use random sampling across all species for efficiency.
        
        Args:
            memory_mode: For precomputed mode:
                - 'basic': BasicPairedCollator (workers do embedding lookup)
                - 'sql': BasicPairedCollator + SQLEmbeddingLookup (streaming from SQLite)
        """
        # Create appropriate collator based on mode
        # We need two collators:
        # 1. train_collator: checks species homogeneity (vital for training logic)
        # 2. eval_collator: does NOT check species (allows mixed batches during valid/test)
        if self.embed_mode == EmbedMode.ON_THE_FLY:
            # Use tokenizer_b only if different encoders
            tokenizer_b = self.tokenizer_b if not self.same_encoder else None
            
            train_collator = TokenizingCollator(
                tokenizer=self.tokenizer_a,
                max_length=self.args.max_length_a,
                check_species=True,
                tokenizer_b=tokenizer_b,
                max_length_b=self.args.max_length_b,
            )
            eval_collator = TokenizingCollator(
                tokenizer=self.tokenizer_a,
                max_length=self.args.max_length_a,
                check_species=False,
                tokenizer_b=tokenizer_b,
                max_length_b=self.args.max_length_b,
            )
            self.embedding_lookup = None
            
        elif self.embed_mode == EmbedMode.PRECOMPUTED:
            if memory_mode == 'sql':
                # SQL mode - streaming embeddings from SQLite database
                if self.same_encoder:
                    self._print(f"Using SQL embedding mode with shared database: {self.a_sql_db_path}")
                else:
                    self._print(f"Using SQL embedding mode with dual databases:")
                    self._print(f"  A-track: {self.a_sql_db_path}")
                    self._print(f"  B-track: {self.b_sql_db_path}")
                    
                train_collator = BasicPairedCollator(check_species=True)
                eval_collator = BasicPairedCollator(check_species=False)
                
                self.embedding_lookup = SQLEmbeddingLookup(
                    db_path=self.a_sql_db_path,
                    embed_dim=self.a_embed_dim,
                    max_length=self.args.max_length_a,
                    cache_size=10000,
                    b_db_path=self.b_sql_db_path if not self.same_encoder else None,
                    b_embed_dim=self.b_embed_dim,
                    max_length_b=self.args.max_length_b,
                )
            else:
                # Basic mode - embedding lookup in main process
                train_collator = BasicPairedCollator(check_species=True)
                eval_collator = BasicPairedCollator(check_species=False)
                
                self.embedding_lookup = EmbeddingLookup(
                    embedding_dict=self.a_embed_dict,
                    embed_dim=self.a_embed_dim,
                    max_length=self.args.max_length_a,
                    b_embedding_dict=self.b_embed_dict if not self.same_encoder else None,
                    b_embed_dim=self.b_embed_dim,
                    max_length_b=self.args.max_length_b,
                )
        else:
            raise ValueError(f"Unknown embed_mode: {self.embed_mode}")
        
        # === TRAINING: Single-species batch sampling ===
        # For multi-organism data, enforce single-species batches.
        # For single-organism data, use standard sampling to avoid overhead.
        if self.is_single_organism:
            if self.is_distributed:
                train_sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    drop_last=True,
                )
                self.train_sampler = train_sampler
            else:
                train_sampler = RandomSampler(self.train_dataset)
        else:
            if self.is_distributed:
                train_sampler = DistributedOneSpeciesBatchSampler(
                    species_to_indices=self.train_dataset.species_to_indices,
                    batch_size=self.patch_size,
                    rank=self.rank,
                    world_size=self.world_size,
                    drop_last=True,
                    group_consecutive=self.patch_accum,
                    shuffle=True,
                )
                self.train_sampler = train_sampler
            else:
                train_sampler = OneSpeciesBatchSampler(
                    species_to_indices=self.train_dataset.species_to_indices,
                    batch_size=self.patch_size,
                    drop_last=True,
                    group_consecutive=self.patch_accum,
                )
        
        # === VALIDATION/TEST: Random sampling across all species ===
        # No need to enforce single-species batches during evaluation
        if self.is_distributed:
            valid_sampler = DistributedSampler(
                self.valid_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )
            test_sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )
            self.valid_sampler = valid_sampler
            self.test_sampler = test_sampler
        else:
            valid_sampler = RandomSampler(self.valid_dataset)
            test_sampler = RandomSampler(self.test_dataset)
        
        # Create DataLoaders - optimized for GPU training
        # Note: We must pass collate_fn specifically for each loader now
        loader_kwargs = dict(
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=torch.cuda.is_available(),  # Enable for all GPU training
            persistent_workers=(self.num_workers > 0),  # Avoid worker restart overhead
        )
        
        # Train uses batch_sampler (multi-org) or sampler+batch_size (single-org)
        if self.is_single_organism:
            train_loader = DataLoader(
                self.train_dataset,
                sampler=train_sampler,
                batch_size=self.patch_size,
                collate_fn=train_collator,
                drop_last=True,
                **loader_kwargs
            )
        else:
            train_loader = DataLoader(
                self.train_dataset, 
                batch_sampler=train_sampler, 
                collate_fn=train_collator,
                **loader_kwargs
            )
        
        # Valid/test use sampler with batch_size (random sampling) and UNCHECKED collator
        valid_loader = DataLoader(
            self.valid_dataset, 
            sampler=valid_sampler, 
            batch_size=self.patch_size,
            collate_fn=eval_collator,
            drop_last=False,
            **loader_kwargs
        )
        test_loader = DataLoader(
            self.test_dataset, 
            sampler=test_sampler, 
            batch_size=self.patch_size,
            collate_fn=eval_collator,
            drop_last=False,
            **loader_kwargs
        )
        
        return train_loader, valid_loader, test_loader

    def get_loss_fct(self):
        """Initialize loss function with class imbalance weighting."""
        pos_weight = (self.batch_size * self.batch_size - self.batch_size) / self.batch_size
        pos_weight = int(pos_weight * 0.85)  # Accounting for violations
        pos_weight = torch.tensor(pos_weight, device=self.device)
        self.loss_type = self.args.loss_type
        if self.loss_type == 'cosine':
            self.loss_fct = CosineEmbeddingLoss(ignore_index=-100.0, pos_weight=pos_weight, reduction='none')
        else:
            self.loss_fct = SoftBCEWithLogitsLoss(ignore_index=-100.0, pos_weight=pos_weight, reduction='none')
        return self.loss_fct

    # ==================== Metrics ====================
    def _metrics_helper(self, logits: np.ndarray, labels: np.ndarray, prefix: Optional[str] = '') -> Dict[str, float]:
        """Compute classification metrics using consolidated utils.metrics."""
        logits = np.asarray(logits).flatten()
        labels = np.asarray(labels).flatten()

        assert len(logits) == len(labels), 'Logits and labels must have the same length'
        assert len(logits) > 0 and len(labels) > 0, 'Logits and labels must not be empty'

        # Guard against NaN/Inf
        finite_mask = np.isfinite(logits) & np.isfinite(labels)
        if not finite_mask.all():
            n_bad = int((~finite_mask).sum())
            self._print(f"[metrics] Dropping {n_bad}/{len(finite_mask)} non-finite (NaN/Inf) logit/label pairs.")
            logits = logits[finite_mask]
            labels = labels[finite_mask]

        if len(logits) == 0:
            return {
                f'{prefix}_roc_auc': float('nan'),
                f'{prefix}_pr_auc': float('nan'),
                f'{prefix}_f1': float('nan'),
                f'{prefix}_precision': float('nan'),
                f'{prefix}_recall': float('nan'),
                f'{prefix}_accuracy': float('nan'),
                f'{prefix}_mcc': float('nan'),
                f'{prefix}_threshold': float('nan'),
            }

        # Use consolidated function with optimal threshold finding
        metrics = calculate_classification_metrics(
            labels, logits, threshold=None, find_optimal_threshold=True
        )
        
        # Rename keys with prefix
        renamed_metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
        return renamed_metrics

    def calculate_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor, 
    ) -> Dict[str, float]:
        """Calculate classification metrics with balanced sampling."""
        def _nan_balanced_metrics() -> Dict[str, float]:
            return {
                'balanced_roc_auc': float('nan'),
                'balanced_pr_auc': float('nan'),
                'balanced_f1': float('nan'),
                'balanced_precision': float('nan'),
                'balanced_recall': float('nan'),
                'balanced_accuracy': float('nan'),
                'balanced_mcc': float('nan'),
                'balanced_threshold': float('nan'),
            }

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().float().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().float().cpu().numpy()

        logits = logits.flatten()
        labels = labels.flatten()

        # Filter out violation labels (-100.0)
        valid_label_mask = (labels == 0) | (labels == 1)
        assert valid_label_mask.any(), 'No valid labels found (all are violations)'

        # Also drop any non-finite values before sklearn metrics.
        finite_mask = np.isfinite(logits) & np.isfinite(labels)
        combined_mask = valid_label_mask & finite_mask

        if not combined_mask.any():
            self._print("[metrics] No finite, non-violation label/logit pairs remain; returning NaN metrics.")
            metrics = _nan_balanced_metrics()
            metrics['ratio'] = float('nan')
            metrics['avg_pos'] = float('nan')
            metrics['avg_neg'] = float('nan')
            metrics['n_total'] = int(len(labels))
            metrics['n_valid_labels'] = int(valid_label_mask.sum())
            metrics['n_finite_valid'] = int(combined_mask.sum())
            return metrics

        if not finite_mask.all():
            n_bad = int((~finite_mask).sum())
            self._print(f"[metrics] Dropping {n_bad}/{len(finite_mask)} non-finite (NaN/Inf) entries before balancing.")

        valid_logits = logits[combined_mask]
        valid_labels = labels[combined_mask]

        pos_values_full = valid_logits[valid_labels == 1]
        neg_values_full = valid_logits[valid_labels == 0]
        pos_avg_full = float(pos_values_full.mean()) if pos_values_full.size > 0 else float('nan')
        neg_avg_full = float(neg_values_full.mean()) if neg_values_full.size > 0 else float('nan')
        ratio_full = (pos_avg_full / neg_avg_full) if (neg_avg_full != 0 and np.isfinite(neg_avg_full) and np.isfinite(pos_avg_full)) else float('nan')

        # Balance: sample negatives to match positives
        pos_indices = np.where(valid_labels == 1)[0]
        neg_indices = np.where(valid_labels == 0)[0]
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Degenerate case: metrics requiring both classes can't be computed.
        if n_pos == 0 or n_neg == 0:
            self._print(f"[metrics] Degenerate labels after filtering: n_pos={n_pos}, n_neg={n_neg}; returning NaN metrics.")
            metrics = _nan_balanced_metrics()
            metrics['ratio'] = float('nan')
            metrics['avg_pos'] = float('nan')
            metrics['avg_neg'] = float('nan')
            metrics['n_total'] = int(len(labels))
            metrics['n_valid_labels'] = int(valid_label_mask.sum())
            metrics['n_finite_valid'] = int(combined_mask.sum())
            metrics['n_pos'] = int(n_pos)
            metrics['n_neg'] = int(n_neg)
            return metrics

        if n_neg > n_pos:
            sampled_neg_indices = np.random.choice(neg_indices, size=n_pos, replace=False)
        else:
            sampled_neg_indices = neg_indices

        balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
        np.random.shuffle(balanced_indices)

        logits_bal = valid_logits[balanced_indices]
        labels_bal = valid_labels[balanced_indices]

        metrics = self._metrics_helper(logits_bal, labels_bal, prefix='balanced')
        
        # Add top-k metrics
        topk_metrics_result = calculate_topk_metrics(labels_bal, logits_bal)
        for k, v in topk_metrics_result.items():
            metrics[k] = v
        
        metrics['ratio'] = ratio_full
        metrics['avg_pos'] = pos_avg_full
        metrics['avg_neg'] = neg_avg_full
        metrics['n_total'] = int(len(labels))
        metrics['n_valid_labels'] = int(valid_label_mask.sum())
        metrics['n_finite_valid'] = int(combined_mask.sum())
        metrics['n_pos'] = int(n_pos)
        metrics['n_neg'] = int(n_neg)
        return metrics
    
    # ==================== Violation Detection ====================
    def _build_interaction_partners(self) -> None:
        """
        Pre-build a mapping from each protein ID to its known interaction partners.
        
        This enables O(batch) violation detection instead of O(batch²) by:
        1. For each protein in ids_a, look up its partners in O(1)
        2. Check which ids_b are in that partner set using set intersection
        """
        valid_ids: Optional[Set[str]] = None
        if self.seq_dict is not None or self.seq_dict_a is not None:
            seq_dict_a = self.seq_dict_a if self.seq_dict_a is not None else self.seq_dict
            seq_dict_b = self.seq_dict_b if self.seq_dict_b is not None else seq_dict_a
            valid_ids = set(seq_dict_a.keys())
            valid_ids.update(seq_dict_b.keys())

        self.interaction_partners = build_interaction_partners(
            interaction_set=self.interaction_set,
            skip_violations=self.skip_violations,
            valid_ids=valid_ids,
            print_fn=self._print,
        )

    def _find_violations_fast(
        self,
        ids_a: List[Union[str, int]],
        ids_b: List[Union[str, int]],
    ) -> Optional[torch.Tensor]:
        """
        Find violation indices using vectorized set operations.
        
        Returns:
            Tensor of flat indices where violations occur, or None if no violations.
            These are indices into the flattened (batch, batch) logits matrix.
        """
        return find_violation_indices(
            ids_a=ids_a,
            ids_b=ids_b,
            interaction_partners=self.interaction_partners,
            skip_violations=self.skip_violations,
        )

    def _find_violations(self, ids_a: List[Union[str, int]], ids_b: List[Union[str, int]], targets: torch.Tensor) -> torch.Tensor:
        """Mark known positive pairs in off-diagonal as violations (-100)."""
        violation_indices = find_violation_indices(
            ids_a=ids_a,
            ids_b=ids_b,
            interaction_partners=self.interaction_partners,
            skip_violations=self.skip_violations,
        )
        if violation_indices is not None:
            targets[violation_indices.to(targets.device)] = -100.0
        
        return targets
    
    def _get_patch_species(self, patch: Dict[str, Any]) -> int:
        """
        Extract the species ID from a patch.
        
        Returns:
            The species ID for this patch
        """
        org_a = patch['org_a']
        unique_org_a = org_a.unique()
        assert unique_org_a.numel() == 1, \
            f"Patch has multiple species in org_a: {unique_org_a.tolist()}"
        return unique_org_a.item()

    def _validate_patches_single_species(self, patches: List[Dict[str, Any]]) -> None:
        """Validate that all patches in a group are from a single species."""
        if not patches:
            return
        
        group_species = None
        for patch_idx, patch in enumerate(patches):
            org_a = patch['org_a']
            org_b = patch['org_b']
            
            unique_org_a = org_a.unique()
            assert unique_org_a.numel() == 1, \
                f"Patch {patch_idx} has multiple species in org_a: {unique_org_a.tolist()}"
            
            unique_org_b = org_b.unique()
            assert unique_org_b.numel() == 1, \
                f"Patch {patch_idx} has multiple species in org_b: {unique_org_b.tolist()}"
            
            patch_species_a = unique_org_a.item()
            patch_species_b = unique_org_b.item()
            assert patch_species_a == patch_species_b, \
                f"Patch {patch_idx} has mismatched species: org_a={patch_species_a}, org_b={patch_species_b}"
            
            patch_species = patch_species_a
            
            if group_species is None:
                group_species = patch_species
            else:
                assert patch_species == group_species, \
                    f"Patch {patch_idx} has species {patch_species}, but group expected {group_species}"
    
    def _calculate_effective_batches(self, dataset: TorchDataset, drop_last: bool) -> int:
        """
        Calculate the true number of effective batches accounting for species boundaries.
        
        Since _accumulate_single_species_patches stops at species boundaries,
        each species contributes ceil(num_patches / patch_accum) effective batches.
        
        Args:
            dataset: The dataset with species_to_indices attribute
            drop_last: Whether incomplete patches are dropped
            
        Returns:
            Total number of effective batches across all species
        """
        total_effective = 0
        for species, indices in dataset.species_to_indices.items():
            num_samples = len(indices)
            
            # Calculate number of patches for this species
            if drop_last:
                num_patches = num_samples // self.patch_size
            else:
                num_patches = int(np.ceil(num_samples / self.patch_size))
            
            # Calculate effective batches (groups of patch_accum patches)
            if num_patches > 0:
                num_effective = int(np.ceil(num_patches / self.patch_accum))
                total_effective += num_effective
        
        # For distributed training, divide by world size
        if self.is_distributed:
            # Each GPU processes 1/world_size of the patches per species
            # Recalculate with distributed patches
            total_effective = 0
            for species, indices in dataset.species_to_indices.items():
                num_samples = len(indices)
                
                if drop_last:
                    num_patches = num_samples // self.patch_size
                else:
                    num_patches = int(np.ceil(num_samples / self.patch_size))
                
                # Patches are distributed round-robin across GPUs
                patches_per_gpu = num_patches // self.world_size
                if num_patches > 0:
                    num_effective = int(np.ceil(patches_per_gpu / self.patch_accum))
                    total_effective += num_effective
        
        return max(1, total_effective)

    def _accumulate_single_species_patches(
        self, 
        data_iter: Iterator, 
        pending_patch: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
        """
        Accumulate up to patch_accum patches, ensuring all are from the same species.
        
        When data is limited (e.g., bugfix mode), a species may not have enough
        batches to fill patch_accum. This method stops accumulating when:
        1. We've collected patch_accum patches, OR
        2. We encounter a patch from a different species, OR
        3. The iterator is exhausted
        
        Args:
            data_iter: Iterator over batches
            pending_patch: A patch from a previous call that couldn't be used
                          (because it was from a different species)
        
        Returns:
            Tuple of:
            - patches: List of patches (all same species)
            - next_pending: A patch to carry over if we hit a species boundary, else None
            - exhausted: True if the iterator is exhausted
        """
        patches = []
        current_species = None
        exhausted = False
        
        # First, use any pending patch from previous call
        if pending_patch is not None:
            patches.append(pending_patch)
            current_species = self._get_patch_species(pending_patch)
        
        while len(patches) < self.patch_accum:
            try:
                patch = next(data_iter)
            except StopIteration:
                exhausted = True
                break
            
            patch_species = self._get_patch_species(patch)
            
            if current_species is None:
                # First patch in this group
                current_species = patch_species
                patches.append(patch)
            elif patch_species == current_species:
                # Same species, add to group
                patches.append(patch)
            else:
                # Different species - return current patches and save this one for next call
                return patches, patch, exhausted
        
        return patches, None, exhausted

    # ==================== Forward Pass Helpers ====================
    def _resolve_embeddings(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add embeddings to patch if using precomputed mode with basic collator.
        """
        if self.embedding_lookup is not None and 'ids_a' in patch:
            if 'a' in patch and 'b' in patch and 'a_mask' in patch and 'b_mask' in patch:
                return patch
            patch = self.embedding_lookup.add_embeddings(patch)
            if self.pin_embeddings and torch.cuda.is_available():
                if patch['a'].device.type == 'cpu':
                    patch['a'] = patch['a'].pin_memory()
                    patch['b'] = patch['b'].pin_memory()
                    patch['a_mask'] = patch['a_mask'].pin_memory()
                    patch['b_mask'] = patch['b_mask'].pin_memory()
            return patch
        return patch
    
    def _resolve_all_embeddings(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve embeddings for all patches in a group.
        
        This is the method used by AsyncEmbeddingResolver for prefetching.
        """
        if self._resolve_executor is None or len(patches) == 1:
            return [self._resolve_embeddings(p) for p in patches]
        return list(self._resolve_executor.map(self._resolve_embeddings, patches))
    
    def _create_async_resolver(self) -> Optional[AsyncEmbeddingResolver]:
        """
        Create an async embedding resolver if using precomputed mode.
        
        Returns:
            AsyncEmbeddingResolver if precomputed mode, None otherwise
        """
        if self.embed_mode == EmbedMode.PRECOMPUTED and self.embedding_lookup is not None:
            # Create a wrapper that resolves all patches in a group
            class PatchGroupResolver:
                def __init__(self, trainer):
                    self.trainer = trainer
                
                def add_embeddings(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                    return self.trainer._resolve_all_embeddings(patches)
            
            return AsyncEmbeddingResolver(PatchGroupResolver(self), max_queue_size=self.async_resolve_queue)
        return None
    
    def _forward_model(self, patch: Dict[str, Any], both_directions: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through model, handling both embedding modes.
        
        Returns:
            Tuple of (a_features, b_features)
        """
        if not self.same_encoder:
            both_directions = False

        if self.embed_mode == EmbedMode.ON_THE_FLY:
            output_ab = self.model(
                input_ids_a=patch['input_ids_a'].to(self.device, dtype=torch.long, non_blocking=True),
                input_ids_b=patch['input_ids_b'].to(self.device, dtype=torch.long, non_blocking=True),
                attention_mask_a=patch['attention_mask_a'].to(self.device, non_blocking=True),
                attention_mask_b=patch['attention_mask_b'].to(self.device, non_blocking=True),
            )
            if both_directions:
                output_ba = self.model(
                    input_ids_a=patch['input_ids_b'].to(self.device, dtype=torch.long, non_blocking=True),
                    input_ids_b=patch['input_ids_a'].to(self.device, dtype=torch.long, non_blocking=True),
                    attention_mask_a=patch['attention_mask_b'].to(self.device, non_blocking=True),
                    attention_mask_b=patch['attention_mask_a'].to(self.device, non_blocking=True),
                )
                return output_ab.a, output_ab.b, output_ba.a, output_ba.b
            else:
                return output_ab.a, output_ab.b
        else:  # PRECOMPUTED
            output_ab = self.model(
                a=patch['a'].to(self.device, non_blocking=True),
                b=patch['b'].to(self.device, non_blocking=True),
                a_mask=patch['a_mask'].to(self.device, non_blocking=True),
                b_mask=patch['b_mask'].to(self.device, non_blocking=True),
            )
            if both_directions:
                output_ba = self.model(
                    a=patch['b'].to(self.device, non_blocking=True),
                    b=patch['a'].to(self.device, non_blocking=True),
                    a_mask=patch['b_mask'].to(self.device, non_blocking=True),
                    b_mask=patch['a_mask'].to(self.device, non_blocking=True),
                )
                return output_ab.a, output_ab.b, output_ba.a, output_ba.b
            else:
                return output_ab.a, output_ab.b

    # ==================== Training/Eval Steps ====================
    def train_step(self, patches: List[Dict[str, Any]]) -> Tuple[torch.Tensor, float, float]:
        """
        Training step processing multiple patches.
        
        Works with both embedding modes automatically.
        """
        self._validate_patches_single_species(patches)
        
        # Resolve embeddings if using precomputed mode with basic collator
        patches = [self._resolve_embeddings(p) for p in patches]
        
        all_ids_a, all_ids_b, all_confidences = [], [], []
        for patch in patches:
            all_ids_a.extend(patch['ids_a'])
            all_ids_b.extend(patch['ids_b'])
            all_confidences.append(patch['confidences'])
        
        all_confidences = torch.cat(all_confidences).to(self.device)
        effective_batch_size = len(all_ids_a)
        
        # Use autocast for mixed precision if enabled
        amp_context = autocast('cuda', dtype=self.amp_dtype) if self.use_amp else nullcontext()
        
        with amp_context:
            a_features, b_features = [], []
            for patch in patches:
                a, b = self._forward_model(patch)
                a_features.append(a)
                b_features.append(b)
            
            a = torch.cat(a_features, dim=0)
            b = torch.cat(b_features, dim=0)

            if self.loss_type == 'cosine':
                a = F.normalize(a, p=2, dim=-1)
                b = F.normalize(b, p=2, dim=-1)
            
            logits = torch.matmul(a, b.transpose(-1, -2))

            targets = torch.eye(effective_batch_size, device=self.device).flatten()
            targets = self._find_violations(all_ids_a, all_ids_b, targets)

            loss_matrix = self.loss_fct(logits.view(-1), targets.view(-1))
            loss_matrix = loss_matrix.reshape(effective_batch_size, effective_batch_size)
            targets_matrix = targets.reshape(effective_batch_size, effective_batch_size)
            weighted_loss_matrix = loss_matrix * all_confidences.unsqueeze(-1)
            loss = mean_over_non_ignored_entries(weighted_loss_matrix, targets_matrix)

        # Check for NaN/Inf if halt_on_nan is enabled (used in sweeps to skip bad configs)
        if self.halt_on_nan and (torch.isnan(loss) or torch.isinf(loss)):
            raise TrainingNaNError(f"NaN/Inf loss detected in train_step: {loss.item()}")

        if self.loss_type != 'cosine':
            logits = logits.detach().sigmoid()
        
        mask = torch.eye(effective_batch_size, device=self.device, dtype=torch.bool)
        pos_values = logits[mask]
        neg_values = logits[~mask]
        
        if self.is_distributed:
            pos_sum = pos_values.sum()
            neg_sum = neg_values.sum()
            pos_count = torch.tensor([float(pos_values.numel())], device=self.device, dtype=pos_sum.dtype)
            neg_count = torch.tensor([float(neg_values.numel())], device=self.device, dtype=neg_sum.dtype)
            stats = torch.stack([pos_sum, pos_count, neg_sum, neg_count])
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            pos_sum = stats[0].item()
            pos_count = stats[1].item()
            neg_sum = stats[2].item()
            neg_count = stats[3].item()
            avg_pos_value = (pos_sum / pos_count) if pos_count > 0 else float('nan')
            avg_neg_value = (neg_sum / neg_count) if neg_count > 0 else float('nan')
        else:
            avg_pos_value = float(pos_values.mean().cpu())
            avg_neg_value = float(neg_values.mean().cpu())

        # Explicitly delete large intermediate tensors to free GPU memory
        del a_features, b_features, a, b, logits, targets, mask, all_confidences, pos_values, neg_values

        return loss, avg_pos_value, avg_neg_value

    @torch.no_grad()
    def eval_step(self, patches: List[Dict[str, Any]], both_directions: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluation step processing multiple patches.
        
        Works with both embedding modes automatically.
        Note: During evaluation, batches may contain samples from multiple species
        (random sampling is used for efficiency).
        """
        # Note: We don't validate single-species during evaluation since we use random sampling
        patches = [self._resolve_embeddings(p) for p in patches]
        
        all_ids_a, all_ids_b, all_confidences = [], [], []
        for patch in patches:
            all_ids_a.extend(patch['ids_a'])
            all_ids_b.extend(patch['ids_b'])
            all_confidences.append(patch['labels'])
        
        all_confidences = torch.cat(all_confidences).to(self.device)
        effective_batch_size = len(all_ids_a)
        
        # Use autocast for mixed precision if enabled
        amp_context = autocast('cuda', dtype=self.amp_dtype) if self.use_amp else nullcontext()
        
        with amp_context:
            ab_a_features, ab_b_features, ba_a_features, ba_b_features = [], [], [], []
            for patch in patches:
                if both_directions:
                    ab_a, ab_b, ba_a, ba_b = self._forward_model(patch, both_directions=True)
                else:
                    ab_a, ab_b = self._forward_model(patch)
                ab_a_features.append(ab_a)
                ab_b_features.append(ab_b)
                if both_directions:
                    ba_a_features.append(ba_a)
                    ba_b_features.append(ba_b)
                
            ab_a = torch.cat(ab_a_features, dim=0)
            ab_b = torch.cat(ab_b_features, dim=0)
            if both_directions:
                ba_a = torch.cat(ba_a_features, dim=0)
                ba_b = torch.cat(ba_b_features, dim=0)

            if self.loss_type == 'cosine':
                ab_a = F.normalize(ab_a, p=2, dim=-1)
                ab_b = F.normalize(ab_b, p=2, dim=-1)
                if both_directions:
                    ba_a = F.normalize(ba_a, p=2, dim=-1)
                    ba_b = F.normalize(ba_b, p=2, dim=-1)

            ab_logits = torch.matmul(ab_a, ab_b.transpose(-1, -2))
            if both_directions:
                ba_logits = torch.matmul(ba_a, ba_b.transpose(-1, -2))

            targets = torch.eye(effective_batch_size, device=self.device).flatten()
            targets = self._find_violations(all_ids_a, all_ids_b, targets)

            if both_directions:
                logits = (ab_logits + ba_logits) / 2.0
            else:
                logits = ab_logits

            loss_matrix = self.loss_fct(logits.view(-1), targets.view(-1))
            loss_matrix = loss_matrix.reshape(effective_batch_size, effective_batch_size)
            targets_matrix = targets.reshape(effective_batch_size, effective_batch_size)
            weighted_loss_matrix = loss_matrix * all_confidences.unsqueeze(-1)
            loss = mean_over_non_ignored_entries(weighted_loss_matrix, targets_matrix)

        # Check for NaN/Inf if halt_on_nan is enabled (used in sweeps to skip bad configs)
        if self.halt_on_nan and (torch.isnan(loss) or torch.isinf(loss)):
            raise TrainingNaNError(f"NaN/Inf loss detected in eval_step: {loss.item()}")

        if self.loss_type != 'cosine':
            ab_logits = ab_logits.detach().sigmoid()
            if both_directions:
                ba_logits = ba_logits.detach().sigmoid()

        if both_directions:
            logits = (ab_logits + ba_logits) / 2.0
        else:
            logits = ab_logits

        logits = logits.flatten().cpu()
        labels = targets.detach().cpu() # already flattened
        confidences = all_confidences.flatten().repeat(effective_batch_size).detach().cpu()

        # Explicitly delete large intermediate tensors to free GPU memory
        del ab_a_features, ab_b_features, ab_a, ab_b, ab_logits, targets, all_confidences
        if both_directions:
            del ba_a_features, ba_b_features, ba_a, ba_b, ba_logits

        return loss, logits, labels, confidences
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, prefix: str = 'test', both_directions: bool = False) -> Dict[str, float]:
        """
        Run evaluation with distributed all-gather.
        
        Uses random sampling across all species for efficiency (unlike training
        which enforces single-species batches).
        """
        if not self.same_encoder:
            both_directions = False

        self._print("Starting evaluation...")
        
        self.model.eval()
        total_loss_sum, total_loss_count = 0.0, 0
        total_logits, total_labels, total_confidences = [], [], []
        
        # Simple iteration - evaluation uses random sampling, no species grouping needed
        total_batches = len(data_loader)
        
        if self.is_main_process:
            progress_bar = tqdm(total=total_batches, desc="Evaluating")
        
        batch_count = 0
        for patch in data_loader:
            # Wrap single patch in list for eval_step compatibility
            eval_output = self.eval_step([patch], both_directions=both_directions)                
            batch_count += 1
            
            if self.is_main_process:
                progress_bar.update(1)
            
            loss, logits, labels, confidences = eval_output
            
            total_loss_sum += loss.item()
            total_loss_count += 1
            total_logits.append(logits)
            total_labels.append(labels)
            total_confidences.append(confidences)

        if self.is_main_process:
            progress_bar.close()
        
        # Gather results from all GPUs
        if batch_count > 0:
            total_logits = torch.cat(total_logits)
            total_labels = torch.cat(total_labels)
            total_confidences = torch.cat(total_confidences)
            
            if self.is_distributed:
                # Use batched all-gather for efficiency (1 communication instead of 3)
                total_logits, total_labels, total_confidences = self._all_gather_tensors_batched(
                    total_logits, total_labels, total_confidences
                )
                loss_stats = torch.tensor(
                    [total_loss_sum, float(total_loss_count)],
                    device=self.device,
                    dtype=torch.float32
                )
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                total_loss_sum = loss_stats[0].item()
                total_loss_count = int(loss_stats[1].item())
            
            total_loss = (total_loss_sum / total_loss_count) if total_loss_count > 0 else float('nan')
        else:
            total_logits = torch.tensor([])
            total_labels = torch.tensor([])
            total_confidences = torch.tensor([])
            total_loss = float('nan')
        
        # Compute metrics on main process only
        if self.is_main_process:
            metrics = self.calculate_metrics(total_logits, total_labels)
            metrics['loss'] = total_loss
            self.log_metrics(metrics, prefix)
            
            print(f"{prefix} metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
        else:
            metrics = {}
        
        # Clean up GPU memory after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._synchronize()
        return metrics

    # ==================== Training Loop ====================
    def train(self):
        """Main training loop with multi-GPU support and async embedding prefetching."""
        best_val_mcc, patience_counter, global_step = 0.0, 0, 0

        # Initial evaluation
        metrics = self.evaluate(self.test_loader, prefix='test', both_directions=False)
        
        # Clean up GPU memory after initial eval before training loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grad_accum = self.args.grad_accum
        
        # Calculate evaluation interval in steps based on eval_examples
        # Total samples per optimizer step = world_size * batch_size * grad_accum
        total_batch_size = self.world_size * self.args.batch_size * grad_accum
        eval_interval_steps = max(1, self.args.eval_examples // total_batch_size)
        self._print(f"Evaluation interval: {eval_interval_steps} steps (approx {self.args.eval_examples} examples)")
        
        # Create async embedding resolver for precomputed mode
        async_resolver = self._create_async_resolver()
        use_async = async_resolver is not None
        prefetch_depth = 0
        if use_async:
            prefetch_depth = self.async_resolve_queue
            self._print(f"Using async embedding prefetching (depth={prefetch_depth})")
        
        for epoch in range(self.args.num_epochs):
            # Update sampler epoch for proper shuffling
            if self.is_distributed:
                self.train_sampler.set_epoch(epoch)
            
            self.model.train()
            losses, clipped_grads, avg_pos_values, avg_neg_values = [], [], [], []
            
            train_iter = iter(self.train_loader)
            # Calculate true effective batches accounting for species boundaries
            drop_last = True  # Train loader uses drop_last=True
            total_effective_batches = self._calculate_effective_batches(self.train_dataset, drop_last)
            
            if self.is_main_process:
                progress_bar = tqdm(
                    total=total_effective_batches // grad_accum,
                    desc=f"Epoch {epoch+1}/{self.args.num_epochs}"
                )

            effective_batch_idx = 0
            pending_patch = None
            exhausted = False
            
            def _prefetch_async() -> None:
                nonlocal pending_patch, exhausted
                if not use_async:
                    return
                while not exhausted and async_resolver.pending_count() < prefetch_depth:
                    next_patches, pending_patch, exhausted = self._accumulate_single_species_patches(
                        train_iter, pending_patch
                    )
                    if not next_patches:
                        break
                    async_resolver.submit(next_patches)
            
            while not exhausted or (use_async and async_resolver.pending_count() > 0):
                # Get patches for this iteration
                if use_async and async_resolver.pending_count() > 0:
                    patches = async_resolver.get()
                else:
                    # No prefetched patches, accumulate synchronously
                    patches, pending_patch, exhausted = self._accumulate_single_species_patches(
                        train_iter, pending_patch
                    )
                
                if not patches:
                    break

                # Async prefetching: keep the queue filled while we train on current
                if use_async:
                    _prefetch_async()

                loss, avg_pos_value, avg_neg_value = self.train_step(patches)
                loss = loss / grad_accum

                if self.args.bugfix and global_step % 10 == 0 and self.is_main_process:
                    print('Patch preview')
                    print('-' * 100)
                    for k, v in patches[0].items():
                        print(f"{k}: {v[:10]}")
                    print('-' * 100)

                losses.append(loss.item() * grad_accum)
                avg_pos_values.append(avg_pos_value)
                avg_neg_values.append(avg_neg_value)

                # Determine if this is the last step before optimizer.step()
                is_sync_step = (effective_batch_idx + 1) % grad_accum == 0 or (effective_batch_idx + 1) == total_effective_batches
                
                # Use no_sync() for intermediate accumulation steps to avoid redundant gradient syncs
                # Only sync gradients on the final accumulation step before optimizer.step()
                sync_context = nullcontext() if (not self.is_distributed or is_sync_step) else self.model.no_sync()
                with sync_context:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if is_sync_step:
                    if self.use_amp:
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.optimizer)
                    clipped_grad = self.auto_grad_clipper.clip_gradients() if self.auto_grad_clipper else 0
                    clipped_grads.append(clipped_grad)
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    global_step += 1
                    
                    if self.is_main_process:
                        progress_bar.update(1)

                    self.model.zero_grad(set_to_none=True)

                    if global_step % 100 == 0 and global_step > 0 and self.is_main_process:
                        avg_loss = sum(losses) / len(losses)
                        avg_pos_value = sum(avg_pos_values) / len(avg_pos_values)
                        avg_neg_value = sum(avg_neg_values) / len(avg_neg_values)
                        avg_clipped_grad = sum(clipped_grads) / len(clipped_grads) if clipped_grads else 0

                        train_metrics = {
                            "loss": avg_loss,
                            "avg_pos": avg_pos_value,
                            "avg_neg": avg_neg_value,
                            "avg_clipped_grad": avg_clipped_grad,
                        }
                        
                        if os.environ['WANDB_AVAILABLE'] == 'true' and self.is_main_process:
                            wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=global_step)
                        
                        self.metrics_logger.info(f"=== TRAIN STEP {global_step} ===")
                        for k, v in train_metrics.items():
                            if isinstance(v, float):
                                self.metrics_logger.info(f"train/{k}: {v:.4f}")
                            else:
                                self.metrics_logger.info(f"train/{k}: {v}")
                        self.metrics_logger.info("=" * 30)

                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'avg_pos': f'{avg_pos_value:.4f}',
                            'avg_neg': f'{avg_neg_value:.4f}',
                        })

                        losses, clipped_grads, avg_pos_values, avg_neg_values = [], [], [], []

                    # === STEP-BASED EVALUATION ===
                    if global_step % eval_interval_steps == 0 and global_step > 0:
                        if patience_counter >= self.args.patience:
                            self._print(f"Early stopping after {patience_counter} evaluations without improvement")
                            exhausted = True  # Break out of inner loop
                            break
                        
                        metrics = self.evaluate(self.valid_loader, prefix='valid', both_directions=False)
                        
                        if self.is_main_process:
                            mcc = metrics['balanced_mcc']
                            if mcc > best_val_mcc:
                                print(f"Step {global_step}: New best validation MCC: {mcc:.4f}")
                                best_val_mcc = mcc
                                patience_counter = 0
                                self.save_model(global_step, is_best=True)
                            else:
                                patience_counter += 1
                                print(f"Step {global_step}: Val MCC {mcc:.4f} (best: {best_val_mcc:.4f}, patience: {patience_counter}/{self.args.patience})")
                                self.save_model(global_step, is_best=False)
                        
                        # Broadcast patience counter for distributed training
                        patience_counter = self._broadcast_value(patience_counter, src=0)

                effective_batch_idx += 1
                    
            if self.is_main_process:
                progress_bar.close()
            
            # Check early stopping after epoch if needed (already handled by step-based eval if it triggered)
            if patience_counter >= self.args.patience:
                break
        
        # Shutdown async resolver if used
        if use_async:
            async_resolver.shutdown()
        if self._resolve_executor is not None:
            self._resolve_executor.shutdown(wait=True)
            self._resolve_executor = None
        
        # Load best model for final test
        self.load_best_model()
        
        # Final validation
        metrics = self.evaluate(self.valid_loader, prefix='valid', both_directions=True)
        
        # Final test
        metrics = self.evaluate(self.test_loader, prefix='test', both_directions=True)
        
        if self.is_main_process:
            model_to_push = self._unwrap_model()
            # Skip hub upload if --skip_hub_upload is set (useful for sweeps)
            if self.args.skip_hub_upload:
                self._print("Skipping final hub upload (--skip_hub_upload is set)")
            else:
                try:
                    # Use safe_serialization=False to handle shared weights (e.g., when same_encoder=True)
                    model_to_push.push_to_hub(self.hf_save_path, private=True, safe_serialization=False)
                except RuntimeError as e:
                    error_text = str(e)
                    if 'shared tensors' not in error_text:
                        raise
                    self._print(
                        "push_to_hub failed due to shared tensor aliases; "
                        "saving fallback state_dict checkpoint in final_push and uploading that folder."
                    )
                    final_push_dir = os.path.join(self.save_dir, 'final_push')
                    os.makedirs(final_push_dir, exist_ok=True)
                    save_format = self._save_model_artifacts(model_to_push, final_push_dir)
                    self._print(f"Final push fallback saved using format: {save_format}")
                    create_repo(repo_id=self.hf_save_path, private=True, exist_ok=True)
                    upload_folder(
                        folder_path=final_push_dir,
                        repo_id=self.hf_save_path,
                        repo_type="model",
                        path_in_repo="final_push",
                    )
                    self._print(f"Fallback final model uploaded to: {self.hf_save_path}/final_push")
        
        return self._unwrap_model()

    def get_model(self) -> nn.Module:
        """
        Initialize and return the Synteract4/5 model.
        
        Returns:
        - SynteractModel (with PLM embedder) for ON_THE_FLY mode
        - SynteractBaseModel (without PLM) for PRECOMPUTED mode
        """
        model_version = self.args.model_version
        if model_version == "synteract4":
            config = Synteract4Config(
                input_size=self.args.input_size, # kept for legacy compat
                a_input_size=self.a_embed_dim if self.embed_mode == EmbedMode.PRECOMPUTED else self.args.a_input_size,
                b_input_size=self.b_embed_dim if self.embed_mode == EmbedMode.PRECOMPUTED else self.args.b_input_size,
                hidden_size=self.args.hidden_size,
                proj_size=self.args.proj_size,
                output_size=self.args.output_size,
                expansion_ratio=self.args.expansion_ratio,
                head_size=self.args.head_size,
                num_layers=self.args.num_layers,
                dropout=self.args.dropout,
                rotary=self.args.rotary,
                bias=self.args.bias,
                ffn_type=self.args.ffn_type,
                pooling_types=self.args.pooling_types,
                a_encoder_path=self.args.a_encoder_path,
                b_encoder_path=self.args.b_encoder_path,
                a_encoder_trainable=self.args.a_encoder_trainable,
                b_encoder_trainable=self.args.b_encoder_trainable,
                a_encoder_layer_processing=self.args.a_encoder_layer_processing,
                b_encoder_layer_processing=self.args.b_encoder_layer_processing,
                a_encoder_precision=self.args.a_encoder_precision,
                b_encoder_precision=self.args.b_encoder_precision,
                kernel_size=self.args.kernel_size,
            )
            base_cls = Synteract4BaseModel
            full_cls = Synteract4Model
        elif model_version == "synteract5":
            config = Synteract5Config(
                input_size=self.args.input_size, # kept for legacy compat
                a_input_size=self.a_embed_dim if self.embed_mode == EmbedMode.PRECOMPUTED else self.args.a_input_size,
                b_input_size=self.b_embed_dim if self.embed_mode == EmbedMode.PRECOMPUTED else self.args.b_input_size,
                hidden_size=self.args.hidden_size,
                proj_size=self.args.proj_size,
                output_size=self.args.output_size,
                expansion_ratio=self.args.expansion_ratio,
                head_size=self.args.head_size,
                dropout=self.args.dropout,
                rotary=self.args.rotary,
                bias=self.args.bias,
                ffn_type=self.args.ffn_type,
                pooling_types=self.args.pooling_types,
                a_encoder_path=self.args.a_encoder_path,
                b_encoder_path=self.args.b_encoder_path,
                a_encoder_trainable=self.args.a_encoder_trainable,
                b_encoder_trainable=self.args.b_encoder_trainable,
                a_encoder_layer_processing=self.args.a_encoder_layer_processing,
                b_encoder_layer_processing=self.args.b_encoder_layer_processing,
                a_encoder_precision=self.args.a_encoder_precision,
                b_encoder_precision=self.args.b_encoder_precision,
                kernel_size=self.args.kernel_size,
            )
            base_cls = Synteract5BaseModel
            full_cls = Synteract5Model
        else:
            raise ValueError(f"Invalid model_version: {model_version}")
        
        self._print(f"Model config:\n{config}")
        self._print(f"Same encoder: {config.same_encoder}")
        self._print(f"Swap allowed during training: {self.swap_allowed}")
        
        if self.embed_mode == EmbedMode.PRECOMPUTED:
            # Use base model without PLM embedder - embeddings are precomputed
            model = base_cls(config)
            self._print(f"Initialized {model_version} base model (for precomputed embeddings)")
        else:
            # Use full model with PLM embedder for on-the-fly embedding
            model = full_cls(config)
            self._print(f"Initialized {model_version} model (with PLM for on-the-fly embedding)")
            if config.same_encoder:
                self._print(f"  Using shared encoder: {config.a_encoder_path}")
            else:
                self._print(f"  A-track encoder: {config.a_encoder_path}")
                self._print(f"  B-track encoder: {config.b_encoder_path}")
        
        if self.is_main_process:
            summary(model)
            print(model)
        return model

