import numpy as np
import os
import queue
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torch.amp import autocast, GradScaler
from torchinfo import summary
from torch.utils.data import DistributedSampler, RandomSampler, DataLoader, Dataset as TorchDataset
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from contextlib import nullcontext
from tqdm.auto import tqdm

from models.synteract.synteract_vec.model import SynteractVecBaseModel
from models.synteract.synteract_vec.config import SynteractVecConfig
from losses.softbce import SoftBCEWithLogitsLoss
from losses.contrastive import CosineEmbeddingLoss
from data_classes.torch_datasets import BasicPairedDataset
from data_classes.samplers import OneSpeciesBatchSampler, DistributedOneSpeciesBatchSampler
from data_classes.collators import PooledPairedCollator, PooledEmbeddingLookup
from utils.embedding import embed_and_pool_dataset
from utils.metrics import calculate_classification_metrics, calculate_topk_metrics
from utils.exceptions import TrainingNaNError
from utils.violations import build_interaction_partners, find_violation_indices
from .base_trainer import BaseTrainer
from .loss_utils import mean_over_non_ignored_entries


if os.environ.get('WANDB_AVAILABLE', 'false') == 'true':
    import wandb


class AsyncPooledEmbeddingResolver:
    """
    Asynchronous embedding resolver that prefetches pooled embeddings in a background thread.
    
    This overlaps embedding lookup with GPU computation, hiding the latency of
    dictionary lookups and tensor stacking for the next batch while the current
    batch is being processed on GPU.
    
    Optimized for pooled embeddings which are simpler than full sequence embeddings.
    
    Usage:
        resolver = AsyncPooledEmbeddingResolver(embedding_lookup)
        
        # Start prefetching first batch
        resolver.submit(first_patches)
        
        for patches in patch_iterator:
            # Get the prefetched result (blocks if not ready)
            current_patches = resolver.get()
            
            # Start prefetching next batch while processing current
            resolver.submit(patches)
            
            # Process current_patches on GPU...
        
        # Don't forget the last batch
        last_patches = resolver.get()
        
        resolver.shutdown()
    """
    
    def __init__(self, embedding_lookup: PooledEmbeddingLookup, max_queue_size: int = 2):
        """
        Args:
            embedding_lookup: PooledEmbeddingLookup instance with add_embeddings method
            max_queue_size: Maximum number of prefetched batches to queue
        """
        self.embedding_lookup = embedding_lookup
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = False
    
    def _resolve_patches(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve embeddings for a list of patches."""
        return [self.embedding_lookup.add_embeddings(p) for p in patches]
    
    def submit(self, patches: List[Dict[str, Any]]) -> None:
        """Submit a list of patches for async embedding resolution."""
        if self._shutdown:
            return
        future = self.executor.submit(self._resolve_patches, patches)
        self.result_queue.put(future)
    
    def get(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get the next resolved batch of patches. Blocks until ready.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            List of patches with embeddings added
        """
        future = self.result_queue.get(timeout=timeout)
        return future.result()
    
    def shutdown(self) -> None:
        """Shutdown the executor and clean up."""
        self._shutdown = True
        self.executor.shutdown(wait=True)


class BaseSynteractVecTrainer(BaseTrainer):
    """
    Base trainer for Synteract-Vec model with pooled embeddings.
    
    This trainer pre-embeds and pools sequences before training, storing
    compact 1D vectors instead of full sequence embeddings.
    
    Child classes must set these attributes in __init__ BEFORE calling prep_for_training():
        - self.train_dataframes: Dict[str, pd.DataFrame]
        - self.valid_dataframes: Dict[str, pd.DataFrame] 
        - self.test_dataframes: Dict[str, pd.DataFrame]
        - self.seq_dict: Dict[str, str] - protein_id -> sequence
        - self.interaction_set: Set[str] - known interactions for violation detection
        - self.skip_violations: bool - whether to skip violation detection
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # Embedding dicts (will be populated in get_embeddings)
        self.a_embed_dict = None
        self.b_embed_dict = None
        self.a_pooled_dim = None
        self.b_pooled_dim = None
        
        # Sequence dictionaries
        self.seq_dict = None
        self.seq_dict_a = None
        self.seq_dict_b = None
        self.interaction_set = None
        self.interaction_partners = {}
        self.skip_violations = False
        
        self.embedding_lookup = None
        
        # Dual encoder support
        self.same_encoder = args.same_encoder
        self.swap_allowed = self.same_encoder
        
        # NaN detection
        self.halt_on_nan = getattr(args, 'halt_on_nan', False)
        
        # Interaction partners lookup (built in prep_for_training)
        self.interaction_partners: Dict[str, set] = {}
        
        # Batch sizing
        self.patch_size = args.patch_size
        self.patch_accum = args.patch_accum
        self.batch_size = args.batch_size
        
        # Mixed precision (AMP) setup
        self.use_amp = args.use_amp
        self.amp_dtype = torch.bfloat16
        if self.use_amp:
            self.scaler = GradScaler('cuda')
            self._print(f"Using Automatic Mixed Precision (AMP) with {self.amp_dtype}")
        else:
            self.scaler = None

    def prep_for_training(self):
        """Prepare for training."""
        # Get pooled embeddings
        self.get_embeddings()
        
        # Build interaction partner lookup for fast O(batch) violation detection
        self._build_interaction_partners()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get model
        self.model = self.get_model()
        
        # Move to device and compile
        self.model = self.model.to(self.device)
        self.model = self._compile_model(self.model)
        
        # Wrap with DDP if distributed
        self.model = self._wrap_model_ddp(self.model)
        
        # Create datasets and loaders
        self.train_dataset, self.valid_dataset, self.test_dataset = self.get_datasets()
        self.train_loader, self.valid_loader, self.test_loader = self.get_data_loaders()
        
        self.optimizer, self.scheduler = self.get_optimizers()
        self.loss_fct = self.get_loss_fct()
        
        if self.is_main_process:
            self.setup_logging()

    def get_embeddings(self):
        """Compute and cache pooled embeddings."""
        self._synchronize()
        
        assert self.seq_dict is not None or self.seq_dict_a is not None, (
            "Missing seq_dict/seq_dict_a for embedding."
        )
        seq_dict_a = self.seq_dict_a if self.seq_dict_a is not None else self.seq_dict
        seq_dict_b = self.seq_dict_b if self.seq_dict_b is not None else seq_dict_a
        
        pooling_types = self.args.pooling_types
        
        if self.same_encoder:
            merged_seq_dict = dict(seq_dict_a)
            for seq_id, seq in seq_dict_b.items():
                if seq_id in merged_seq_dict:
                    assert merged_seq_dict[seq_id] == seq, f"Sequence mismatch for id '{seq_id}'"
                else:
                    merged_seq_dict[seq_id] = seq

            all_ids = list(set(merged_seq_dict.keys()))
            self._print(f"Embedding and pooling {len(all_ids)} unique ids (shared encoder)")
            self._print(f"Pooling types: {pooling_types}")
            
            embed_dict, pooled_dim = embed_and_pool_dataset(
                plm_path=self.args.a_encoder_path,
                ids=all_ids,
                seq_dict=merged_seq_dict,
                pooling_types=pooling_types,
                max_len=self.args.max_length_a,
                encoder_precision=self.args.a_encoder_precision,
                device=self.device,
                batch_size=self.args.embedding_batch_size,
                bugfix=self.args.bugfix,
                use_compile=self.args.use_compile,
                print_fn=self._print,
            )
            
            self.a_embed_dict = embed_dict
            self.b_embed_dict = embed_dict
            self.a_pooled_dim = pooled_dim
            self.b_pooled_dim = pooled_dim
        else:
            a_ids = list(set(seq_dict_a.keys()))
            b_ids = list(set(seq_dict_b.keys()))
            self._print(f"Embedding and pooling {len(a_ids)} A-track ids")
            self._print(f"Embedding and pooling {len(b_ids)} B-track ids")
            self._print(f"Pooling types: {pooling_types}")
            
            a_embed_dict, a_pooled_dim = embed_and_pool_dataset(
                plm_path=self.args.a_encoder_path,
                ids=a_ids,
                seq_dict=seq_dict_a,
                pooling_types=pooling_types,
                max_len=self.args.max_length_a,
                encoder_precision=self.args.a_encoder_precision,
                device=self.device,
                batch_size=self.args.embedding_batch_size,
                bugfix=self.args.bugfix,
                use_compile=self.args.use_compile,
                print_fn=self._print,
                cache_path=f'{self.args.a_encoder_path.split("/")[-1].lower()}_pooled_a_embeddings.pth',
            )
            
            b_embed_dict, b_pooled_dim = embed_and_pool_dataset(
                plm_path=self.args.b_encoder_path,
                ids=b_ids,
                seq_dict=seq_dict_b,
                pooling_types=pooling_types,
                max_len=self.args.max_length_b,
                encoder_precision=self.args.b_encoder_precision,
                device=self.device,
                batch_size=self.args.embedding_batch_size,
                bugfix=self.args.bugfix,
                use_compile=self.args.use_compile,
                print_fn=self._print,
                cache_path=f'{self.args.b_encoder_path.split("/")[-1].lower()}_pooled_b_embeddings.pth',
            )
            
            self.a_embed_dict = a_embed_dict
            self.b_embed_dict = b_embed_dict
            self.a_pooled_dim = a_pooled_dim
            self.b_pooled_dim = b_pooled_dim
        
        self._synchronize()

    def get_datasets(self) -> Tuple[TorchDataset, TorchDataset, TorchDataset]:
        """Create datasets with skip_sequences=True for pooled embedding mode."""
        # For pooled embeddings, we don't need sequences - only IDs for embedding lookup
        # This avoids unnecessary dictionary lookups and string allocations per sample
        train_dataset = BasicPairedDataset(
            self.train_dataframes, 
            seq_dict=None,
            seq_dict_a=None,
            seq_dict_b=None,
            eval_mode=False,
            swap_allowed=self.swap_allowed,
            skip_sequences=True,  # Skip sequence lookups - only IDs needed
        )
        valid_dataset = BasicPairedDataset(
            self.valid_dataframes, 
            seq_dict=None,
            seq_dict_a=None,
            seq_dict_b=None,
            eval_mode=True,
            swap_allowed=self.swap_allowed,
            skip_sequences=True,
        )
        test_dataset = BasicPairedDataset(
            self.test_dataframes, 
            seq_dict=None,
            seq_dict_a=None,
            seq_dict_b=None,
            eval_mode=True,
            swap_allowed=self.swap_allowed,
            skip_sequences=True,
        )
        return train_dataset, valid_dataset, test_dataset

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders with pooled embedding lookup and async prefetching."""
        train_collator = PooledPairedCollator(check_species=True)
        eval_collator = PooledPairedCollator(check_species=False)
        
        # Build species_to_ids mapping for pre-stacking optimization
        # This enables O(1) batch assembly via tensor slicing
        species_to_ids = self._build_species_to_ids()
        
        # Create pooled embedding lookup with species pre-stacking
        self.embedding_lookup = PooledEmbeddingLookup(
            embedding_dict=self.a_embed_dict,
            pooled_dim=self.a_pooled_dim,
            b_embedding_dict=self.b_embed_dict if not self.same_encoder else None,
            b_pooled_dim=self.b_pooled_dim,
            species_to_ids=species_to_ids,
        )
        
        # Training: single-species batch sampling
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
        
        # Validation/test: random sampling
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
        
        loader_kwargs = dict(
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            collate_fn=train_collator,
            **loader_kwargs
        )
        
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

    def _build_species_to_ids(self) -> Dict[int, List[str]]:
        """
        Build a mapping from species to all protein IDs for that species.
        
        This enables the PooledEmbeddingLookup to pre-stack embeddings by species
        for O(1) batch assembly via tensor indexing.
        """
        species_to_ids: Dict[int, set] = {}
        
        # Collect IDs from all dataframes
        for dataframes in [self.train_dataframes, self.valid_dataframes, self.test_dataframes]:
            for org, df in dataframes.items():
                if org not in species_to_ids:
                    species_to_ids[org] = set()
                species_to_ids[org].update(df['IdA'].tolist())
                species_to_ids[org].update(df['IdB'].tolist())
        
        # Convert to lists
        return {species: list(ids) for species, ids in species_to_ids.items()}

    def _create_async_resolver(self) -> Optional[AsyncPooledEmbeddingResolver]:
        """
        Create an async embedding resolver for prefetching.
        
        Returns:
            AsyncPooledEmbeddingResolver if embedding_lookup exists, None otherwise
        """
        if self.embedding_lookup is not None:
            return AsyncPooledEmbeddingResolver(self.embedding_lookup, max_queue_size=2)
        return None

    def get_loss_fct(self):
        """Initialize loss function with class imbalance weighting."""
        pos_weight = (self.batch_size * self.batch_size - self.batch_size) / self.batch_size
        pos_weight = int(pos_weight * 0.85)
        pos_weight = torch.tensor(pos_weight, device=self.device)
        self.loss_type = self.args.loss_type
        if self.loss_type == 'cosine':
            self.loss_fct = CosineEmbeddingLoss(ignore_index=-100.0, pos_weight=pos_weight, reduction='none')
        else:
            self.loss_fct = SoftBCEWithLogitsLoss(ignore_index=-100.0, pos_weight=pos_weight, reduction='none')
        return self.loss_fct

    def get_model(self) -> nn.Module:
        """Initialize and return the SynteractVec model."""
        # Infer input size from pooled embeddings
        input_size_a = self.a_pooled_dim // len(self.args.pooling_types)
        input_size_b = self.b_pooled_dim // len(self.args.pooling_types)
        
        config = SynteractVecConfig(
            input_size=input_size_a,
            a_input_size=input_size_a,
            b_input_size=input_size_b,
            hidden_size=self.args.hidden_size,
            output_size=self.args.output_size,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
            bias=self.args.bias,
            use_residual=self.args.use_residual,
            pooling_types=self.args.pooling_types,
            a_encoder_path=self.args.a_encoder_path,
            b_encoder_path=self.args.b_encoder_path,
            a_encoder_precision=self.args.a_encoder_precision,
            b_encoder_precision=self.args.b_encoder_precision,
            loss_type=self.args.loss_type,
        )
        
        self._print(f"Model config:\n{config}")
        self._print(f"Same encoder: {config.same_encoder}")
        self._print(f"Swap allowed during training: {self.swap_allowed}")
        self._print(f"Pooled input size A: {config.a_pooled_input_size}")
        self._print(f"Pooled input size B: {config.b_pooled_input_size}")
        
        model = SynteractVecBaseModel(config)
        self._print("Initialized SynteractVecBaseModel (for pooled embeddings)")
        
        if self.is_main_process:
            summary(model)
            print(model)
        return model

    # ==================== Metrics ====================
    def _metrics_helper(self, logits: np.ndarray, labels: np.ndarray, prefix: Optional[str] = '') -> Dict[str, float]:
        """Compute classification metrics."""
        logits = np.asarray(logits).flatten()
        labels = np.asarray(labels).flatten()

        assert len(logits) == len(labels), 'Logits and labels must have the same length'
        assert len(logits) > 0 and len(labels) > 0, 'Logits and labels must not be empty'

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

        metrics = calculate_classification_metrics(
            labels, logits, threshold=None, find_optimal_threshold=True
        )
        
        renamed_metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
        return renamed_metrics

    def calculate_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
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

        valid_label_mask = (labels == 0) | (labels == 1)
        assert valid_label_mask.any(), 'No valid labels found (all are violations)'

        finite_mask = np.isfinite(logits) & np.isfinite(labels)
        combined_mask = valid_label_mask & finite_mask

        if not combined_mask.any():
            self._print("[metrics] No finite, non-violation label/logit pairs remain; returning NaN metrics.")
            metrics = _nan_balanced_metrics()
            metrics['ratio'] = float('nan')
            metrics['avg_pos'] = float('nan')
            metrics['avg_neg'] = float('nan')
            metrics['n_total'] = int(len(labels))
            return metrics

        valid_logits = logits[combined_mask]
        valid_labels = labels[combined_mask]

        pos_values_full = valid_logits[valid_labels == 1]
        neg_values_full = valid_logits[valid_labels == 0]
        pos_avg_full = float(pos_values_full.mean()) if pos_values_full.size > 0 else float('nan')
        neg_avg_full = float(neg_values_full.mean()) if neg_values_full.size > 0 else float('nan')
        ratio_full = (pos_avg_full / neg_avg_full) if (neg_avg_full != 0 and np.isfinite(neg_avg_full) and np.isfinite(pos_avg_full)) else float('nan')

        pos_indices = np.where(valid_labels == 1)[0]
        neg_indices = np.where(valid_labels == 0)[0]
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        if n_pos == 0 or n_neg == 0:
            self._print(f"[metrics] Degenerate labels after filtering: n_pos={n_pos}, n_neg={n_neg}; returning NaN metrics.")
            metrics = _nan_balanced_metrics()
            metrics['ratio'] = float('nan')
            metrics['avg_pos'] = float('nan')
            metrics['avg_neg'] = float('nan')
            metrics['n_total'] = int(len(labels))
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
        
        topk_metrics_result = calculate_topk_metrics(labels_bal, logits_bal)
        for k, v in topk_metrics_result.items():
            metrics[k] = v
        
        metrics['ratio'] = ratio_full
        metrics['avg_pos'] = pos_avg_full
        metrics['avg_neg'] = neg_avg_full
        metrics['n_total'] = int(len(labels))
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
        
        Called once during prep_for_training after interaction_set is populated.
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
        """Extract the species ID from a patch."""
        org_a = patch['org_a']
        unique_org_a = org_a.unique()
        assert unique_org_a.numel() == 1, f"Patch has multiple species in org_a: {unique_org_a.tolist()}"
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
            assert unique_org_a.numel() == 1, f"Patch {patch_idx} has multiple species in org_a"
            
            unique_org_b = org_b.unique()
            assert unique_org_b.numel() == 1, f"Patch {patch_idx} has multiple species in org_b"
            
            patch_species_a = unique_org_a.item()
            patch_species_b = unique_org_b.item()
            assert patch_species_a == patch_species_b, f"Patch {patch_idx} has mismatched species"
            
            patch_species = patch_species_a
            
            if group_species is None:
                group_species = patch_species
            else:
                assert patch_species == group_species, f"Patch {patch_idx} has species {patch_species}, but group expected {group_species}"

    def _calculate_effective_batches(self, dataset: TorchDataset, drop_last: bool) -> int:
        """Calculate the true number of effective batches accounting for species boundaries."""
        total_effective = 0
        for species, indices in dataset.species_to_indices.items():
            num_samples = len(indices)
            
            if drop_last:
                num_patches = num_samples // self.patch_size
            else:
                num_patches = int(np.ceil(num_samples / self.patch_size))
            
            if num_patches > 0:
                num_effective = int(np.ceil(num_patches / self.patch_accum))
                total_effective += num_effective
        
        if self.is_distributed:
            total_effective = 0
            for species, indices in dataset.species_to_indices.items():
                num_samples = len(indices)
                
                if drop_last:
                    num_patches = num_samples // self.patch_size
                else:
                    num_patches = int(np.ceil(num_samples / self.patch_size))
                
                patches_per_gpu = num_patches // self.world_size
                if num_patches > 0:
                    num_effective = int(np.ceil(patches_per_gpu / self.patch_accum))
                    total_effective += num_effective
        
        return max(1, total_effective)

    def _accumulate_single_species_patches(self, data_iter, pending_patch=None):
        """Accumulate up to patch_accum patches, ensuring all are from the same species."""
        patches = []
        current_species = None
        exhausted = False
        
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
                current_species = patch_species
                patches.append(patch)
            elif patch_species == current_species:
                patches.append(patch)
            else:
                return patches, patch, exhausted
        
        return patches, None, exhausted

    # ==================== Forward Pass ====================
    def _resolve_embeddings(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Add pooled embeddings to patch."""
        if self.embedding_lookup is not None and 'ids_a' in patch:
            return self.embedding_lookup.add_embeddings(patch)
        return patch

    def _forward_model(self, patch: Dict[str, Any], both_directions: bool = False):
        """Forward pass through model with optimized GPU transfers."""
        if not self.same_encoder:
            both_directions = False

        # Use non_blocking=True for async transfers from pinned memory
        # This overlaps CPU->GPU transfer with GPU computation
        a_gpu = patch['a'].to(self.device, non_blocking=True)
        b_gpu = patch['b'].to(self.device, non_blocking=True)

        output_ab = self.model(a=a_gpu, b=b_gpu)
        
        if both_directions:
            output_ba = self.model(a=b_gpu, b=a_gpu)
            return output_ab.a, output_ab.b, output_ba.a, output_ba.b
        else:
            return output_ab.a, output_ab.b

    # ==================== Training/Eval Steps ====================
    def train_step(self, patches: List[Dict[str, Any]]) -> Tuple[torch.Tensor, float, float]:
        """
        Training step with optimized O(batch) violation detection.
        
        The violation detection uses pre-built interaction partner lookup for O(batch)
        instead of O(batch²) Python loops. Loss computation uses unified GPU operations.
        """
        self._validate_patches_single_species(patches)
        
        # Only resolve if not already resolved (async prefetching resolves ahead of time)
        if patches and 'a' not in patches[0]:
            patches = [self._resolve_embeddings(p) for p in patches]
        
        all_ids_a, all_ids_b, all_confidences = [], [], []
        for patch in patches:
            all_ids_a.extend(patch['ids_a'])
            all_ids_b.extend(patch['ids_b'])
            all_confidences.append(patch['confidences'])
        
        all_confidences = torch.cat(all_confidences).to(self.device, non_blocking=True)
        effective_batch_size = len(all_ids_a)
        
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

            # Create targets and apply fast violation detection
            targets = torch.eye(effective_batch_size, device=self.device).flatten()
            targets = self._find_violations(all_ids_a, all_ids_b, targets)

            loss_matrix = self.loss_fct(logits.view(-1), targets.view(-1))
            loss_matrix = loss_matrix.reshape(effective_batch_size, effective_batch_size)
            targets_matrix = targets.reshape(effective_batch_size, effective_batch_size)
            weighted_loss_matrix = loss_matrix * all_confidences.unsqueeze(-1)
            loss = mean_over_non_ignored_entries(weighted_loss_matrix, targets_matrix)

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

        del a_features, b_features, a, b, logits, targets, mask, all_confidences, pos_values, neg_values

        return loss, avg_pos_value, avg_neg_value

    @torch.no_grad()
    def eval_step(self, patches: List[Dict[str, Any]], both_directions: bool = False):
        """Evaluation step processing multiple patches."""
        # Only resolve if not already resolved
        if patches and 'a' not in patches[0]:
            patches = [self._resolve_embeddings(p) for p in patches]
        
        all_ids_a, all_ids_b, all_confidences = [], [], []
        for patch in patches:
            all_ids_a.extend(patch['ids_a'])
            all_ids_b.extend(patch['ids_b'])
            all_confidences.append(patch['labels'])
        
        all_confidences = torch.cat(all_confidences).to(self.device)
        effective_batch_size = len(all_ids_a)
        
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
        labels = targets.detach().cpu()
        confidences = all_confidences.flatten().repeat(effective_batch_size).detach().cpu()

        del ab_a_features, ab_b_features, ab_a, ab_b, ab_logits, targets, all_confidences
        if both_directions:
            del ba_a_features, ba_b_features, ba_a, ba_b, ba_logits

        return loss, logits, labels, confidences

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, prefix: str = 'test', both_directions: bool = False) -> Dict[str, float]:
        """Run evaluation with distributed all-gather."""
        if not self.same_encoder:
            both_directions = False

        self._print("Starting evaluation...")
        
        self.model.eval()
        total_loss_sum, total_loss_count = 0.0, 0
        total_logits, total_labels, total_confidences = [], [], []
        
        total_batches = len(data_loader)
        
        if self.is_main_process:
            progress_bar = tqdm(total=total_batches, desc="Evaluating")
        
        batch_count = 0
        
        for patch in data_loader:
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
        
        if batch_count > 0:
            total_logits = torch.cat(total_logits)
            total_labels = torch.cat(total_labels)
            total_confidences = torch.cat(total_confidences)
            
            if self.is_distributed:
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._synchronize()
        return metrics

    # ==================== Training Loop ====================
    def train(self):
        """Main training loop with async embedding prefetching for improved GPU utilization."""
        best_val_mcc, patience_counter, global_step = 0.0, 0, 0

        metrics = self.evaluate(self.test_loader, prefix='test', both_directions=False)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grad_accum = self.args.grad_accum
        
        total_batch_size = self.world_size * self.args.batch_size * grad_accum
        eval_interval_steps = max(1, self.args.eval_examples // total_batch_size)
        self._print(f"Evaluation interval: {eval_interval_steps} steps (approx {self.args.eval_examples} examples)")
        
        # Create async embedding resolver for prefetching
        async_resolver = self._create_async_resolver()
        use_async = async_resolver is not None
        if use_async:
            self._print("Using async embedding prefetching for improved GPU utilization")
        
        for epoch in range(self.args.num_epochs):
            if self.is_distributed:
                self.train_sampler.set_epoch(epoch)
            
            self.model.train()
            losses, clipped_grads, avg_pos_values, avg_neg_values = [], [], [], []
            
            train_iter = iter(self.train_loader)
            drop_last = True
            total_effective_batches = self._calculate_effective_batches(self.train_dataset, drop_last)
            
            if self.is_main_process:
                progress_bar = tqdm(
                    total=total_effective_batches // grad_accum,
                    desc=f"Epoch {epoch+1}/{self.args.num_epochs}"
                )

            effective_batch_idx = 0
            pending_patch = None
            exhausted = False
            
            # For async prefetching: track prefetched patches and next batch info
            prefetched_patches = None
            next_patches_info = None  # (patches, pending_patch, exhausted)
            
            while not exhausted:
                # Get patches for this iteration
                if prefetched_patches is not None:
                    # Use prefetched patches from async resolver
                    patches = prefetched_patches
                    prefetched_patches = None
                    # Restore state from when we submitted these patches
                    if next_patches_info is not None:
                        pending_patch, exhausted = next_patches_info[1], next_patches_info[2]
                        next_patches_info = None
                else:
                    # No prefetched patches, accumulate and resolve synchronously
                    raw_patches, pending_patch, exhausted = self._accumulate_single_species_patches(
                        train_iter, pending_patch
                    )
                    if not raw_patches:
                        break
                    # Resolve embeddings synchronously for first batch
                    patches = [self._resolve_embeddings(p) for p in raw_patches]
                
                if not patches:
                    break

                # Async prefetching: start prefetching next batch while we train on current
                if use_async and not exhausted:
                    # Get next batch info
                    next_raw_patches, next_pending, next_exhausted = self._accumulate_single_species_patches(
                        train_iter, pending_patch
                    )
                    if next_raw_patches:
                        # Submit for async resolution (embeddings resolved in background thread)
                        async_resolver.submit(next_raw_patches)
                        next_patches_info = (next_raw_patches, next_pending, next_exhausted)
                    else:
                        exhausted = next_exhausted
                        pending_patch = next_pending

                loss, avg_pos_value, avg_neg_value = self.train_step(patches)
                
                # If we submitted async work, get the result for next iteration
                if use_async and next_patches_info is not None:
                    prefetched_patches = async_resolver.get()
                
                loss = loss / grad_accum

                losses.append(loss.item() * grad_accum)
                avg_pos_values.append(avg_pos_value)
                avg_neg_values.append(avg_neg_value)

                is_sync_step = (effective_batch_idx + 1) % grad_accum == 0 or (effective_batch_idx + 1) == total_effective_batches
                
                sync_context = nullcontext() if (not self.is_distributed or is_sync_step) else self.model.no_sync()
                with sync_context:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if is_sync_step:
                    if self.use_amp:
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
                        
                        if os.environ.get('WANDB_AVAILABLE', 'false') == 'true' and self.is_main_process:
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

                    if global_step % eval_interval_steps == 0 and global_step > 0:
                        if patience_counter >= self.args.patience:
                            self._print(f"Early stopping after {patience_counter} evaluations without improvement")
                            exhausted = True
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
                        
                        patience_counter = self._broadcast_value(patience_counter, src=0)

                effective_batch_idx += 1
                    
            if self.is_main_process:
                progress_bar.close()
            
            if patience_counter >= self.args.patience:
                break
        
        # Shutdown async resolver
        if use_async:
            async_resolver.shutdown()
        
        self.load_best_model()
        
        metrics = self.evaluate(self.valid_loader, prefix='valid', both_directions=True)
        metrics = self.evaluate(self.test_loader, prefix='test', both_directions=True)
        
        if self.is_main_process:
            model_to_push = self._unwrap_model()
            if self.args.skip_hub_upload:
                self._print("Skipping final hub upload (--skip_hub_upload is set)")
            else:
                model_to_push.push_to_hub(self.hf_save_path, private=True, safe_serialization=False)
        
        return self._unwrap_model()
