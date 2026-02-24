"""
MuonClip - Combining Muon Optimizer with QK-Clip for Attention Stability

This module combines the Muon optimizer with QK-Clip to provide a unified
optimizer that handles both general parameter updates and attention layer stability.

Adapted from:
- https://github.com/nil0x9/flash-muon/blob/main/flash_muon/muon.py
- https://kellerjordan.github.io/posts/muon/
- Kimi K2 paper (QK-Clip)
"""

import torch
from torch import Tensor
from typing import List, Dict, Optional
from muon import Muon
from qk_clip import QKClip


class MuonClip(Muon):
    """
    MuonClip - Combined optimizer that applies both Muon optimization and QK-Clip.
    
    This optimizer inherits from Muon and extends it with QK-Clip functionality:
    1. Muon: MomentUm Orthogonalized by Newton-schulz for general parameters
    2. QK-Clip: Attention weight clipping based on maximum logits
    
    Arguments:
        params: Parameters to be optimized with Muon
        attention_params: List of attention layer modules for QK-Clip
        mode: Either "mha" (Multi-Head Attention) or "mla" (Multi-Head Latent Attention)
        metadata: Dictionary mapping attention weight names to their actual attribute names
                 - For MHA: {'w_q': 'weight_q', 'w_k': 'weight_k'}
                 - For MLA: {'w_qc': 'weight_qc', 'w_kc': 'weight_kc', 'w_qr': 'weight_qr'}
        n_head: Number of attention heads
        lr: Learning rate for Muon (default: 0.02)
        weight_decay: Weight decay for Muon (default: 0.01)
        momentum: Momentum for Muon (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        tau: QK-Clip threshold for maximum logit (default: 100.0)
        alpha: QK-Clip balancing factor (default: 0.5)
        rank: GPU rank for distributed training (required)
        world_size: Number of GPUs for distributed training (required)
    
    Usage Example:
        ```python
        # Setup
        muonclip = MuonClip(
            params=model.parameters(),
            attention_params=[model.layer1.attn, model.layer2.attn],
            mode='mha',
            metadata={'w_q': 'weight_q', 'w_k': 'weight_k'},
            n_head=8,
            rank=rank,
            world_size=world_size
        )
        
        # Training loop
        for batch in dataloader:
            # Forward pass (model should track and return s_max)
            loss, s_max = model(batch)
            
            # Backward pass
            loss.backward()
            
            # Optimization step (applies both Muon and QK-Clip)
            muonclip.step(s_max)
            muonclip.zero_grad()
        ```
    """
    def __init__(
        self,
        params,
        attention_params: List[torch.nn.Module],
        mode: str,
        metadata: Dict[str, str],
        n_head: int,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        tau: float = 100.0,
        alpha: float = 0.5,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        # Initialize Muon parent class
        super().__init__(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            rank=rank,
            world_size=world_size
        )
        
        # Initialize QK-Clip component
        self.qk_clip = QKClip(
            attention_params=attention_params,
            mode=mode,
            metadata=metadata,
            n_head=n_head,
            tau=tau,
            alpha=alpha,
            rank=rank,
            world_size=world_size
        )
    
    @torch.no_grad()
    def step(self, s_max: List[List[Tensor]]):
        """
        Perform a single optimization step: Muon update followed by QK-Clip.
        
        Arguments:
            s_max: List of lists of maximum logits per head for each attention layer.
                   Format: [[head1, head2, ...] for layer1, [head1, head2, ...] for layer2, ...]
                   Each head value is S_h^max = (1/√d) * max_{X∈B} max_{i,j} Q_h^i K_h^T_j
        """
        # Apply Muon update (from parent class)
        super().step()
        
        # Apply QK-Clip to attention weights
        self.qk_clip.step(s_max)
