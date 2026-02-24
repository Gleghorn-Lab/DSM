import torch
import torch.distributed as dist
from torch import Tensor
from typing import List, Dict, Optional

class QKClip:
    """
    QK-Clip for Attention Modules. It's used to mitigate instability of Muon optimizer.
    Arguments:
        attention_params: List of attention layer modules to be clipped
        mode: Either "mha" (Multi-Head Attention) or "mla" (Multi-Head Latent Attention)
        metadata: Dictionary mapping attention weight names to their actual attribute names
                 - For MHA: {
                     'w_q': 'your_naming_for_weight_for_q', 
                     'w_k': 'your_naming_for_weight_for_k', 
                 }
                 - For MLA: {
                     'w_qc': 'your_naming_for_weight_for_qc', 
                     'w_kc': 'your_naming_for_weight_for_kc', 
                     'w_qr': 'your_naming_for_weight_for_qr'
                 }
             ex) If you use MHA and you have self.weight_q = nn.Linear(d_model, d_model), 
                 then metadata should be {'w_q': 'weight_q'}.
                 
        n_head: Number of attention heads
        tau: Threshold for maximum logit (default: 100.0)
        alpha: balancing factor (default: 0.5)
        
    Distributed Training:
        - For single GPU: rank=None, world_size=None (or rank=0, world_size=1)
        - For multi-GPU: Must provide rank and world_size matching your DDP setup
        - IMPORTANT: Call QKClip.step() AFTER Muon.step() (weights already synchronized by Muon)
    """
    def __init__(
        self, 
        attention_params: List[torch.nn.Module] ,
        mode: str, # 'mha' or 'mla'
        metadata: Dict[str, str],
        n_head: int,
        tau: float = 100.0,
        alpha: float = 0.5,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        assert mode.lower() in ["mha", "mla"], f"mode must be either 'mha' or 'mla', got {mode}"
        
        self.attention_params = attention_params
        self.mode = mode
        self.metadata = metadata
        self.n_head = n_head
        self.tau = tau
        self.alpha = alpha
        self.rank = rank
        self.world_size = world_size
        self.is_multigpu = (self.world_size is not None and self.world_size > 1 and dist.is_initialized())
            
    @torch.no_grad()
    def step(self, s_max: List[List[Tensor]]):
        """
        Apply QK-Clip to attention weights based on the magnitude of the maximum logit (s_max) in each head.
                
        S_h^max = (1/√d) * max_{X∈B} max_{i,j} Q_h^i K_h^T_j (from paper Kimi K2)
        In multi-gpu, simply using the fact that batch size = effective batch size -> we find global max across all gpus. (Only DDP.)
        
        
        where:
            - h is the head index
            - d is the dimension per head
            - Q_h^i is the i-th query vector for head h
            - K_h^j is the j-th key vector for head h
            - The outer max is over all samples X in batch B
            - The inner max is over all query-key pairs (i,j)
        
        Arguments:
            s_max: A list of lists of tensors. Each inner list contains per-head max logits for one attention layer.
                   In distributed training, pass your local s_max - it will be automatically
                   synchronized via all_reduce(MAX) to get the global maximum.
            
            ex): 
                s_max = [
                    [torch.tensor(57.3), torch.tensor(100.2), torch.tensor(13.4), torch.tensor(3.1)],
                    [torch.tensor(12.5), torch.tensor(41.3), torch.tensor(100.41), torch.tensor(0.5)],
                    [torch.tensor(91.3), torch.tensor(11.2), torch.tensor(103), torch.tensor(0.1)]
                ]
                
            ex): when there is 3 attention layers and 4 heads per layer.
            Each value is S_h^max as a scalar tensor.
        """
        assert len(s_max) == len(self.attention_params), f"Length of s_max ({len(s_max)}) must match number of attention layers ({len(self.attention_params)})"
        
        if self.is_multigpu:
            # Let's say we have 2 gpus. If rank=0's head has s_max > tau, and rank=1's head has s_max < tau, 
            # then we will use rank=0's s_max for QKClip.
            # Stack all s_max tensors for efficient all_reduce & unstack back to list of lists
            s_max_stacked = torch.stack([torch.stack(layer_s_max) for layer_s_max in s_max])
            dist.all_reduce(s_max_stacked, op=dist.ReduceOp.MAX)
            s_max = [[s_max_stacked[i, j] for j in range(s_max_stacked.size(1))] for i in range(s_max_stacked.size(0))]
        
        for layer_idx, (layer, layer_s_max) in enumerate(zip(self.attention_params, s_max)):
            assert len(layer_s_max) == self.n_head, f"Layer {layer_idx}: expected {self.n_head} heads, got {len(layer_s_max)} values in s_max"
            for head_idx, s_h_max in enumerate(layer_s_max):
                # No if statement (unlike paper) to avoid conditional branching on multi-gpu setup. (and possible torch.compile optimization & efficiency-related benchmark)
                gamma = self.tau / (s_h_max.item() + 1e-6)
                gamma_clamped = min(gamma, 1.0)
                sqrt_gamma_clamped = gamma_clamped ** self.alpha
                
                # Always apply scaling (no-op when gamma_clamped == 1.0)
                if self.mode == "mla":
                    # MLA mode: scale W_qc, W_kc, W_qr
                    self._scale_weight(layer, self.metadata['w_qc'], head_idx, sqrt_gamma_clamped)
                    self._scale_weight(layer, self.metadata['w_kc'], head_idx, sqrt_gamma_clamped)
                    self._scale_weight(layer, self.metadata['w_qr'], head_idx, gamma_clamped)
                else:  # mha
                    # MHA mode: scale W_q, W_k
                    self._scale_weight(layer, self.metadata['w_q'], head_idx, sqrt_gamma_clamped)
                    self._scale_weight(layer, self.metadata['w_k'], head_idx, sqrt_gamma_clamped)
    
    def _scale_weight(self, layer, weight_name: str, head_idx: int, scale: float):
        """
        Scale the weight for a specific head.
        
        Arguments:
            layer: The attention layer module
            weight_name: The attribute name of the weight (e.g., 'weight_q', 'weight_qc')
            head_idx: The index of the head to scale
            scale: The scaling factor
        """
        assert hasattr(layer, weight_name), f"Layer {layer} does not have attribute {weight_name}"
        weight = getattr(layer, weight_name)
        assert isinstance(weight, torch.nn.Parameter), f"Weight attribute {weight_name} is not a torch.nn.Parameter"
        
        head_dim = weight.size(0) // self.n_head
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim
        
        weight[start_idx:end_idx].mul_(scale)
    
