import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Any, Tuple, List
from transformers import PreTrainedTokenizer


class GenerateMixin:
    def __init__(self):
        self.special_token_ids = torch.empty(0)
        self.special_tokens = []
    """
    A mixin class that provides text generation functionality for models.
    
    This class implements two main generation methods:
    1. mask_diffusion_generate: A diffusion-based approach for generating text from masked tokens
    2. auto_regressive_generate: A traditional autoregressive generation approach
    
    Models that inherit from this mixin should implement the _get_logits method to 
    compute logits from input tokens and any additional context.
    """
    def _get_special_token_ids(self, extra_tokens: Optional[List[str]] = None):
        # Do not include the mask token
        mask_token = self.tokenizer.mask_token
        special_token_ids = [self.tokenizer.convert_tokens_to_ids(v) for k, v in self.tokenizer.special_tokens_map.items() if v != mask_token]
        if extra_tokens is not None:
            special_token_ids.extend([self.tokenizer.convert_tokens_to_ids(v) for v in extra_tokens])
        special_token_ids = list(set(special_token_ids))
        self.special_tokens = [self.tokenizer.decode(v) for v in special_token_ids]
        # Don't convert to tensor here - will be done in mask_diffusion_generate with proper device
        return special_token_ids

    def _get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("get_logits must be implemented")

    def _decode_seq(self, ids: torch.Tensor) -> str:
        # (1, L)
        return self.tokenizer.decode(ids).replace(' ', '').replace('<mask>', '-').replace('<pad>', '')

    def decode_output(self, ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> List[str]:
        # (B, L)
        final_seqs = []
        if attention_mask is None:
            attention_mask = torch.ones_like(ids)
        
        for id, mask in zip(ids, attention_mask):
            decoded = self.tokenizer.decode(id[mask.bool()]).replace(' ', '') # remove spaces
            decoded = decoded.replace('<mask>', '-')
            for special_token in self.special_tokens:
                decoded = decoded.replace(special_token, '')
            final_seqs.append(decoded)
        return final_seqs

    def decode_dual_input(self, ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, seperator: Optional[str] = '<eos>') -> Tuple[List[str], List[str]]:
        # (B, L)
        final_seqs_a, final_seqs_b = [], []
        if attention_mask is None:
            attention_mask = torch.ones_like(ids)

        for id, mask in zip(ids, attention_mask):
            decoded = self.tokenizer.decode(id[mask.bool()]).replace(' ', '') # remove spaces
            decoded = decoded.replace('<mask>', '-').replace('<pad>', '')
            seq_a, seq_b = decoded.split(seperator)[:2] # remove final eos and split by seperator
            final_seqs_a.append(seq_a)
            final_seqs_b.append(seq_b)
        return final_seqs_a, final_seqs_b    

    def _add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        '''
        The Gumbel max is a method for sampling categorical distributions.
        According to the papers, low-precision Gumbel Max might affect generation quality.
        Thus, we use float64.
        '''
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def _mask_sampling(self, logits: torch.Tensor, temperature: float, remasking: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)
        
        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
        if remasking == 'low_confidence':
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = p.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # b, l
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)
        elif remasking == 'low_logit':
            x0_p = logits.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # b, l
        elif remasking == 'dual':
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p_1 = p.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # b, l
            x0_p_2 = torch.rand((x0.shape[0], x0.shape[1]), device=device)
            x0_p = (x0_p_1 * x0_p_2) / 2 
        else:
            raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
        return x0, x0_p

    def _get_num_transfer_tokens(self, mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
        return num_transfer_tokens
        
    def mask_diffusion_generate(
        self,
        tokenizer: PreTrainedTokenizer,
        vocab_size: Optional[int] = None,
        extra_tokens: Optional[List[str]] = None,
        input_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        step_divisor: Optional[int] = 10,
        temperature: Optional[float] = 0.7,
        remasking: Optional[str] = "random",
        preview: Optional[bool] = False,
        slow: Optional[bool] = False,
        return_trajectory: Optional[bool] = False,
        **kwargs: Any
    ) -> torch.LongTensor:
        """
        Mask diffusion generation that combines all aspects of different generation methods.
        
        Args:
            input_tokens: Optional tensor with input tokens for conditional generation
            attention_mask: Optional attention mask for input tokens
            block_wise: Whether to use block-wise generation or global sampling
            block_length: Length of blocks if using block-wise generation
            steps: Number of diffusion steps
            temperature: Sampling temperature
            remasking: Remasking strategy ('low_confidence', 'random', 'low_logit', or 'dual')
            preview: Whether to show generation progress
            slow: Whether to slow down generation for visualization
            
        Returns:
            Generated token IDs
        """
        device = next(self.parameters()).device

        if vocab_size is None:
            vocab_size = tokenizer.vocab_size

        mask_token_id = tokenizer.mask_token_id
        self.special_token_ids = self._get_special_token_ids(extra_tokens)
        self.special_token_ids = torch.tensor(self.special_token_ids, device=device).flatten()

        num_mask_tokens = (input_tokens == mask_token_id).sum().item()
        steps = max(1, num_mask_tokens // step_divisor)

        trajectory = []
        total_length = input_tokens.shape[1]
        x = input_tokens
        if attention_mask is None:
            attention_mask = torch.ones_like(input_tokens)

        special_mask = torch.isin(x, self.special_token_ids)

        def __step(
                x: torch.Tensor,
                num_transfer_tokens: torch.Tensor,
                step_idx: int,
                temperature: float,
                remasking: str,
                **kwargs: Any
            ):
            mask_index = (x == mask_token_id)
            
            logits = self._get_logits(input_ids=x, attention_mask=attention_mask)
            x0, x0_p = self._mask_sampling(logits, temperature, remasking, device)
            
            # Don't consider special tokens
            x0_p[special_mask] = -np.inf
            
            # Prepare tokens to transfer
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
    
            # Select tokens to transfer
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, step_idx])
                transfer_index[j, select_index] = True

            # Update selected tokens
            x[transfer_index] = x0[transfer_index]
            
            # Don't remask tokens that were just transferred - only keep existing mask tokens
            # and non-transferred positions as masks
            new_mask_positions = mask_index & ~transfer_index
            # Keep special tokens unchanged
            new_mask_positions[special_mask] = False
            x = torch.where(new_mask_positions, mask_token_id, x)
            
            decoded_tokens = self._decode_seq(x[0])
            if preview:
                print(f'\r{decoded_tokens}', end='', flush=True)
                if slow:
                    import time
                    time.sleep(0.2)
            trajectory.append(decoded_tokens)
            return x

        mask_index = (x == mask_token_id)
        num_transfer_tokens = self._get_num_transfer_tokens(mask_index, steps)

        for i in range(steps):
            x = __step(x, num_transfer_tokens, i, temperature, remasking)

        # Check if any mask tokens remain
        mask_index = (x == mask_token_id)
        if mask_index.any():
            # Final step to fill all remaining masks
            logits = self._get_logits(input_ids=x, attention_mask=attention_mask)
            x0, _ = self._mask_sampling(logits, temperature, remasking, device)
            x = torch.where(mask_index, x0, x)

        final_seq = self._decode_seq(x[0])
        if preview:
            print('\nFinal sequence:')
            print('=' * total_length)
            print(final_seq)
            print('=' * total_length)
        trajectory.append(final_seq)
        if return_trajectory:
            return x, trajectory
        else:
            return x
