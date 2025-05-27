import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Any, Tuple
from transformers import PreTrainedTokenizer


class GenerateMixin:
    """
    A mixin class that provides text generation functionality for models.
    
    This class implements two main generation methods:
    1. mask_diffusion_generate: A diffusion-based approach for generating text from masked tokens
    2. auto_regressive_generate: A traditional autoregressive generation approach
    
    Models that inherit from this mixin should implement the _get_logits method to 
    compute logits from input tokens and any additional context.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        canonical_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.mask_token_id = self.tokenizer.mask_token_id
        self.methionine_token_id = self.tokenizer.encode('M', add_special_tokens=False)[0]
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.canonical_amino_acid_ids = set(self.tokenizer.convert_tokens_to_ids(list(canonical_amino_acids)))
        self.canonical_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for aa_id in self.canonical_amino_acid_ids:
            self.canonical_mask[aa_id] = True

    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_tokens: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        raise NotImplementedError("get_logits must be implemented")

    def _decode_seq(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids).replace(' ', '').replace('<mask>', '-').replace('<cls>', '').replace('<eos>', '')

    def _is_canonical_amino_acid(self, token_id: int) -> bool:
        '''
        Check if a token ID represents a canonical amino acid.
        '''
        return token_id in self.canonical_amino_acid_ids

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
        # Prevent non-canonical amino acids from being selected
        for i in range(self.vocab_size):
            if not self._is_canonical_amino_acid(i):
                logits_with_noise[:, :, i] = -torch.inf
        
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
        template_tokens: Optional[torch.LongTensor] = None,
        prompt_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        batch_size: Optional[int] = 1,
        block_wise: Optional[bool] = False,
        block_length: Optional[int] = 32,
        length: Optional[int] = 128,
        steps: Optional[int] = 20,
        temperature: Optional[float] = 0.7,
        remasking: Optional[str] = "random",
        start_with_methionine: Optional[bool] = False,
        preview: Optional[bool] = False,
        slow: Optional[bool] = False,
    ) -> torch.LongTensor:
        """
        Mask diffusion generation that combines all aspects of different generation methods.
        
        Args:
            prompt_tokens: Optional tensor with prompt tokens for conditional generation
            attention_mask: Optional attention mask for prompt tokens
            block_wise: Whether to use block-wise generation or global sampling
            block_length: Length of blocks if using block-wise generation
            length: Length of sequence to generate
            steps: Number of diffusion steps
            temperature: Sampling temperature
            remasking: Remasking strategy ('low_confidence', 'random', 'low_logit', or 'dual')
            start_with_methionine: Whether to start protein sequences with methionine
            preview: Whether to show generation progress
            slow: Whether to slow down generation for visualization
            
        Returns:
            Generated token IDs
        """

        assert template_tokens is None or prompt_tokens is None, "Cannot provide both template and prompt tokens"

        device = next(self.parameters()).device

        if template_tokens is not None:
            batch_size = template_tokens.shape[0]
        else:
            batch_size = batch_size if prompt_tokens is None else prompt_tokens.shape[0]

        # Add 2 for CLS and EOS tokens
        if template_tokens is not None:
            total_length = template_tokens.shape[1]
            x = template_tokens
            has_prompt = False
            if attention_mask is None:
                attention_mask = torch.ones_like(template_tokens)
        else:
            total_length = length + 2
            # Initialize with all mask tokens and add CLS/EOS tokens
            x = torch.full((batch_size, total_length), self.mask_token_id, dtype=torch.long, device=device)
            x[:, 0], x[:, -1] = self.cls_token_id, self.eos_token_id
            if attention_mask is None:
                attention_mask = torch.ones_like(x)
        

        if prompt_tokens is None:
            has_prompt = False
        else:
            has_prompt = True
        
        cls_mask = (x == self.cls_token_id)
        eos_mask = (x == self.eos_token_id)

        if start_with_methionine:
            x[:, 1] = self.methionine_token_id        

        def __step(
                x: torch.Tensor,
                num_transfer_tokens: torch.Tensor,
                temperature: float,
                remasking: str,
                num_block: Optional[int] = None,
                num_blocks: Optional[int] = None,
                **kwargs: Any
            ):
            mask_index = (x == self.mask_token_id)
            
            logits = self._get_logits(
                input_ids=x,
                attention_mask=attention_mask,
                prompt_tokens=prompt_tokens,
                prompt_attention_mask=torch.ones_like(prompt_tokens) if has_prompt else None,
            )
            
            x0, x0_p = self._mask_sampling(logits, temperature, remasking, device)
            
            # Don't consider tokens outside current block
            if num_block is not None:
                if num_block < num_blocks - 1:
                    x0_p[:, block_end:] = -np.inf
            
            # Don't consider CLS and EOS 
            x0_p[cls_mask] = -np.inf
            x0_p[eos_mask] = -np.inf
            
            # If start_with_methionine is True, don't consider the first token after CLS
            if start_with_methionine:
                x0_p[:, 1] = -np.inf
            
            # Prepare tokens to transfer
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # Select tokens to transfer
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                
            # Update selected tokens
            x[transfer_index] = x0[transfer_index]
            
            # Remask any tokens that are not canonical amino acids
            non_mask_tokens = (x != self.mask_token_id)
            canonical_check = self.canonical_mask.to(device)[x]
            to_remask = non_mask_tokens & ~canonical_check
            # Don't consider CLS and EOS 
            to_remask[cls_mask] = False
            to_remask[eos_mask] = False
            x[to_remask] = self.mask_token_id
            
            if preview:
                decoded_tokens = self._decode_seq(x[0])
                print(f'\r{decoded_tokens}', end='', flush=True)
                if slow:
                    import time
                    time.sleep(0.2)
            return x

        if block_wise:
            # Block-wise generation
            assert length % block_length == 0, "Length must be divisible by block_length"
            num_blocks = length // block_length
            
            assert steps % num_blocks == 0, "Steps must be divisible by number of blocks"
            steps_per_block = steps // num_blocks
            
            for num_block in range(num_blocks):
                # Adjust block indices to account for CLS token
                block_start = num_block * block_length + 1  # +1 for CLS token
                block_end = (num_block + 1) * block_length + 1  # +1 for CLS token
                
                block_mask_index = (x[:, block_start:block_end] == self.mask_token_id)
                num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps_per_block)
                
                for i in range(steps_per_block):
                    x = __step(x, num_transfer_tokens, temperature, remasking, num_block, num_blocks)
        else:
            # Global sampling approach
            mask_index = (x == self.mask_token_id)
            num_transfer_tokens = self._get_num_transfer_tokens(mask_index, steps)
            
            for i in range(steps):
                x = __step(x, num_transfer_tokens, temperature, remasking)
        
        # Final step: convert any special tokens (except CLS and EOS) to masks and fill them all
        # Necessary for when steps is not divisible by the number of tokens in the sequence
        special_tokens = ~self.canonical_mask.to(device)
        special_tokens[self.cls_token_id] = False  # Don't mask CLS
        special_tokens[self.eos_token_id] = False  # Don't mask EOS
        
        # Find remaining special tokens in the sequence
        remaining_special = special_tokens[x]
        
        # Convert special tokens to mask tokens
        x[remaining_special] = self.mask_token_id
        
        # Check if any mask tokens remain
        mask_index = (x == self.mask_token_id)
        if mask_index.any():
            
            # Final step to fill all remaining masks
            logits = self._get_logits(
                input_ids=x,
                attention_mask=attention_mask,
                prompt_tokens=prompt_tokens,
                prompt_attention_mask=torch.ones_like(prompt_tokens) if has_prompt else None,
            )
            
            x0, _ = self._mask_sampling(logits, temperature, remasking, device)
            x = torch.where(mask_index, x0, x)
        
        if preview:
            print('\nFinal sequence:')
            print('=' * length)
            print(self._decode_seq(x[0]))
            print('=' * length)
        return x

    def auto_regressive_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        preview: bool = False,
    ) -> torch.LongTensor:
        """Autoregressive sampling for causal language modeling."""
        batch_size = input_ids.shape[0]
        pad_token_id = pad_token_id or self.tokenizer.pad_token_id
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id
        
        # Initialize sequence buffer and attention mask
        seq_length = input_ids.shape[1]
        position = seq_length
        
        # Preallocate output tensor
        output_ids = torch.full(
            (batch_size, max_length), 
            pad_token_id, 
            dtype=torch.long, 
            device=input_ids.device
        )
        output_ids[:, :seq_length] = input_ids
        
        # Create expanded attention mask
        expanded_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, max_length - seq_length, device=input_ids.device)
        ], dim=1)
        
        # List to track which sequences have completed
        active_sequences = torch.ones(batch_size, device=input_ids.device, dtype=torch.bool)
        
        while position < max_length and active_sequences.any():
            # Forward pass to get logits for next token
            logits = self.get_logits(
                input_ids=output_ids[:, :position],
                attention_mask=expanded_mask[:, :position],
            )
            
            # Get next token logits (last position)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(output_ids[i, :position].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, 
                    index=sorted_indices, 
                    src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Set next tokens
            output_ids[:, position] = next_tokens
            
            # Preview current generation if requested
            if preview:
                # Only show the newly generated tokens (after the prompt)
                prompt_len = input_ids.shape[1]
                if position >= prompt_len:
                    current_output = output_ids[0, prompt_len:position+1]
                    decoded = self.preview_decode(current_output)
                    print(f"\rGenerated: {decoded}", end="", flush=True)
            
            # Update active sequences (if a sequence has hit EOS, it becomes inactive)
            if eos_token_id is not None:
                active_sequences = active_sequences & (next_tokens != eos_token_id)
                
            position += 1
        
        # Final preview of completed sequences
        if preview:
            # Only show the generated part (after the prompt)
            prompt_len = input_ids.shape[1]
            if eos_token_id is not None:
                eos_positions = (output_ids[0, prompt_len:] == eos_token_id).nonzero()
                end_pos = prompt_len + eos_positions[0].item() + 1 if len(eos_positions) > 0 else position
            else:
                end_pos = position
            
            final_output = output_ids[0, prompt_len:end_pos]
            decoded = self.preview_decode(final_output)
            print(f"\nFinal: {decoded}")
        
        return output_ids
