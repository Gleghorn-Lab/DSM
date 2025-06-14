import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Union, List
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from .FastPLMs.modeling_fastesm import FastEsmModel, FastEsmForMaskedLM, FastEsmConfig
from .generate_mixin import GenerateMixin
from .modeling_transformer import Transformer
from .utils import wrap_lora


class DSMConfig(FastEsmConfig):
    model_type = "dsm"
    def __init__(
        self,
        num_at_layers: int = 1,
        at_vocab_size: int = 30000,
        lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_at_layers = num_at_layers
        self.at_vocab_size = at_vocab_size
        self.lora = lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


@dataclass
class DSMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    last_hidden_state: Optional[torch.Tensor] = None
    t: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, soft_logit_cap: float = 30.0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.soft_logit_cap = soft_logit_cap
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.act(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)


class DSM(FastEsmModel, GenerateMixin): # FastEsmModel already inherits EmbeddingMixin
    config_class = DSMConfig
    def __init__(self, config: DSMConfig, **kwargs):
        FastEsmModel.__init__(self, config, **kwargs)
        GenerateMixin.__init__(self, self.tokenizer)
        self.config = config
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config.hidden_size, config.vocab_size)
        # tie to word embeddings
        self.lm_head.decoder.weight = self.esm.embeddings.word_embeddings.weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.special_token_ids = self.get_special_token_ids()

    def get_special_token_ids(self, extra_tokens: Optional[List[str]] = None):
        # Do not include the mask token
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask_token = self.tokenizer.mask_token
        self.special_token_ids = [self.tokenizer.convert_tokens_to_ids(v) for k, v in self.tokenizer.special_tokens_map.items() if v != mask_token]
        if extra_tokens is not None:
            self.special_token_ids.extend([self.tokenizer.convert_tokens_to_ids(v) for v in extra_tokens])

        self.special_token_ids = torch.tensor(self.special_token_ids, device=device).flatten()
        return self.special_token_ids

    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        with torch.no_grad():
            x = self.esm.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state # (b, L, d)
            logits = self.lm_head(x) # (b, L, v)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> DSMOutput:
        eps = 1e-3
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        if self.training: # sample uniform between 0 and 1
            t = torch.rand(batch_size, device=device)
            t = (1 - eps) * t + eps
        else: # evaluate at classic 15%
            t = torch.full((batch_size,), 0.15, device=device)
        
        p_mask = t[:, None].repeat(1, seq_len)
        mask_indices = torch.rand(batch_size, seq_len, device=device) < p_mask
        # prevent special tokens from being masked (cls, sep, eos, etc.)
        special_mask = torch.isin(input_ids, self.special_token_ids)
        mask_indices = mask_indices & ~special_mask & attention_mask.bool()

        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        labels = input_ids.clone()
        non_mask_indices = ~mask_indices | (attention_mask == 0)
        labels[non_mask_indices] = -100

        x = super().forward(
            input_ids=noisy_batch,
            attention_mask=attention_mask,
        ).last_hidden_state # (b, L, d)
        lm_logits = self.lm_head(x) # (b, L, v)

        joint_mask = mask_indices & attention_mask.bool()
        if not joint_mask.any():
            # fall back comparing the logits and input_ids directly if no tokens were masked
            joint_mask = attention_mask.bool()

        token_loss = self.ce_loss(
            lm_logits[joint_mask].view(-1, self.vocab_size),
            input_ids[joint_mask].view(-1)) / p_mask[joint_mask]

        loss = token_loss.sum() / (batch_size * seq_len)

        return DSMOutput(
            loss=loss,
            logits=(lm_logits, labels),
            last_hidden_state=x,
            hidden_states=None,
            attentions=None,
            t=t,
        )


class DSM_Binders(FastEsmModel, GenerateMixin):
    config_class = DSMConfig
    def __init__(self, config: DSMConfig, **kwargs):
        FastEsmModel.__init__(self, config, **kwargs)
        GenerateMixin.__init__(self, self.tokenizer)
        self.config = config
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config.hidden_size, config.vocab_size)
        # tie to word embeddings
        self.lm_head.decoder.weight = self.esm.embeddings.word_embeddings.weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _make_interactor_mask(eos_mask: torch.BoolTensor) -> torch.BoolTensor:
        """
        eos_mask: (b, L) - True where <eos> appears.
        Returns:  (b, L) - True for tokens strictly between the first and second
                           <eos> (the interactor), False elsewhere.
        """
        # Cumulative count of <eos> along sequence dimension:
        #   0 : in target segment
        #   1 : in interactor segment (until next <eos>)
        #  >=2: everything after second <eos>
        eos_cumsum = eos_mask.cumsum(dim=1)
        return (eos_cumsum == 1) & ~eos_mask # exclude the two <eos> tokens

    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        with torch.no_grad():
            x = self.esm.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state # (b, L, d)
            logits = self.lm_head(x) # (b, L, v)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> DSMOutput:
        """
        For DSM_Binders, there are two input sequences in each input_ids
        The first sequence is the target sequence, and the second sequence is a known interactor
        We only want to mask the interactor sequence, not the target sequence
        The format is cls, target, eos, interactor, eos
        so we can use the first eos to mask the interactor sequence
        """

        eps = 1e-3
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        eos_mask = input_ids == self.eos_token_id # (b, L) – True on every <eos>
        interactor_mask = self._make_interactor_mask(eos_mask) # (b, L)

        # valid_mask = tokens we ARE ALLOWED to corrupt
        valid_mask = interactor_mask & attention_mask.bool() # never touch target / padding

        if self.training:
            t = (1 - eps) * torch.rand(batch_size, device=device) + eps
        else:
            t = torch.full((batch_size,), 0.15, device=device)

        p_mask = t[:, None].expand(batch_size, seq_len)  # broadcast over length
        bernoulli = torch.rand(batch_size, seq_len, device=device) < p_mask
        mask_indices = bernoulli & valid_mask # (b, L)

        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        labels = input_ids.clone()
        labels[~mask_indices] = -100 # ignore non-masked tokens

        x = super().forward(
            input_ids=noisy_batch,
            attention_mask=attention_mask,
        ).last_hidden_state # (b, L, d)
        lm_logits = self.lm_head(x) # (b, L, |V|)

        joint_mask = mask_indices & attention_mask.bool() # (b, L)
        if not joint_mask.any():
            # fall back comparing the logits and input_ids directly if no tokens were masked
            joint_mask = attention_mask.bool()

        token_loss = self.ce_loss(
            lm_logits[joint_mask].view(-1, self.vocab_size),
            input_ids[joint_mask].view(-1)
        ) / p_mask[joint_mask]
        loss = token_loss.sum() / (batch_size * seq_len)

        return DSMOutput(
            loss = loss,
            logits = (lm_logits, labels),
            last_hidden_state = x,
            hidden_states = None,
            attentions = None,
            t = t,
        )


class DSM_AV(DSM):
    config_class = DSMConfig
    def __init__(self, config: DSMConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.at_embedding = nn.Embedding(config.at_vocab_size, config.hidden_size)
        self.at = Transformer(
            hidden_size=config.hidden_size,
            n_heads=config.num_attention_heads,
            expansion_ratio=2.0,
            dropout=0.1,
            rotary=True,
            causal=False,
            n_layers=config.num_at_layers
        )
    
    def apply_lora(self, r: int, lora_alpha: float, lora_dropout: float):
        self.esm = wrap_lora(self.esm, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        x_at = self.at_embedding(prompt_tokens)
        x_at = self.at(x_at, prompt_attention_mask)
        _, seq_len = input_ids.shape
        x = self.esm.embeddings.word_embeddings(input_ids)
        x_hat = torch.cat([x_at, x], dim=1)
        attention_mask_hat = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        x = self.esm.forward(
            inputs_embeds=x_hat,
            attention_mask=attention_mask_hat,
        ).last_hidden_state # (b, L, d)
        x = x[:, -seq_len:]
        logits = self.lm_head(x) # (b, L, v)
        return logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        at_ids: Optional[torch.Tensor] = None,
        at_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> DSMOutput:
        x_at = self.at_embedding(at_ids)
        x_at = self.at(x_at, at_attention_mask)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        eps = 1e-3
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        t = torch.rand(batch_size, device=device)
        t = (1 - eps) * t + eps
        p_mask = t[:, None].repeat(1, seq_len)
        mask_indices = torch.rand(batch_size, seq_len, device=device) < p_mask
        # prevent cls and eos from being masked
        cls_mask = input_ids == self.cls_token_id
        eos_mask = input_ids == self.eos_token_id
        mask_indices = mask_indices & ~cls_mask & ~eos_mask & attention_mask.bool()

        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        labels = input_ids.clone()
        non_mask_indices = ~mask_indices | (attention_mask == 0)
        labels[non_mask_indices] = -100
        x = self.esm.embeddings.word_embeddings(noisy_batch)

        x_hat = torch.cat([x_at, x], dim=1)
        attention_mask_hat = torch.cat([at_attention_mask, attention_mask], dim=1)

        x = self.esm.forward(
            inputs_embeds=x_hat,
            attention_mask=attention_mask_hat,
        ).last_hidden_state # (b, L, d)
        x = x[:, -seq_len:]
        logits = self.lm_head(x) # (b, L, v)

        joint_mask = mask_indices & attention_mask.bool()
    
        if not joint_mask.any():
            # fall back comparing the logits and input_ids directly if no tokens were masked
            joint_mask = attention_mask.bool()

        token_loss = self.ce_loss(
            logits[joint_mask].view(-1, self.vocab_size),
            input_ids[joint_mask].view(-1)) / p_mask[joint_mask]

        loss = token_loss.sum() / (batch_size * seq_len)

        return DSMOutput(
            loss=loss,
            logits=(logits, labels),
            last_hidden_state=x,
            hidden_states=None,
            attentions=None,
            t=t,
        )



class DSM_ESM2(FastEsmForMaskedLM, GenerateMixin):
    config_class = DSMConfig
    def __init__(self, config: DSMConfig, **kwargs):
        FastEsmForMaskedLM.__init__(self, config, **kwargs)
        GenerateMixin.__init__(self, self.tokenizer)
        self.config = config
        
    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        return logits


if __name__ == "__main__":
    # py -m models.modeling_dsm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test DSM Model
    print("\n=== Testing DSM Model ===")
    model = DSM.from_pretrained('Synthyra/ESM2-8M').to(device)
    model.train()  # Enable training mode for masking
    
    # Get special token IDs for testing
    cls_id = model.tokenizer.cls_token_id
    eos_id = model.tokenizer.eos_token_id
    pad_id = model.tokenizer.pad_token_id
    mask_id = model.tokenizer.mask_token_id
    
    print(f"Special tokens - CLS: {cls_id}, EOS: {eos_id}, PAD: {pad_id}, MASK: {mask_id}")
    
    # Create test input with known special tokens
    batch_size = 4
    seq_len = 64
    
    # Create input with special tokens at specific positions
    input_ids = torch.randint(4, 33, (batch_size, seq_len)).to(device)  # Avoid special tokens initially
    input_ids[:, 0] = cls_id  # Start with CLS
    input_ids[:, -1] = eos_id  # End with EOS
    input_ids[:, seq_len//2] = eos_id  # Add EOS in middle
    if pad_id is not None:
        input_ids[:, -5:-1] = pad_id  # Add some padding tokens
    
    attention_mask = torch.ones_like(input_ids).to(device)
    if pad_id is not None:
        attention_mask[input_ids == pad_id] = 0  # Mask padding tokens
    
    # Run multiple forward passes to test masking consistency
    special_token_masked_count = 0
    total_runs = 10
    
    for run in range(total_runs):
        output = model(input_ids, attention_mask)
        lm_logits, labels = output.logits
        
        # Check if any special tokens were masked (labels != -100 means token was masked)
        for special_id in model.special_token_ids:
            special_positions = (input_ids == special_id)
            masked_special = (labels != -100) & special_positions
            if masked_special.any():
                special_token_masked_count += 1
                print(f"WARNING: Special token {special_id} was masked in run {run}")
    
    print(f"DSM Special Token Masking Test: {special_token_masked_count}/{total_runs} runs had special tokens masked")
    assert special_token_masked_count == 0, "Special tokens should never be masked in DSM!"
    
    # Test that regular tokens can be masked
    regular_token_positions = ~torch.isin(input_ids, model.special_token_ids) & attention_mask.bool()
    masked_regular = (labels != -100) & regular_token_positions
    print(f"Regular tokens masked: {masked_regular.sum().item()}/{regular_token_positions.sum().item()}")
    assert masked_regular.any(), "Some regular tokens should be masked!"
    
    print("✓ DSM masking test passed!")
    
    # Test DSM_Binders Model
    print("\n=== Testing DSM_Binders Model ===")
    binder_model = DSM_Binders.from_pretrained('Synthyra/ESM2-8M').to(device)
    binder_model.train()  # Enable training mode for masking
    
    # Create test data with format: [CLS, target, EOS, interactor, EOS]
    batch_size = 3
    target_len = 8
    interactor_len = 12
    
    # Build sequences with clear structure
    target_tokens = torch.randint(4, 33, (batch_size, target_len)).to(device)
    interactor_tokens = torch.randint(4, 33, (batch_size, interactor_len)).to(device)
    
    input_ids = torch.cat([
        torch.full((batch_size, 1), cls_id, device=device),  # CLS token
        target_tokens,                                        # Target sequence
        torch.full((batch_size, 1), eos_id, device=device), # First EOS
        interactor_tokens,                                    # Interactor sequence
        torch.full((batch_size, 1), eos_id, device=device)  # Second EOS
    ], dim=1)
    
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Define expected regions
    cls_region = slice(0, 1)
    target_region = slice(1, 1 + target_len)
    first_eos_region = slice(1 + target_len, 1 + target_len + 1)
    interactor_region = slice(1 + target_len + 1, 1 + target_len + 1 + interactor_len)
    second_eos_region = slice(-1, None)
    
    # Run multiple tests to ensure consistency
    target_masked_count = 0
    interactor_masked_count = 0
    eos_masked_count = 0
    cls_masked_count = 0
    
    for run in range(total_runs):
        output = binder_model(input_ids, attention_mask)
        _, labels = output.logits
        masked_positions = (labels != -100)
        
        # Check each region
        if masked_positions[:, cls_region].any():
            cls_masked_count += 1
        if masked_positions[:, target_region].any():
            target_masked_count += 1
        if masked_positions[:, first_eos_region].any() or masked_positions[:, second_eos_region].any():
            eos_masked_count += 1
        if masked_positions[:, interactor_region].any():
            interactor_masked_count += 1
    
    print(f"DSM_Binders Masking Results over {total_runs} runs:")
    print(f"  CLS tokens masked: {cls_masked_count}/{total_runs}")
    print(f"  Target tokens masked: {target_masked_count}/{total_runs}")
    print(f"  EOS tokens masked: {eos_masked_count}/{total_runs}")
    print(f"  Interactor tokens masked: {interactor_masked_count}/{total_runs}")
    
    # Assertions
    assert cls_masked_count == 0, "CLS tokens should never be masked in DSM_Binders!"
    assert target_masked_count == 0, "Target tokens should never be masked in DSM_Binders!"
    assert eos_masked_count == 0, "EOS tokens should never be masked in DSM_Binders!"
    assert interactor_masked_count > 0, "Interactor tokens should sometimes be masked in DSM_Binders!"
    
    print("✓ DSM_Binders masking test passed!")
    
    # Test _make_interactor_mask function directly
    print("\n=== Testing _make_interactor_mask Function ===")
    test_input = torch.tensor([
        [cls_id, 5, 6, 7, eos_id, 8, 9, 10, eos_id],  # CLS, target, EOS, interactor, EOS
        [cls_id, 11, 12, eos_id, 13, 14, 15, 16, eos_id]  # Different lengths
    ]).to(device)
    
    eos_mask = (test_input == eos_id)
    interactor_mask = DSM_Binders._make_interactor_mask(eos_mask)
    
    print("Test input:")
    print(test_input)
    print("EOS mask:")
    print(eos_mask)
    print("Interactor mask (should be True only between first and second EOS):")
    print(interactor_mask)
    
    # Verify interactor mask is correct
    expected_interactor_0 = torch.tensor([False, False, False, False, False, True, True, True, False])
    expected_interactor_1 = torch.tensor([False, False, False, False, True, True, True, True, False])
    
    assert torch.equal(interactor_mask[0], expected_interactor_0), "Interactor mask incorrect for sequence 0"
    assert torch.equal(interactor_mask[1], expected_interactor_1), "Interactor mask incorrect for sequence 1"
    
    print("✓ _make_interactor_mask function test passed!")
    
    # Test DSM_ESM2 model
    print("\n=== Testing DSM_ESM2 Model ===")
    esm2_model = DSM_ESM2.from_pretrained('Synthyra/ESM2-8M').to(device)
    test_input = torch.randint(0, 33, (2, 32)).to(device)
    test_attention = torch.ones_like(test_input).to(device)
    
    logits = esm2_model._get_logits(test_input, test_attention)
    print(f"DSM_ESM2 logits shape: {logits.shape}")
    assert logits.shape == (2, 32, esm2_model.config.vocab_size), "DSM_ESM2 logits shape incorrect"
    
    print("✓ DSM_ESM2 test passed!")
    print("\n=== All Tests Passed! ===")

