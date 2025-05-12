
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
This is a modified version of the DPLM model from https://github.com/bytedance/dplm/blob/main/src/byprot/models/lm/esm_dplm.py
"""

import numpy as np
import math
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union, Dict
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmSelfAttention,
    EsmLayer,
    EsmEncoder,
    EsmModel,
    EsmEmbeddings,
    EsmPooler,
    EsmContactPredictionHead,
    EsmIntermediate,
    EsmOutput,
    EsmLMHead,
    EsmForMaskedLM,
    EsmSelfOutput,
    EsmPreTrainedModel,
)
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm


MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


@dataclass
class NetConfig:
    arch_type: str = "esm"
    name: str = "esm2_t33_650M_UR50D"
    dropout: float = 0.1
    pretrain: bool = False
    pretrained_model_name_or_path: str = ""

@dataclass
class LoRAConfig:
    lora: bool = field(
        default=False
    )
    lora_rank: int = field(
        default=16
    )
    lora_dropout: float = field(
        default=0.1
    )
    lora_target_module: str = field(
        default=""
    )
    modules_to_save: str = field(
        default=""
    )

def get_net_class(arch_type):
    if arch_type == 'esm':
        return EsmForDPLM
    # TODO: dplm will support more architectures, such as Llama
    else:
        raise NotImplementedError
    
def get_net(cfg):
    if cfg.net.arch_type == 'esm':
        config = AutoConfig.from_pretrained(f'{cfg.net.name}')
        net = EsmForDPLM(config, dropout=cfg.net.dropout)
    # TODO: dplm will support more architectures, such as Llama
    else:
        raise NotImplementedError
    
    # 2-stage training (please refer to our paper for more details.)
    ## stage 1: pretrain a masked language model (MLM) from scratch
    ## stage 2: continue pretrain a diffusion language model based on the pretrained MLM
    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        is_local = os.path.isfile(pretrained_model_name_or_path)
        if is_local:
            # load your pretrained MLM from local
            state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')['state_dict']
            net.load_state_dict(state_dict, strict=True)
        else:
            # or you can load a pretrained MLM from huggingface
            ptrn_net = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
            net.load_state_dict(ptrn_net.state_dict(), strict=True)
            del ptrn_net
            
    # activate lora training if possible
    if cfg.lora.lora:
        # QKVO, MLP
        lora_target_module = cfg.lora.lora_target_module
        modules_to_save = cfg.lora.modules_to_save.split(',')

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            target_modules=lora_target_module,
            modules_to_save=modules_to_save,
            inference_mode=False, r=cfg.lora.lora_rank, lora_alpha=32, lora_dropout=cfg.lora.lora_dropout
        )
        net = get_peft_model(net, peft_config)
            
    return net

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores

def stochastic_sample_from_categorical(logits=None, temperature=1.0, noise_scale=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    logits = logits + noise_scale * gumbel_noise
    tokens, scores = sample_from_categorical(logits, temperature)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores

def top_k_top_p_filtering(logits, top_k=0, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    ori_shape = logits.shape
    logits = logits.reshape(-1, ori_shape[-1])
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    logits = logits.reshape(ori_shape) 
    return logits


class ModifiedEsmSelfAttention(EsmSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = query_layer * self.attention_head_size**-0.5
        
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise NotImplementedError

        # Mask heads if we want to
        if head_mask is not None:
            raise NotImplementedError
        
        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()
        context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=attention_mask, scale=1.0)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    

class ModifiedEsmAttention(EsmAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = ModifiedEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    
class ModifiedEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ModifiedEsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ModifiedEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        

class ModifiedEsmEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList([ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False


class ModifiedEsmModel(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = ModifiedEsmEncoder(config)

        self.pooler = EsmPooler(config) if add_pooling_layer else None

        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            # encoder_extended_attention_mask = None
            encoder_extended_attention_mask = encoder_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@register_model('mlm_esm')
class EsmForDPLM(EsmForMaskedLM):
    def __init__(self, config, dropout=0.1):
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        config.hidden_dropout_prob = dropout
        
        EsmPreTrainedModel.__init__(self, config)
        self.esm = ModifiedEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)

        self.init_weights()
        
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.x_id = tokenizer._token_to_id['X']
        
        self.contact_head = None
        self.tokenizer = tokenizer
    
    def forward(self,
                input_ids,
                attention_mask=None,
                inputs_embeds=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
            ):
        attention_mask = input_ids.ne(self.pad_id)
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        
        result = {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
        return result


@dataclass
class DPLMConfig:
    num_diffusion_timesteps: int = field(default=500)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    net: NetConfig = field(default_factory=NetConfig)
    gradient_ckpt: bool = field(default=False)
    rdm_couple: bool = field(default=False)


@register_model("dplm")
class DiffusionProteinLanguageModel(nn.Module):
    _default_cfg = DPLMConfig()

    def __init__(self, cfg, net=None):
        super().__init__()
        self._update_cfg(cfg)

        self.net = get_net(self.cfg) if net is None else net
        self.tokenizer = self.net.tokenizer

        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(
        cls, net_name, cfg_override={}, net_override={}, from_huggingface=True
    ):
        # Load DPLM model checkpoint from huggingface
        net_type = AutoConfig.from_pretrained(net_name).model_type
        net_class = get_net_class(net_type)
        net = net_class.from_pretrained(net_name, **net_override)
        return cls(cfg=cfg_override, net=net)

    def _update_cfg(self, cfg):
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        t1_eq_t2_mask = t1 == t2
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)

        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float)
        t2_mask[t1_eq_t2_mask] = (
            u < (t1[t1_eq_t2_mask] / self.cfg.num_diffusion_timesteps)[:, None]
        ) & (maskable_mask[t1_eq_t2_mask])
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id)

        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0),
        }

    def q_sample(self, x_0, t1, maskable_mask):
        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)
        x_t1 = x_t1.masked_fill(t1_mask, self.mask_id)

        return {
            "x_t": x_t1,
            "t": t1,
            "mask_mask": t1_mask,
        }

    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):
        outputs = self.net(
            input_ids=input_ids,
        )
        logits = outputs["logits"]
        if return_last_hidden_state:
            last_hidden_state = outputs["last_hidden_state"]
            return logits, last_hidden_state
        else:
            return logits

    def compute_loss(self, batch, weighting="constant"):
        target = batch["targets"]

        t1, t2 = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (2 * target.size(0),),
            device=target.device,
        ).chunk(2)

        if self.cfg.rdm_couple:
            # couple training
            # refer to Appendix G: Improved Training with Conditioning
            # and Algorithm 3 in Zheng et al., 2023 (https://arxiv.org/pdf/2302.05737)
            x_t, t, loss_mask = list(
                self.q_sample_coupled(
                    target, t1, t2, maskable_mask=self.get_non_special_sym_mask(target)
                ).values()
            )
            target = target.repeat(2, 1)
        else:
            x_t, t, loss_mask = list(
                self.q_sample(
                    target, t1, maskable_mask=self.get_non_special_sym_mask(target)
                ).values()
            )

        logits = self.forward(x_t)

        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
            "linear": (
                num_timesteps - (t - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t),
        }[weighting][:, None].float() / num_timesteps

        return logits, target, loss_mask, weight

    def forward_encoder(self, batch, **kwargs):
        return {}

    def initialize_output_tokens(self, batch, partial_masks=None, **kwargs):
        tokens = batch["input_ids"]
        if tokens is None:
            raise NotImplementedError
        else:
            output_mask = self.get_non_special_sym_mask(
                tokens, partial_masks=partial_masks
            )

            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores

    def resample(self, _tokens, _scores, ratio, scale):
        """
        Rejection sampling for eliminating the unexpected repeat patterns in generation results, e.g., GGGGG....
        We first calculate the frequency of all tokens,
        and for the tokens that have a frequency higher than the threshold (length * ratio),
        we mask them and resample conditioning on the remaining tokens.

        For example, the generation result is MLKNVVVVVVVVVVLDN,
        we mask the 'V' tokens to get MLKN<mask><mask><mask><mask><mask><mask><mask><mask><mask><mask>LDN,
        and resample to get MLKNVTKYYGEVKALDN.
        """
        to_be_resample_idx = []
        resample_input = []
        resample_input_mask = []
        resample_input_scores = []

        # Calculate the frequency of all tokens
        for i, seq in enumerate(_tokens):
            most_token_dict = {}
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * ratio:
                # For all tokens with a frequency higher than the threshold, transform them to mask token.
                to_be_resample_idx.append(i)
                resample_input_scores.append(_scores[i])
                mask = torch.zeros_like(seq).bool()
                for k, v in most_token_dict.items():
                    if len(v) > len(seq) * ratio:
                        mask |= seq.eq(k)
                resample_input_mask.append(mask)
                resample_input.append(seq.masked_fill(mask, self.mask_id))

        if len(to_be_resample_idx) > 0:
            # Resample the sequences that have tokens with higher frequency than threthold.
            resample_input = torch.stack(resample_input, dim=0).type_as(_tokens)
            resample_input_scores = torch.stack(resample_input_scores, dim=0).type_as(
                _scores
            )
            resample_input_mask = (
                torch.stack(resample_input_mask, dim=0).type_as(_tokens).bool()
            )
            resample_logits = self.net(
                input_ids=resample_input,
            )["logits"]
            if resample_logits.dtype != _scores.dtype:
                resample_logits = resample_logits.type_as(_scores)
            resample_logits[..., self.mask_id] = -math.inf
            resample_logits[..., self.x_id] = -math.inf
            resample_logits[..., self.pad_id] = -math.inf
            resample_logits[..., self.bos_id] = -math.inf
            resample_logits[..., self.eos_id] = -math.inf

            resample_logits = top_k_top_p_filtering(resample_logits, top_p=0.95)
            noise_scale = scale
            assert resample_logits.size(0) == len(to_be_resample_idx)
            resample_tokens, resample_scores = stochastic_sample_from_categorical(
                resample_logits, temperature=0.0, noise_scale=noise_scale
            )
            resample_input.masked_scatter_(
                resample_input_mask, resample_tokens[resample_input_mask]
            )
            resample_input_scores.masked_scatter_(
                resample_input_mask, resample_scores[resample_input_mask]
            )
            _tokens[to_be_resample_idx], _scores[to_be_resample_idx] = (
                resample_input,
                resample_input_scores,
            )

    def forward_decoder(
        self,
        prev_decoder_out,
        encoder_out=None,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="gumbel_argmax",
        disable_resample=False,
        resample_ratio=0.25,
    ):
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        step, max_step = prev_decoder_out["step"], prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        output_masks = self.get_non_special_sym_mask(
            output_tokens, partial_masks=partial_masks
        )

        net_out = self.net(
            input_ids=output_tokens,
        )

        logits = net_out["logits"]
        attentions = net_out["attentions"] if need_attn_weights else None

        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        logits[..., self.mask_id] = -math.inf
        logits[..., self.x_id] = -math.inf
        logits[..., self.pad_id] = -math.inf
        logits[..., self.bos_id] = -math.inf
        logits[..., self.eos_id] = -math.inf

        # logits = top_k_top_p_filtering(logits, top_p=0.95)

        if sampling_strategy == "vanilla":
            _tokens, _scores = sample_from_categorical(logits, temperature=temperature)
        elif sampling_strategy == "argmax":
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == "gumbel_argmax":
            noise_scale = 1.0
            _tokens, _scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )

            if not disable_resample:
                self.resample(_tokens, _scores, ratio=resample_ratio, scale=1.0)
        else:
            raise NotImplementedError

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,  # [B, L, H, T, T]
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=net_out["last_hidden_state"],
        )

    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.bos_id)
            & output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= ~partial_masks
        return non_special_sym_mask

    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        non_special_sym_mask,
        t,
        max_step,
        noise,
    ):
        """
        This function is used to perform reparameterized decoding.
        """
        # output_tokens: [B, N]
        # output_scores: [B, N]
        # cur_tokens: [B, N]
        # cur_scores: [B, N]
        # xt_neq_x0: equivalent to not_b_t [B, N]
        # non_special_sym_mask: [B, N]
        # noise: either [B, N] or scalar (if using the mask noise)

        # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        # first set the denoising rate according to the schedule
        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # compute the cutoff length for denoising top-k positions
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
        ).long()
        # set the scores of special symbols to a large value so that they will never be selected
        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)

        # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(
                _scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate
            )
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError

        # Various choices to generate v_t := [v1_t, v2_t].
        # Note that
        #   v1_t governs the outcomes of tokens where b_t = 1,
        #   v2_t governs the outcomes of tokens where b_t = 0.

        # #### the `uncond` mode ####
        # In our reparameterized decoding,
        # both v1_t and v2_t can be fully determined by the current token scores .

        # #### the `cond` mode ####
        # However, we can also impose some conditional constraints on v1_t so that
        # the decoding can be performed in a more conservative manner.
        # For example, we can set v1_t = 0 only when
        # (the newly output tokens are the same as previous denoised results, AND
        # the current token score becomes lower, AND
        # the current token score is not in the top-k share among all tokens).
        if condition == "cond":
            not_v1_t = (
                (cur_tokens == output_tokens)
                & (cur_scores < output_scores)
                & lowest_k_mask
            )
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError

        # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask

        last_mask_position = xt_neq_x0
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
        # b_{t} = (b_{t+1} & u_t) | v_t
        # For convenience, save the NOT of b_t for the next iteration
        # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
        #
        # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        assert (new_xt_neq_x0 == not_v2_t).all()
        return new_xt_neq_x0, output_tokens, output_scores

    def generate(
        self,
        batch,
        tokenizer=None,
        max_iter=None,
        temperature=None,
        partial_masks=None,
        sampling_strategy="gumbel_argmax",
        disable_resample=False,
        resample_ratio=0.25,
    ):
        tokenizer = tokenizer
        max_iter = max_iter
        temperature = temperature

        # 0) encoding
        encoder_out = self.forward_encoder(batch)
        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
            batch, encoder_out=encoder_out, partial_masks=partial_masks
        )
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )

        prev_decoder_out["output_masks"] = self.get_non_special_sym_mask(
            prev_decoder_out["output_tokens"], partial_masks=partial_masks
        )

        #for step in tqdm(range(max_iter), desc="Decoding"):
        for step in range(max_iter):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    encoder_out=encoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    disable_resample=disable_resample,
                    resample_ratio=resample_ratio,
                )

            output_tokens = decoder_out["output_tokens"]
            output_scores = decoder_out["output_scores"]

            # 2.2: re-mask skeptical parts of low confidence
            non_special_sym_mask = self.get_non_special_sym_mask(
                prev_decoder_out["output_tokens"], partial_masks=partial_masks
            )

            output_masks, result_tokens, result_scores = self._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy="reparam-uncond-deterministic-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                noise=self.mask_id,
            )

            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out["history"],
            )

        decoder_out = prev_decoder_out
        return decoder_out["output_tokens"], decoder_out["output_scores"]
