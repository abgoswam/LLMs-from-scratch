# GPTFS ("GPT from scratch") model for Hugging Face transformers.
#
# Same architecture and submodule names as GPTModel in custom/modeling_gptc.py,
# wrapped in PreTrainedModel so it supports save_pretrained / from_pretrained
# (via trust_remote_code) and model.generate().
#
# This file is intentionally self-contained (no imports from ../) because
# save_pretrained copies it next to the weights, and it must load from there.
#
# Differences from GPTModel:
#   - the causal mask is built on the fly instead of stored as a buffer
#   - out_head is called lm_head, and the rest lives under `model.`
#   - no KV cache: every generation step re-runs the full sequence

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from .configuration_gptfs import GPTFSConfig


class GPTFSMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_out = config.emb_dim
        self.num_heads = config.n_heads
        self.head_dim = self.d_out // self.num_heads

        self.W_query = nn.Linear(config.emb_dim, self.d_out, bias=config.qkv_bias)
        self.W_key = nn.Linear(config.emb_dim, self.d_out, bias=config.qkv_bias)
        self.W_value = nn.Linear(config.emb_dim, self.d_out, bias=config.qkv_bias)
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x):
        b, num_tokens, _ = x.shape

        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = torch.triu(
            torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=x.device), diagonal=1)
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class GPTFSLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GPTFSGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class GPTFSFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            GPTFSGELU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class GPTFSBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = GPTFSMultiHeadAttention(config)
        self.ff = GPTFSFeedForward(config)
        self.norm1 = GPTFSLayerNorm(config.emb_dim)
        self.norm2 = GPTFSLayerNorm(config.emb_dim)
        self.drop_resid = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = x + self.drop_resid(self.att(self.norm1(x)))
        x = x + self.drop_resid(self.ff(self.norm2(x)))
        return x


class GPTFSPreTrainedModel(PreTrainedModel):
    config: GPTFSConfig
    config_class = GPTFSConfig
    base_model_prefix = "model"
    _no_split_modules = ["GPTFSBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GPTFSLayerNorm):
            init.ones_(module.scale)
            init.zeros_(module.shift)


class GPTFSModel(GPTFSPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(*[GPTFSBlock(config) for _ in range(config.n_layers)])
        self.final_norm = GPTFSLayerNorm(config.emb_dim)
        self.post_init()

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, value):
        self.tok_emb = value

    def forward(self, input_ids, **kwargs):
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.drop_emb(self.tok_emb(input_ids) + self.pos_emb(pos))
        x = self.trf_blocks(x)
        return BaseModelOutput(last_hidden_state=self.final_norm(x))


class GPTFSForCausalLM(GPTFSPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = GPTFSModel(config)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.tok_emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids[:, -self.config.context_length:]}

    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.model(input_ids).last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

        return CausalLMOutput(loss=loss, logits=logits)


__all__ = ["GPTFSForCausalLM", "GPTFSModel", "GPTFSPreTrainedModel"]
