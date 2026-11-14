"""Sections 3.6.1 and 3.6.2: MultiHeadAttentionWrapper and MultiHeadAttention."""
import torch
import torch.nn as nn

from causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    """Section 3.6.1: several CausalAttention heads run side by side.

    __init__ creates an nn.ModuleList named heads holding num_heads CausalAttention modules,
    each built with the same (d_in, d_out, context_length, dropout, qkv_bias).

    forward(x) runs every head on x and concatenates their outputs along the last dim,
    so (batch, num_tokens, d_in) becomes (batch, num_tokens, num_heads * d_out).
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # create an nn.ModuleList named heads holding num_heads CausalAttention modules
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        # x shape: (b, num_tokens, d_in)
        logits_list = []
        for head in self.heads:
            head_logits = head(x)  # (b, num_tokens, d_out)
            logits_list.append(head_logits)

        logits = torch.cat(logits_list, dim=-1)
        return logits



class MultiHeadAttention(nn.Module):
    """Section 3.6.2: multi-head attention with weight splits.

    d_out must be divisible by num_heads; each head works on head_dim = d_out // num_heads.
    __init__ stores d_out, num_heads and head_dim, then creates, in this order:
    nn.Linear(d_in, d_out) layers with bias=qkv_bias named W_query, W_key, W_value;
    nn.Linear(d_out, d_out) named out_proj (with its default bias); nn.Dropout(dropout) named
    dropout; and a buffer named mask, the same causal mask as in CausalAttention.

    forward(x) takes (batch, num_tokens, d_in):
    project to queries, keys, values of size d_out; split the last dim into (num_heads, head_dim)
    and move the head dim in front of the token dim, giving (batch, num_heads, num_tokens, head_dim);
    then per head do the same masked, scaled (by the square root of head_dim) softmax attention
    with dropout as CausalAttention; move the head dim back behind the token dim, merge heads
    back into d_out, and pass the result through out_proj.
    Returns shape (batch, num_tokens, d_out).
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        ones_ = torch.ones(context_length, context_length)
        mask_ = torch.triu(ones_, diagonal=1)
        # Register as a buffer instead of a standard attribute
        self.register_buffer("mask", mask_)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)


    def forward(self, x):
        # raise NotImplementedError
        # x shape (b, num_tokens, d_in)

        _, num_tokens, _ = x.shape
        queries = self.W_query(x)   # (b, num_tokens, d_out)
        keys = self.W_key(x)        # (b, num_tokens, d_out)
        values = self.W_value(x)     # (b, num_tokens, d_out)

        # construct the head dimension
        queries = queries.reshape(-1, num_tokens, self.num_heads, self.head_dim)    # (b, num_tokens, num_heads, head_dim)
        keys = keys.reshape(-1, num_tokens, self.num_heads, self.head_dim)          # (b, num_tokens, num_heads, head_dim)
        values = values.reshape(-1, num_tokens, self.num_heads, self.head_dim)      # (b, num_tokens, num_heads, head_dim)

        queries = queries.transpose(1, 2)   # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)         # (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)     # (b, num_heads, num_tokens, head_dim)

        attention_scores = queries @ keys.transpose(2,3)  # # (b, num_heads, num_tokens, num_tokens)

        # causal
        attention_scores = torch.masked_fill(attention_scores, self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # (b, num_heads, num_tokens, num_tokens)

        # prob
        attention_weights = torch.softmax(attention_scores / (self.head_dim**0.5), dim=-1) # (b, num_heads, num_tokens, num_tokens)
        
        # dropout
        attention_weights = self.dropout(attention_weights) # (b, num_heads, num_tokens, num_tokens)

        # context
        context_vec = attention_weights @ values  # (b, num_heads, num_tokens, head_dim)
        context_vec = context_vec.transpose(1,2)  # (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.reshape(-1, num_tokens, self.num_heads*self.head_dim)  #  (b, num_tokens, d_out)   

        # out_proj
        context_vec = self.out_proj(context_vec)

        return context_vec