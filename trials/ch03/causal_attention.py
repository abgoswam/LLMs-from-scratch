"""Section 3.5.3: CausalAttention."""
import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    """Section 3.5.3: batched self-attention with a causal mask and dropout.

    __init__ stores d_out, then creates, in this order: three nn.Linear(d_in, d_out) layers with
    bias=qkv_bias named W_query, W_key, W_value; an nn.Dropout(dropout) named dropout; and a
    buffer (register_buffer, not a parameter) named mask, holding the upper-triangular matrix of
    ones of shape (context_length, context_length) above the diagonal (diagonal itself is 0).

    forward(x) takes a 3-D input (batch, num_tokens, d_in):
    attention scores are queries times keys transposed over their last two dims;
    positions where the mask (cut to num_tokens x num_tokens, as booleans) is set are filled
    with minus infinity, so each token only attends to itself and earlier tokens;
    softmax over the last dim of scores divided by the square root of the key dimension;
    dropout is applied to the attention weights;
    returns weights times values, shape (batch, num_tokens, d_out).
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        # raise NotImplementedError

        self.d_out = d_out
        
        self.W_query = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(p=dropout)

        # upper triangular matix of ones of shape (context_length, context_length) above the diagonal 
        ones_ = torch.ones(context_length, context_length)
        mask_ = torch.triu(ones_, diagonal=1).bool()  # (context_length, context_length)
        
        # Register as a buffer instead of a standard attribute
        self.register_buffer("mask", mask_)


    def forward(self, x):
        # raise NotImplementedError

        # x shape : (batch, num_tokens , d_in)

        _, num_tokens, _ = x.shape
        Q = self.W_query(x) # (batch, num_tokens , d_out)
        K = self.W_key(x) # (batch, num_tokens , d_out)
        V = self.W_value(x) # (batch, num_tokens , d_out)

        # compute attention_scores
        attention_scores = Q @ K.transpose(1,2)  # (batch, num_tokens , num_tokens)

        # apply the causal mask , by setting upper triangle to -inf
        attention_scores = attention_scores.masked_fill(self.mask[:num_tokens, :num_tokens], -torch.inf)

        # compute attention weights
        attention_weights_sf = torch.softmax(attention_scores / self.d_out**0.5, dim=-1) # (batch, num_tokens , num_tokens)
        attention_weights_dp = self.dropout(attention_weights_sf)
        
        context_vec = attention_weights_dp @ V  # (batch, num_tokens , d_out)

        return context_vec

