"""Sections 3.3.1 and 3.4.2: softmax_naive, SelfAttention_v1, SelfAttention_v2."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_naive(x):
    """Section 3.3.1: softmax written by hand.

    Exponentiate x and divide by the sum of the exponentials over the last dim
    (the book uses dim 0, which is the same thing for its 1-D input).
    Returns a tensor of the same shape as x; for a 2-D input each row sums to 1.
    """
    # raise NotImplementedError
    exp_x = torch.exp(x)  # (b, num_tokens)
    exp_x_sum = torch.sum(exp_x, dim=-1, keepdim=True)  # (b)
    
    sftmax = exp_x / exp_x_sum

    assert x.shape == sftmax.shape
    return sftmax


class SelfAttention_v1(nn.Module):
    """Section 3.4.2: self-attention with raw weight matrices.

    __init__ creates three nn.Parameter attributes, each filled with torch.rand of shape
    (d_in, d_out), in this order: W_query, W_key, W_value (order matters under the seed).

    forward(x) takes a 2-D input (num_tokens, d_in):
    queries, keys and values are x matrix-multiplied by the matching weight matrix;
    attention scores are queries times the transpose of keys;
    attention weights are the softmax over the last dim of the scores divided by
    the square root of the key dimension (d_out);
    returns the weights times the values, shape (num_tokens, d_out).
    """

    def __init__(self, d_in, d_out):
        super().__init__()

        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        # raise NotImplementedError

    def forward(self, x):
        # raise NotImplementedError

        # x shape (num_tokens, d_in)
        queries = x @ self.W_query   # (num_tokens, d_out)
        keys = x @ self.W_key        # (num_tokens, d_out)
        values = x @ self.W_value    # (num_tokens, d_out)

        attention_scores = queries @ keys.T # (num_tokens, num_tokens)

        d_emb = torch.tensor(queries.shape[-1])
        attention_weights = \
            F.softmax(attention_scores / torch.sqrt(d_emb), dim=-1) # (num_tokens, num_tokens)

        attn = attention_weights @ values # (num_tokens, d_out)
        return attn


class SelfAttention_v2(nn.Module):
    """Section 3.4.2: self-attention using nn.Linear layers.

    __init__ creates three nn.Linear(d_in, d_out) layers with bias=qkv_bias,
    in this order: W_query, W_key, W_value (order matters under the seed).

    forward(x) is the same computation as SelfAttention_v1, except queries, keys and values
    come from calling the linear layers on x. Input (num_tokens, d_in), output (num_tokens, d_out).
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        
        queries = self.W_query(x)   # (num_tokens, d_out)
        keys = self.W_key(x)        # (num_tokens, d_out)
        values = self.W_value(x)    # (num_tokens, d_out)

        attention_scores = queries @ keys.T # (num_tokens, num_tokens)

        d_emb = torch.tensor(queries.shape[-1])
        attention_weights = \
            F.softmax(attention_scores / torch.sqrt(d_emb), dim=-1) # (num_tokens, num_tokens)

        attn = attention_weights @ values # (num_tokens, d_out)
        return attn
