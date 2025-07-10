import torch
import torch.nn as nn
torch.manual_seed(123)

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key       # (l, d_out)
        queries = x @ self.W_query  # (l, d_out)
        values = x @ self.W_value   # (l, d_out)
        
        attn_scores = queries @ keys.T # omega # (l, l)

        # (l, l)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # (l, d_out)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)        # (l, d_out)
        queries = self.W_query(x)   # (l, d_out)
        values = self.W_value(x)    # (l, d_out)
        
        attn_scores = queries @ keys.T  # (l, l)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # (l, l)

        context_vec = attn_weights @ values # (l, d_out)
        return context_vec


inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2              # the output embedding size, d=2

sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

print(f"sa_v2.W_query.weight: {sa_v1.W_query.shape}")
print(f"sa_v2.W_query.weight: {sa_v2.W_query.weight.shape}")
