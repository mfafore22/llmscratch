import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__input__()
        self.W_query = nn.Parameter(d_in, d_out, bias=qkv_bias)
        self.W_key =  nn.Parameter(d_in , d_out, bias=qkv_bias)
        self.W_value = nn.Parameter(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T #omegra
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values 
        return context_vec
    