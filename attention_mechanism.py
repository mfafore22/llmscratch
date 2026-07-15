import torch 
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]

    ]
)

attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j , x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

attn_scores = inputs @ inputs.T


attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)

all_context_vecs = attn_weights @ inputs


#extend self attention mechanism with trainable weights
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

#Next initialize the three weights matrices 
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

#Next compute the query, key and value vectors
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

keys = inputs @ W_key
values = inputs @ W_value

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
attn_scores_2 = query_2 @ keys.T
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, )
context_vec_2 = attn_weights_2 @ values