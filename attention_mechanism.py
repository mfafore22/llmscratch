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
