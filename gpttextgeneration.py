import torch
from gptmodel import GPTModel
from gptconfig import GPT_CONFIG_124M

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()