import math
from typing import Any
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)  

d = 32
M = 24

def softreorder_elementwise(inp, emb, i, j, dimension=d, channels=M):
        # assert inp.shape[1] == d
        # assert emb.shape[1] == d
        # assert emb.shape[0] == M
        det = 0
        for col in inp.t():
            inc = torch.exp(torch.sum(torch.matmul(emb[i].view(-1, 1), col.view(1, -1))))
            det += inc 
        nom = torch.exp(torch.sum(torch.matmul(emb[i].view(-1, 1), emb.t()[j].view(1, -1))))
        return nom / det

def softreorder(inp, emb):
    remap = torch.zeros(emb.shape)
    for i in range(emb.shape[0]):
        for j in range(emb.shape[1]):
            out = softreorder_elementwise(inp, emb, i, j)
            remap[i, j] = out
    return remap

# conv layer to map recorded input to fix dimension of d
class ConvMaxPool(nn.Module):
    def __init__(self, dimension=d, k=3):
        super(ConvMaxPool, self).__init__()
        self.conv_layer = nn.Conv1d(1, d, k, stride=2)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.output_dim = d

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a dimension for the channel
        conv_output = self.conv_layer(x)
        pooled_output = self.max_pool(conv_output)
        output = pooled_output.squeeze(2)  # Remove the extra dimension
        return output
    
# embedding layer
class Embedding(nn.Module):
    def __init__(self, dimension=d, channels=M):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(channels, dimension))  

    def forward(self, x):
        return softreorder(x, self.weight)
    
# define neural net
class CHARM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.map_layer = ConvMaxPool()
        self.embedding_layer = Embedding()
        # tbc
        
    def forward(self, x):
        x = self.map_layer(x)
        x = self.embedding_layer(x)
        # tbc
        return x

model = CHARM()
# nn.init.xavier_uniform_(model.embedding_layer.weight)  
# nn.init.xavier_uniform_(model.map_layer.weight)  
x = torch.rand(22, 1001)
out = model(x)
print(out.shape)