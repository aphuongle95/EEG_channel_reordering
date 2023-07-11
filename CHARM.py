import math
from typing import Any
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)  

def softreorder_elementwise(inp, emb, i, j):
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
    def __init__(self, in_channels, out_channels, k=3):
        super(ConvMaxPool, self).__init__()
        # self.conv_layer = nn.Conv1d(l, d, k, stride=2)
        self.conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=1, padding=1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.output_dim = d

    def forward(self, x):
        # x = x.unsqueeze(1)  # Add a dimension for the channel
        # x = x.view(b * d, l)
        print(x.shape)
        x = x.transpose(1, 2).contiguous()
        print(x.shape)
        x = self.conv_layer(x)
        print(x.shape) 
        output = x.transpose(1, 2).contiguous()        
        print(output.shape) # expect [b, 22, 32]

        return output
    
# embedding layer
class Embedding(nn.Module):
    def __init__(self, dimension, channels):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.weight = nn.Parameter(torch.Tensor(channels, dimension))  

    def _check_sanity(self, inp):
        # assert inp.shape[0] == self.channels 
        print(inp.shape)
        assert inp.shape[-1] == self.dimension # d 
        
    def forward(self, x):
        self._check_sanity(x)
        emb = torch.zeros((x.shape[0], self.channels, self.dimension))
        for t in range(x.shape[0]):
            emb[t] = softreorder(emb[t], self.weight)
        return emb
        
    
# define neural net
class CHARM(nn.Module):
    def __init__(self, dimension, channels, length) -> None:
        super().__init__()
        self.map_layer = ConvMaxPool(in_channels = length, out_channels = dimension)
        self.embedding_layer = Embedding(dimension, channels)
        # tbc
        
    def forward(self, x):
        x = self.map_layer(x)
        x = self.embedding_layer(x)
        return x

if __name__ == "__main__":
    b = 5
    d = 32
    M = 24
    l = 1001
    model = CHARM(dimension=d, channels=M, length=l)
    # nn.init.xavier_uniform_(model.embedding_layer.weight)  
    # nn.init.xavier_uniform_(model.map_layer.weight) 
    x = torch.rand(b, 22, l)
    out = model(x)
    print(out.shape) # expect (b, 24, 32)
    