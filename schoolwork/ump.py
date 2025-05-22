import numpy as np
import torch
from torch import Tensor
from torch import tensor
from torch import nn


class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.rand((1,2))
        self.param = nn.Parameter(self.tensor)
    def forward(self,x):
        x= self.param(x)
        return torch.abs(x)


net = XORNet()

net(torch.tensor([1,0]))