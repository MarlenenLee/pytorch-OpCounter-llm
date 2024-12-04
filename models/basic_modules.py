import torch
import torch.nn as nn


class MatMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class DivScalar(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x / y


class MatAdd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y
