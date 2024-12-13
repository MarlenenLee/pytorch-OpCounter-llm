import argparse
import logging
from .llm_calc_func import *
import torch
import torch.nn as nn

from thop import clever_format


def count_shapes(m, x, y):
    m.input_shapes[0,:x[0].ndim] = torch.tensor(x[0].shape)
    if len(x) >= 2 and torch.is_tensor(x[1]):
        m.input_shapes[1,:x[1].ndim] = torch.tensor(x[1].shape)

    if isinstance(y, tuple):
        m.output_shapes = torch.tensor(y[0].shape)
    else:
        m.output_shapes = torch.tensor(y.shape)

    if hasattr(m, 'weight'):
        m.weight_shapes = torch.tensor(m.weight.shape)


def count_embed(m, x, y):
    num_elements = y.numel()
    # m.weight.size() = (vocab_size, hidden_dim)
    total_mul = m.num_embeddings
    total_add = m.num_embeddings - 1
    total_ops = total_mul + total_add
    m.total_ops += calculate_embed(total_ops, num_elements)
    count_shapes(m, x, y)


def count_silu(m, x, y):
    num_elements = y.numel()
    m.total_ops += calculate_silu(num_elements)
    count_shapes(m, x, y)


def count_rope(m, x, y):
    num_elements = x[0].numel() + x[1].numel()
    m.total_ops += calculate_rope(num_elements)
    count_shapes(m, x, y)


def count_rmsn(m, x, y):
    ndim = x[0].shape[-1]
    nfeatures = x[0].numel() / ndim
    num_elements = y.numel()
    m.total_ops += calculate_rmsn(ndim, nfeatures, num_elements)
    count_shapes(m, x, y)


def count_mm(m, x, y):
    num_elements = y.numel()
    ndim = x[0].shape[-1]
    m.total_ops += calculate_mm(ndim, num_elements)
    count_shapes(m, x, y)


def count_ds(m, x, y):
    """div_scalar"""
    num_elements = x[0].numel()
    m.total_ops += calculate_meta(num_elements)
    count_shapes(m, x, y)


def count_ma(m, x, y):
    num_elements = y.numel()
    m.total_ops += calculate_meta(num_elements)
    count_shapes(m, x, y)
