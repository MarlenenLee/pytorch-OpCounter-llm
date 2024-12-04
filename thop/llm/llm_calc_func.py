import torch


def calculate_embed(embed_dim, num_elements):
    return torch.DoubleTensor([int(embed_dim * num_elements)])


def calculate_silu(num_elements):
    """
        x * 1 / (1 + e ** (-x)) = x * 1 / (1 + 2 ** (-c*x))
    """
    return torch.DoubleTensor([int(6 * num_elements)])


def calculate_rope(num_elements):
    """
        q * cos + rotate_half(q) * sin
        k * cos + rotate_half(k) * sin
        num_elements = q.numel() + k.numel()
    """
    mul = 2
    add = 1
    return torch.DoubleTensor([int((mul + add) * num_elements)])


def calculate_rmsn(ndim, nfeatures, num_elements):
    # rms:
    rms_mul = ndim
    rms_add = ndim - 1
    rms_div = 1
    rms_add += 1 # for adding epsilon
    rms_sqrt = 1
    total_ops = (rms_mul + rms_add + rms_div + rms_sqrt) * nfeatures
    
    # norm:
    ops = 2 # 1 div & 1 mul
    total_ops += num_elements * ops
    return torch.DoubleTensor([int(total_ops)])


def calculate_mm(ndim, num_elements):
    total_ops = num_elements * (2 * ndim - 1)
    return torch.DoubleTensor([int(total_ops)])


def calculate_meta(num_elements):
    return torch.DoubleTensor([int(num_elements)])
