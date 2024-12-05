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
    """
    x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
    """
    square_ops = num_elements
    mean_ops = num_elements
    eps_add_ops = nfeatures
    sqrt_ops = nfeatures
    norm_ops = 2*num_elements
    total_ops = square_ops + mean_ops + eps_add_ops + sqrt_ops + norm_ops
    return torch.DoubleTensor([int(total_ops)])


def calculate_mm(ndim, num_elements):
    total_ops = num_elements * (2 * ndim - 1)
    return torch.DoubleTensor([int(total_ops)])


def calculate_meta(num_elements):
    return torch.DoubleTensor([int(num_elements)])
