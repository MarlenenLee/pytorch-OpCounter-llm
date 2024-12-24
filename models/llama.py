import os
import math
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from .basic_modules import (
    MatMul,
    DivScalar,
    MatAdd,
)
from thop.vision.basic_hooks import *
from thop.llm.llm_hooks import *


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        input_dtype = x.dtype
        output = self._norm(x.float()).type_as(x)
        return output.to(input_dtype) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_q_dtype = xq.dtype
    input_k_dtype = xk.dtype
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(input_q_dtype), xk_out.type_as(xk).to(input_k_dtype)


class RoPE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xq, xk, freqs_cis):
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
        return xq_out, xk_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = args.model_parallel_size
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_local_heads * self.head_dim, args.dim, bias=False)

        self.rotary_emb = RoPE()
        self.matmul_qk = MatMul()
        self.div_dim = DivScalar()
        self.softmax = nn.Softmax(dim=-1)
        self.matmul_v = MatMul()

        self.addmsk = MatAdd()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        #xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = self.rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys   = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)                                 # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)                             # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)                         # (bs, n_local_heads, cache_len + seqlen, head_dim)

        #scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        qk_tmp = self.matmul_qk(xq, keys.transpose(2, 3))
        scores = self.div_dim(qk_tmp, math.sqrt(self.head_dim))
        
        if mask is not None:
            scores = self.addmsk(scores, mask)
            #scores = scores + mask                              # (bs, n_local_heads, seqlen, cache_len + seqlen)

        #scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.softmax(scores.float()).type_as(xq)
        #output = torch.matmul(scores, values)                   # (bs, n_local_heads, seqlen, head_dim)
        output = self.matmul_v(scores, values)
        output = output.to(xq.dtype)

        ## Use fused self-attention:
        #is_causal = True if mask is None and seqlen > 1 else False
        #output = torch.nn.functional.scaled_dot_product_attention(
        #            xq, keys, values, attn_mask=mask, is_causal=is_causal)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        #hidden_dim: int,
        #multiple_of: int,
        #ffn_dim_multiplier: Optional[float],
        intermediate_size: int,
        model_parallel_size: int, ## added
    ):
        super().__init__()
        #hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        #if ffn_dim_multiplier is not None:
        #    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        #hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        #hidden_dim = hidden_dim // model_parallel_size    #added
        hidden_dim = intermediate_size // model_parallel_size

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.silu = nn.SiLU()

    def forward(self, x):
        #return self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            #hidden_dim=4 * args.dim,
            #multiple_of=args.multiple_of,
            #ffn_dim_multiplier=args.ffn_dim_multiplier,
            intermediate_size=args.intermediate_size,
            model_parallel_size=args.model_parallel_size,    #added
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.atten_resadd = MatAdd()
        self.ffn_resadd = MatAdd()


    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        #h = x + self.attention(
        #    self.attention_norm(x), start_pos, freqs_cis, mask
        #)
        h = self.atten_resadd(x, self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask))

        #out = h + self.feed_forward(self.ffn_norm(h))
        out = self.ffn_resadd(h, self.feed_forward(self.ffn_norm(h)))

        return out


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )


#    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h)
        return output


custom_ops = {
    RoPE: count_rope,
    RMSNorm: count_rmsn,
    MatMul: count_mm,
    DivScalar: count_ds,
    MatAdd: count_ma,
    Attention: zero_ops,
    FeedForward: zero_ops,
    TransformerBlock: zero_ops,
    Transformer: zero_ops,
}
