import sys
sys.path.insert(0, "/root/workspace/pytorch-OpCounter/thop")
import torch
from torch import nn
from thop import profile
from thop import clever_format
from models.llama import (
    Transformer,
    custom_ops,
)


class ModelArgs:
    dim: int = 8192                  # 8192
    n_layers: int = 3                # 80
    n_heads: int = 64                # 64
    n_kv_heads: int = 8              # 8
    vocab_size: int = 32256          # 32000
    #multiple_of: int = 4096
    #ffn_dim_multiplier: float = 1.3
    intermediate_size: int = 28672   # 28672 
    norm_eps: float = 1e-5
    max_batch_size: int = 1
    max_seq_len: int = 2048

    model_parallel_size: int = 4

    batch_size: int = 8
    seq_len: int = 2048

    use_amp: bool = True
    reduce_bucket_size = 500000000



if __name__ == "__main__":
    args = ModelArgs()
    batch_size = args.batch_size
    seq_len = args.seq_len
    device = torch.device('cuda:0')
    """
    tokens = torch.randint(low=3,high=args.vocab_size,size=(batch_size,seq_len))
    tokens = tokens.to(device)

    ## ==== Load Model ==== #
    model  = Transformer(args).to(device)
    logits = model(tokens, 0)
    macs, params, ret_dict = profile(
        model,
        custom_ops=custom_ops,
        inputs=(tokens, 0),
        ret_layer_info=True,
        report_missing=True,
    )
    print("\nTotal FLOPs:", clever_format(macs), 
            "\tTotal Params:", clever_format(params))
    """
    total_params = 2 * args.vocab_size / args.model_parallel_size * args.dim + \
                 args.n_layers * (2 * args.dim * args.dim / args.model_parallel_size + \
                 2 * args.dim * args.dim * args.n_kv_heads / args.n_heads / args.model_parallel_size + \
                 3 * args.dim * args.intermediate_size / args.model_parallel_size + 2 * args.dim) + args.dim

    # params_without_norm_weight:
    _params = total_params - 2 * args.n_layers * args.dim - args.dim

    optim_peak_mem = 2 * total_params + 18 * _params + \
            2 * args.batch_size * args.n_heads / args.model_parallel_size * args.seq_len * args.seq_len + \
            4 * args.batch_size * args.seq_len * args.vocab_size / args.model_parallel_size

    peak_mem = 6 * _params + 4 * args.batch_size * args.seq_len * args.dim + \
            2 * args.batch_size * args.n_heads / args.model_parallel_size * args.seq_len * args.seq_len + \
            args.n_layers * (10 * args.batch_size * args.seq_len * args.dim + \
            8 * args.batch_size * args.seq_len * args.n_heads / args.model_parallel_size * (args.dim / args.n_heads) + \
            2 * args.batch_size * args.n_heads / args.model_parallel_size * args.seq_len * args.seq_len + \
            8 * args.batch_size * args.seq_len * args.intermediate_size / args.model_parallel_size) + \
            10 * args.batch_size * args.seq_len * args.vocab_size / args.model_parallel_size + 2 * args.reduce_bucket_size

    print("params:", total_params)
    print("Optim Step Peak Memory:", optim_peak_mem, "=>", clever_format(optim_peak_mem))
    print("BWD Peak Memory:", peak_mem, "=>", clever_format(peak_mem))
