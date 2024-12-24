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
    n_layers: int = 1                # 80
    n_heads: int = 64                # 64
    n_kv_heads: int = 8              # 8
    vocab_size: int = 32000
    #multiple_of: int = 4096
    #ffn_dim_multiplier: float = 1.3
    intermediate_size: int = 28672   # 28672 
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 4096

    model_parallel_size: int = 4

    batch_size: int = 1
    seq_len: int = 4096

    use_amp: bool = True



if __name__ == "__main__":
    args = ModelArgs()
    batch_size = args.batch_size
    seq_len = args.seq_len
    device = torch.device('cuda:0')

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
