import torch

LOWERED_OPS = {}

def register_lowerings():
    LOWERED_OPS.update({
        torch.ops._c10d_functional.all_reduce: torch.ops.spyre.all_reduce,
    })