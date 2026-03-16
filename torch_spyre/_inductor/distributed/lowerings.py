import torch
from .spyre_ops import spyre_all_reduce_async, spyre_wait

LOWERED_OPS = {}

def register_lowerings():
    # This will be exapnded to other collectives in future
    LOWERED_OPS.update({
        torch.ops._c10d_functional.all_reduce: spyre_all_reduce_async,
        torch.ops._c10d_functional.wait_tensor: spyre_wait,
    })