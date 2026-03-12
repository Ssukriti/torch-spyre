import torch
from .spyre_ops import spyre_all_reduce

LOWERED_OPS = {}

def register_lowerings():
    # This will be exapnded to other collectives in future
    LOWERED_OPS[torch.ops._c10d_functional.all_reduce] = spyre_all_reduce