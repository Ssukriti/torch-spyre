import torch
from torch.library import Library

lib = Library("spyre", "DEF")

# Define all_reduce with mutation/aliasing semantics
# Tensor(a!) means it mutates the input, -> Tensor(a!) means it returns the same alias
lib.define(
    "all_reduce_(Tensor(a!) x, str reduce_op='sum', str group_name='default') -> Tensor(a!)"
)

# Define broadcast with mutation/aliasing semantics
# This tells PyTorch that broadcast modifies x in-place and returns it
lib.define(
    "broadcast_(Tensor(a!) x, int src_rank=0, str group_name='default') -> Tensor(a!)"
)