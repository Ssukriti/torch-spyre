import torch
from torch.library import Library

lib = Library("spyre", "DEF")

lib.define(
    "all_reduce(Tensor x, str reduce_op='sum', str group_name='default') -> Tensor"
)