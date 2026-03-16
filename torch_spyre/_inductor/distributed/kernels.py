import torch
from torch.library import Library

lib = Library("_c10d_functional", "IMPL", "AutogradPrivateUse1")


def spyre_all_reduce_passthrough(x, reduce_op, group_name):
    print("Spyre kernel placeholder for all_reduce")
    return x


lib.impl("all_reduce", spyre_all_reduce_passthrough)


def spyre_wait_tensor(x):
    print("Spyre wait_tensor kernel")
    return x


lib.impl("wait_tensor", spyre_wait_tensor)