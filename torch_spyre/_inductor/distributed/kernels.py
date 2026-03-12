import torch
from torch.library import Library

# Register implementation for Spyre backend - This is needed because integration is not happening correctly today
# This file can go away with proper Spyre backend integration
# without this file we get error
# `NotImplementedError: Could not run '_c10d_functional::all_reduce'
# with arguments from the 'AutogradSpyre' backend...`
# when running examples/all_reduce_compile_demo.py
lib = Library("_c10d_functional", "IMPL", "AutogradPrivateUse1")

def spyre_dummy_all_reduce(x, reduce_op, group_name):
    print("Dummy Spyre _c10d_functional.all_reduce kernel")
    return x

lib.impl("all_reduce", spyre_dummy_all_reduce)

def spyre_dummy_wait_tensor(x):
    print("Dummy Spyre wait_tensor kernel")
    return x

lib.impl("wait_tensor", spyre_dummy_wait_tensor)