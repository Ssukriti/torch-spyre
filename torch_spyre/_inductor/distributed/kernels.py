import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.library import Library

if not hasattr(torch, "_spyre_distributed_kernels_registered"):
    torch._spyre_distributed_kernels_registered = True

    # ------------------------------------------------------------
    # Placeholder kernels for original functional collectives
    # on Spyre tensors, so tracing / eager paths do not fail
    # before lowering happens.
    # ------------------------------------------------------------
    c10d_lib = Library("_c10d_functional", "IMPL", "AutogradPrivateUse1")
    
    def spyre_broadcast_passthrough(x, src_rank, group_name):
        return x

    def spyre_wait_tensor(x):
        return x

    c10d_lib.impl("broadcast", spyre_broadcast_passthrough)
    c10d_lib.impl("wait_tensor", spyre_wait_tensor)
