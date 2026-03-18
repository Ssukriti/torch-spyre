import torch
import torch.distributed as dist
from torch.library import Library
import torch.distributed.distributed_c10d as c10d

if not hasattr(torch, "_spyre_distributed_kernels_registered"):
    torch._spyre_distributed_kernels_registered = True

    # ------------------------------------------------------------
    # Placeholder kernels for original functional collectives
    # ------------------------------------------------------------
    c10d_lib = Library("_c10d_functional", "IMPL", "AutogradPrivateUse1")

    def spyre_all_reduce_passthrough(x, reduce_op, group_name):
        print("Spyre kernel placeholder for _c10d all_reduce")
        return x

    def spyre_wait_tensor(x):
        print("Spyre kernel placeholder for _c10d wait_tensor")
        return x

    c10d_lib.impl("all_reduce", spyre_all_reduce_passthrough)
    c10d_lib.impl("wait_tensor", spyre_wait_tensor)

    # ------------------------------------------------------------
    # CPU runtime kernel
    # ------------------------------------------------------------
    spyre_cpu = Library("spyre", "IMPL", "CPU")

    def spyre_all_reduce_cpu(x, reduce_op="sum", group_name="default"):
        print("REAL spyre custom_op runtime called on CPU")

        reduce_map = {
            "sum": dist.ReduceOp.SUM,
            "avg": dist.ReduceOp.AVG,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }
        op = reduce_map.get(str(reduce_op), dist.ReduceOp.SUM)

        if dist.is_initialized():
            dist.all_reduce(x, op=op)

        return x

    spyre_cpu.impl("all_reduce", spyre_all_reduce_cpu)

    # ------------------------------------------------------------
    # Spyre runtime kernel
    # ------------------------------------------------------------
    spyre_privateuse1 = Library("spyre", "IMPL", "AutogradPrivateUse1")

    def spyre_all_reduce_privateuse1(x, reduce_op="sum", group_name="default"):
        print("REAL spyre custom_op runtime called on Spyre tensor")

        cpu_tensor = x.detach().clone().cpu().contiguous()

        # Get default process group
        pg = c10d._get_default_group()

        reduce_map = {
           "sum": c10d.ReduceOp.SUM,
           "avg": c10d.ReduceOp.AVG,
           "max": c10d.ReduceOp.MAX,
           "min": c10d.ReduceOp.MIN,
        }
        op = reduce_map.get(str(reduce_op), c10d.ReduceOp.SUM)

        # IMPORTANT: list of tensors
        work = pg.barrier()
        work.wait()
        return x

    spyre_privateuse1.impl("all_reduce", spyre_all_reduce_privateuse1)