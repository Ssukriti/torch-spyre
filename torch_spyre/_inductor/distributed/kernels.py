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

    def spyre_all_reduce_passthrough(x, reduce_op, group_name):
        print("Spyre kernel placeholder for _c10d all_reduce")
        return x

    def spyre_wait_tensor(x):
        print("Spyre kernel placeholder for _c10d wait_tensor")
        return x

    c10d_lib.impl("all_reduce", spyre_all_reduce_passthrough)
    c10d_lib.impl("wait_tensor", spyre_wait_tensor)

    # ------------------------------------------------------------
    # CPU impl: reference path that stays entirely on CPU
    # ------------------------------------------------------------
    spyre_cpu = Library("spyre", "IMPL", "CPU")

    def spyre_all_reduce_cpu(x, reduce_op="sum", group_name="default"):
        print("Spyre all_reduce called on CPU")

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
    # Lowered Spyre op implementation for Spyre tensors
    # ------------------------------------------------------------
    spyre_impl = Library("spyre", "IMPL", "AutogradPrivateUse1")

    def spyre_all_reduce_impl(x, reduce_op="sum", group_name="default"):
        print("Spyre all_reduce called")

        cpu_tensor = x.detach().clone().cpu().contiguous()

        reduce_map = {
            "sum": dist.ReduceOp.SUM,
            "avg": dist.ReduceOp.AVG,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }
        op = reduce_map.get(str(reduce_op), dist.ReduceOp.SUM)

        if dist.is_initialized():
            # Trial Use the process group API directly instead of dist.all_reduce
            # to try avoid surfacing c10d.allreduce_.default in the compiled path.
            # update: did not help
            if group_name == "default":
                pg = c10d._get_default_group()
            else:
                # just a stub
                pg = c10d._get_default_group()

            work = pg.allreduce([cpu_tensor], op)
            work.wait()

        out = torch.empty_like(cpu_tensor, device=x.device)
        out.copy_(cpu_tensor)
        return out

    spyre_impl.impl("all_reduce", spyre_all_reduce_impl)

    # ------------------------------------------------------------
    # Meta impl for tracing / fake tensor propagation
    # ------------------------------------------------------------
    spyre_meta = Library("spyre", "IMPL", "Meta")

    def spyre_all_reduce_meta(x, reduce_op="sum", group_name="default"):
        return torch.empty_like(x, device="meta")

    spyre_meta.impl("all_reduce", spyre_all_reduce_meta)