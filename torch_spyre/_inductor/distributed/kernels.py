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
    
    def spyre_broadcast_passthrough(x, src_rank, group_name):
        print("Spyre kernel placeholder for _c10d broadcast")
        return x

    def spyre_wait_tensor(x):
        print("Spyre kernel placeholder for _c10d wait_tensor")
        return x

    c10d_lib.impl("all_reduce", spyre_all_reduce_passthrough)
    c10d_lib.impl("broadcast", spyre_broadcast_passthrough)
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

    spyre_impl.impl("all_reduce_", spyre_all_reduce_impl)

    # ------------------------------------------------------------
    # Meta impl for tracing / fake tensor propagation
    # ------------------------------------------------------------
    spyre_meta = Library("spyre", "IMPL", "Meta")

    def spyre_all_reduce_meta(x, reduce_op="sum", group_name="default"):
        return torch.empty_like(x, device="meta")

    spyre_meta.impl("all_reduce", spyre_all_reduce_meta)

    # ------------------------------------------------------------
    # Broadcast operation - NOW USING @torch.library.custom_op in spyre_library.py
    # ------------------------------------------------------------
    # COMMENTED OUT: Old implementation replaced by @torch.library.custom_op
    # The new approach automatically handles FunctionalTensor wrapping and
    # guarantees fake implementation is called during compilation.
    # ------------------------------------------------------------
    
    """
    def is_compile_time_tensor(x):
        # Check if tensor is being used during compilation/tracing (not runtime)
        if x.device.type == "meta":
            return True
        if type(x).__name__ == "FakeTensor":
            return True
        if type(x).__name__ == "FunctionalTensor":
            return True
        if getattr(x, "fake_mode", None) is not None:
            return True
        try:
            if x.untyped_storage().data_ptr() == 0:
                return True
        except Exception:
            pass
        return False
    
    def spyre_broadcast_impl(x, src_rank=0, group_name="default"):
        # Old implementation - see spyre_library.py for new @torch.library.custom_op version
        import torch
        import spyre_comms
        import torch_spyre
        
        if type(x).__name__ == "FunctionalTensor":
            return x.clone()
        if is_compile_time_tensor(x):
            return torch.empty_like(x)
        
        out = x.clone()
        composite_addr_ptr = torch_spyre._C.get_composite_address_ptr(out)
        ctx = spyre_comms.get_world_context()
        rank = ctx.get_rank()
        
        if not out.is_contiguous():
            raise RuntimeError("spyre.broadcast currently requires contiguous input")
        
        shape = list(out.shape)
        dtype_map = {
            torch.float32: spyre_comms.TensorDataTypeEnum.float32,
            torch.float16: spyre_comms.TensorDataTypeEnum.float16,
            torch.bfloat16: spyre_comms.TensorDataTypeEnum.bfloat16,
            torch.int32: spyre_comms.TensorDataTypeEnum.int32,
            torch.int64: spyre_comms.TensorDataTypeEnum.int64,
        }
        
        spyre_dtype = dtype_map.get(out.dtype, spyre_comms.TensorDataTypeEnum.float32)
        tensor_info = spyre_comms.TensorInfo(spyre_dtype, spyre_comms.TensorShape(shape))
        buffer_tensor = spyre_comms.Tensor(tensor_info)
        buffer_tensor.set_spyre_device_address(composite_addr_ptr)
        
        work = ctx.broadcast(buffer_tensor, src_rank)
        work.start()
        work.wait()
        
        return out
    
    # OLD: spyre_impl.impl("broadcast", spyre_broadcast_impl)
    """
    
    # COMMENTED OUT: Meta implementation is now registered via @torch.library.register_fake
    # in spyre_library.py. Registering here causes a conflict.
    # The fake implementation provides shape/dtype metadata for AOT Autograd
    # without executing the operation, which keeps the op in the graph for lowering.
    
    # # Meta implementation for tracing
    # # This is what gets called during AOT Autograd and compilation
    # # It provides shape/dtype information without actual computation
    # def spyre_broadcast_meta(x, src_rank=0, group_name="default"):
    #     print(f"[BROADCAST_META] Called during tracing with x.shape={x.shape}, x.dtype={x.dtype}")
    #     # Return a meta tensor with the same shape and dtype
    #     # This tells the tracer what the output will look like without doing the actual broadcast
    #     return torch.empty_like(x, device="meta")
    #
    # spyre_meta.impl("broadcast", spyre_broadcast_meta)