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

    spyre_impl.impl("all_reduce_", spyre_all_reduce_impl)

    # ------------------------------------------------------------
    # Meta impl for tracing / fake tensor propagation
    # ------------------------------------------------------------
    spyre_meta = Library("spyre", "IMPL", "Meta")

    def spyre_all_reduce_meta(x, reduce_op="sum", group_name="default"):
        return torch.empty_like(x, device="meta")

    spyre_meta.impl("all_reduce", spyre_all_reduce_meta)

    # ------------------------------------------------------------
    # Broadcast operation - using spyre-comms on Spyre tensors
    # ------------------------------------------------------------
    
    def is_fake_or_meta_tensor(x):
        """Check if tensor is a FakeTensor or meta tensor (used during tracing)"""
        return (
            x.device.type == "meta"
            or type(x).__name__ == "FakeTensor"
            or getattr(x, "fake_mode", None) is not None
        )
    
    def spyre_broadcast_impl(x, src_rank=0, group_name="default"):
        """Spyre broadcast implementation using spyre-comms library
        
        Broadcasts tensor from src_rank to all other ranks using spyre-comms.
        Works directly on Spyre device tensors without copying to CPU.
        """
        import spyre_comms
        import torch_spyre
        
        # Check if this is a FakeTensor or meta tensor (during tracing/compilation)
        # These don't have actual storage, so we just return the input
        if is_fake_or_meta_tensor(x):
            print(f"[FakeTensor/Meta] Skipping broadcast during tracing")
            return x
        
        # Check if we're in a tracing/compilation context by checking if storage is allocated
        # During AOT Autograd, tensors may not have storage allocated
        if not x.is_contiguous() or x.untyped_storage().size() == 0:
            print(f"[No Storage] Skipping broadcast during AOT Autograd (storage size: {x.untyped_storage().size()})")
            return x
        
        # Get the world context
        ctx = spyre_comms.get_world_context()
        rank = ctx.get_rank()
        
        print(f"[Rank {rank}] Spyre broadcast called - using spyre-comms (src_rank={src_rank})")
        print(f"[Rank {rank}] Input tensor: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        print(f"[Rank {rank}] Tensor is_contiguous={x.is_contiguous()}, numel={x.numel()}")
        
        # Ensure tensor is contiguous
        if not x.is_contiguous():
            print(f"[Rank {rank}] WARNING: Tensor not contiguous, making contiguous")
            x = x.contiguous()
        
        # Get tensor shape and dtype
        shape = list(x.shape)
        
        # Map torch dtype to spyre_comms dtype
        dtype_map = {
            torch.float32: spyre_comms.TensorDataTypeEnum.float32,
            torch.float16: spyre_comms.TensorDataTypeEnum.float16,
            torch.bfloat16: spyre_comms.TensorDataTypeEnum.bfloat16,
            torch.int32: spyre_comms.TensorDataTypeEnum.int32,
            torch.int64: spyre_comms.TensorDataTypeEnum.int64,
        }
        
        spyre_dtype = dtype_map.get(x.dtype, spyre_comms.TensorDataTypeEnum.float32)
        tensor_info = spyre_comms.TensorInfo(spyre_dtype, spyre_comms.TensorShape(shape))
        
        print(f"[Rank {rank}] Creating spyre_comms tensor with shape={shape}, dtype={spyre_dtype}")
        
        # Create spyre_comms tensor
        buffer_tensor = spyre_comms.Tensor(tensor_info)
        
        # Get CompositeAddress pointer from Spyre tensor
        # This returns the address as an integer that we can pass to spyre-comms
        composite_addr_ptr = torch_spyre._C.get_composite_address_ptr(x)
        
        print(f"[Rank {rank}] Got CompositeAddress pointer: 0x{composite_addr_ptr:x}")
        
        # Set the Spyre device address directly (no CPU copy)
        buffer_tensor.set_spyre_device_address(composite_addr_ptr)
        
        print(f"[Rank {rank}] Set device address, calling broadcast...")
        
        # Execute broadcast on Spyre device
        work = ctx.broadcast(buffer_tensor, src_rank)
        work.start()
        work.wait()
        
        print(f"[Rank {rank}] Spyre broadcast completed successfully")
        
        # Return the input tensor (broadcast modifies in-place)
        return x
    
    spyre_impl.impl("broadcast_", spyre_broadcast_impl)
    
    # Meta implementation for tracing
    def spyre_broadcast_meta(x, src_rank=0, group_name="default"):
        return torch.empty_like(x, device="meta")
    
    spyre_meta.impl("broadcast_", spyre_broadcast_meta)