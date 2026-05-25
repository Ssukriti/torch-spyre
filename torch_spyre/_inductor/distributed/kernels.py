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
    # Broadcast operation - using spyre-comms on Spyre tensors
    # ------------------------------------------------------------
    
    def is_compile_time_tensor(x):
        """Check if tensor is being used during compilation/tracing (not runtime)
        
        This includes:
        - Meta tensors (device='meta')
        - FakeTensors (used by AOT Autograd)
        - FunctionalTensors (used by AOT Autograd functionalization)
        - Tensors with null storage (fake tensors with real device)
        """
        # Check for meta device
        if x.device.type == "meta":
            return True
        
        # Check for FakeTensor by type name
        if type(x).__name__ == "FakeTensor":
            return True
        
        # Check for FunctionalTensor by type name (AOT Autograd functionalization)
        if type(x).__name__ == "FunctionalTensor":
            return True
        
        # Check for fake_mode attribute
        if getattr(x, "fake_mode", None) is not None:
            return True
        
        # Check for null storage pointer (fake tensor with real device)
        try:
            if x.untyped_storage().data_ptr() == 0:
                return True
        except Exception:
            # If we can't check storage, assume it might be compile-time
            pass
        
        return False
    
    def spyre_broadcast_impl(x, src_rank=0, group_name="default"):
        """Spyre broadcast implementation using spyre-comms library
        
        Handles both compile-time (tracing) and runtime execution:
        - During tracing: Returns empty_like for shape propagation
        - At runtime: Performs actual broadcast using spyre-comms
        
        Functional broadcast: clones input, broadcasts into clone, returns clone.
        This matches the functional schema: broadcast(Tensor x) -> Tensor
        """
        import torch
        
        print(f"\n{'='*80}")
        print(f"[BROADCAST_IMPL] KERNEL CALLED!")
        print(f"[BROADCAST_IMPL] x.device={x.device}, src_rank={src_rank}")
        print(f"[BROADCAST_IMPL] x.type={type(x).__name__}")
        try:
            print(f"[BROADCAST_IMPL] x.dispatch_keys={torch._C._dispatch_keys(x)}")
        except Exception as e:
            print(f"[BROADCAST_IMPL] Could not get dispatch keys: {e}")
        
        # COMMENTED OUT: x.clone() also causes AOT to replace the op
        # Neither empty_like() nor clone() preserve the op through AOT
        # The real fix is to not rewrite _c10d_functional.broadcast before AOT
        
        # Special handling for FunctionalTensor (AOT Autograd functionalization)
        if type(x).__name__ == "FunctionalTensor":
            print(f"[BROADCAST_IMPL] FUNCTIONALIZATION PATH: Detected FunctionalTensor")
            print(f"[BROADCAST_IMPL] Returning x.clone() to preserve op in graph")
            return x.clone()
        
        # Check if this is compile-time tensor (including FunctionalTensor)
        if is_compile_time_tensor(x):
            print(f"[BROADCAST_IMPL] COMPILE-TIME PATH: Detected fake/meta tensor")
            print(f"[BROADCAST_IMPL] Returning empty_like for shape propagation")
            print(f"[BROADCAST_IMPL] This means AOT Autograd is tracing, not lowering yet")
            print(f"{'='*80}\n")
            return torch.empty_like(x)
        
        print(f"[BROADCAST_IMPL] RUNTIME PATH: Real tensor detected")
        print(f"[BROADCAST_IMPL] Performing actual broadcast using spyre-comms")
        print(f"{'='*80}\n")
        
        # Real runtime execution starts here
        import spyre_comms
        import torch_spyre
        
        # Real runtime execution starts here - clone input to create output buffer
        out = x.clone()
        
        # Get CompositeAddress from output for broadcast operation
        composite_addr_ptr = torch_spyre._C.get_composite_address_ptr(out)
        
        # Real runtime execution starts here
        ctx = spyre_comms.get_world_context()
        rank = ctx.get_rank()
        
        print(f"[Rank {rank}] Spyre broadcast called - using spyre-comms")
        
        # Ensure output tensor is contiguous
        if not out.is_contiguous():
            raise RuntimeError("spyre.broadcast currently requires contiguous input")
        
        # Get tensor shape and dtype
        shape = list(out.shape)
        
        # Map torch dtype to spyre_comms dtype
        dtype_map = {
            torch.float32: spyre_comms.TensorDataTypeEnum.float32,
            torch.float16: spyre_comms.TensorDataTypeEnum.float16,
            torch.bfloat16: spyre_comms.TensorDataTypeEnum.bfloat16,
            torch.int32: spyre_comms.TensorDataTypeEnum.int32,
            torch.int64: spyre_comms.TensorDataTypeEnum.int64,
        }
        
        spyre_dtype = dtype_map.get(out.dtype, spyre_comms.TensorDataTypeEnum.float32)
        tensor_info = spyre_comms.TensorInfo(spyre_dtype, spyre_comms.TensorShape(shape))
        
        print(f"[Rank {rank}] Creating spyre_comms tensor with shape={shape}, dtype={spyre_dtype}")
        
        # Create spyre_comms tensor
        buffer_tensor = spyre_comms.Tensor(tensor_info)
        
        print(f"[Rank {rank}] Got CompositeAddress pointer: 0x{composite_addr_ptr:x}")
        
        # Set the Spyre device address directly (no CPU copy)
        buffer_tensor.set_spyre_device_address(composite_addr_ptr)
        
        print(f"[Rank {rank}] Set device address, calling broadcast...")
        
        # Execute broadcast on Spyre device (modifies out in-place)
        work = ctx.broadcast(buffer_tensor, src_rank)
        work.start()
        work.wait()
        
        print(f"[Rank {rank}] Spyre broadcast completed successfully")
        
        # Return the output tensor with broadcasted data (functional semantics)
        return out
    
    # Register Python implementation for runtime execution
    # This will be called by the fallback generated by Inductor lowering
    spyre_impl.impl("broadcast", spyre_broadcast_impl)
    
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