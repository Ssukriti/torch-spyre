import torch
from torch.library import Library

lib = Library("spyre", "DEF")

# Define all_reduce WITHOUT aliasing annotations to work with AOT Autograd
# AOT Autograd's functionalization doesn't support custom ops with aliasing
lib.define(
    "all_reduce_(Tensor x, str reduce_op='sum', str group_name='default') -> Tensor"
)

# CRITICAL: Use @torch.library.custom_op for proper FunctionalTensor handling
# This automatically ensures:
# 1. Runtime kernel only receives eager tensors (no FunctionalTensor)
# 2. Fake implementation is GUARANTEED to be called during torch.compile/AOT
# 3. Op stays in graph for GraphLowering to handle
print(f"[MODULE LOAD] Registering broadcast as custom_op with automatic functionalization")

@torch.library.custom_op("spyre::broadcast", mutates_args=())
def broadcast(x: torch.Tensor, src_rank: int = 0, group_name: str = "default") -> torch.Tensor:
    """Runtime kernel for spyre.broadcast - only called with eager tensors.
    
    This is the actual runtime implementation that will be called by the
    generated code. It will NEVER receive FunctionalTensor wrappers.
    
    During compilation, the @register_fake implementation below is used instead.
    """
    print(f"\n{'='*80}")
    print(f"[RUNTIME KERNEL] spyre.broadcast executing")
    print(f"[RUNTIME KERNEL] x.device={x.device}, src_rank={src_rank}")
    print(f"[RUNTIME KERNEL] x.shape={x.shape}, x.dtype={x.dtype}")
    print(f"{'='*80}\n")
    
    # Import here to avoid circular dependencies
    import spyre_comms
    import torch_spyre
    
    # Clone input to create output buffer (functional semantics)
    out = x.clone()
    
    # Get CompositeAddress from output for broadcast operation
    composite_addr_ptr = torch_spyre._C.get_composite_address_ptr(out)
    
    # Get spyre-comms context
    ctx = spyre_comms.get_world_context()
    rank = ctx.get_rank()
    
    print(f"[Rank {rank}] Performing broadcast using spyre-comms")
    
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
    
    # Create spyre_comms tensor
    buffer_tensor = spyre_comms.Tensor(tensor_info)
    
    # Set the Spyre device address directly (no CPU copy)
    buffer_tensor.set_spyre_device_address(composite_addr_ptr)
    
    # Execute broadcast on Spyre device (modifies out in-place)
    work = ctx.broadcast(buffer_tensor, src_rank)
    work.start()
    work.wait()
    
    print(f"[Rank {rank}] Broadcast completed successfully")
    
    # Return the output tensor with broadcasted data
    return out

@broadcast.register_fake
def _(x: torch.Tensor, src_rank: int = 0, group_name: str = "default") -> torch.Tensor:
    """Fake implementation for torch.compile / AOT Autograd shape inference.
    
    This is GUARANTEED to be called during compilation instead of the runtime kernel.
    It provides shape/dtype metadata without executing the actual broadcast.
    """
    print(f"\n{'='*80}")
    print(f"[BROADCAST_FAKE] CALLED SUCCESSFULLY FOR SHAPE INFERENCE!")
    print(f"[BROADCAST_FAKE] x.shape={x.shape}, x.dtype={x.dtype}, x.stride()={x.stride()}")
    print(f"[BROADCAST_FAKE] x.device={x.device}, x.type={type(x).__name__}")
    print(f"[BROADCAST_FAKE] Returning fake tensor with same metadata")
    print(f"{'='*80}\n")
    # Return a tensor with the same shape, stride, dtype, and device
    # This tells AOT what the output will look like without actually broadcasting
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)

print(f"[MODULE LOAD] custom_op broadcast registered with fake implementation")