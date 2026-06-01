import torch
from torch.library import Library

lib = Library("spyre", "DEF")

# Define all_reduce WITHOUT aliasing annotations to work with AOT Autograd
# AOT Autograd's functionalization doesn't support custom ops with aliasing
lib.define(
    "all_reduce_(Tensor x, str reduce_op='sum', str group_name='default') -> Tensor"
)

# Use @torch.library.custom_op for broadcast with C++ dispatcher
# This provides:
# 1. Proper schema registration
# 2. Automatic fake kernel registration
# 3. Better integration with torch.compile
# 4. C++ implementation via TORCH_LIBRARY_IMPL in spyre_distributed.cpp
@torch.library.custom_op("spyre::broadcast", mutates_args=())
def broadcast(x: torch.Tensor, src_rank: int = 0, group_name: str = "default") -> torch.Tensor:
    """Broadcast operation - C++ implementation in torch_spyre/csrc/spyre_distributed.cpp
    
    This function signature is used for:
    1. Schema definition and type checking
    2. Python-side calls in eager mode (dispatches to C++)
    3. Documentation
    
    The actual runtime implementation is in C++ via TORCH_LIBRARY_IMPL(spyre, PrivateUse1).
    When this op is called, PyTorch's dispatcher automatically routes to the C++ implementation.
    """
    # This body is never executed - C++ dispatcher handles all calls
    # But we need a body for the decorator to work
    raise RuntimeError(
        "This should never be called - C++ dispatcher should handle all calls. "
        "Check that spyre_distributed.cpp is compiled and TORCH_LIBRARY_IMPL is registered."
    )

@broadcast.register_fake
def _(x: torch.Tensor, src_rank: int = 0, group_name: str = "default") -> torch.Tensor:
    """Fake implementation for torch.compile / AOT Autograd shape inference.
    
    This is GUARANTEED to be called during compilation instead of the C++ runtime kernel.
    It provides shape/dtype metadata without executing the actual broadcast.
    """
    # Return a tensor with the same shape, stride, dtype, and device
    # This tells AOT what the output will look like without actually broadcasting

# ============================================================================
# Async Communication Operations for Communication-Compute Overlap
# ============================================================================

@torch.library.custom_op("spyre::broadcast_async", mutates_args=())
def broadcast_async(x: torch.Tensor, src_rank: int = 0) -> int:
    """Async broadcast operation - starts communication and returns handle immediately.
    
    This enables communication-compute overlap by:
    1. Starting the broadcast operation (non-blocking)
    2. Returning an integer handle immediately
    3. Allowing independent compute to execute while communication is in flight
    4. Requiring explicit wait() call to ensure completion
    
    C++ implementation in torch_spyre/csrc/spyre_distributed.cpp
    
    Args:
        x: Tensor to broadcast
        src_rank: Source rank for broadcast
        
    Returns:
        int: Handle to the WorkSchedule (opaque integer ID)
        
    Example:
        >>> handle = torch.ops.spyre.broadcast_async(tensor, src=0)
        >>> # ... independent compute here ...
        >>> result = torch.ops.spyre.wait(handle, tensor)
    """
    raise RuntimeError(
        "This should never be called - C++ dispatcher should handle all calls. "
        "Check that spyre_distributed.cpp is compiled and TORCH_LIBRARY_IMPL is registered."
    )

@broadcast_async.register_fake
def _(x: torch.Tensor, src_rank: int = 0) -> int:
    """Fake implementation for torch.compile / AOT Autograd.
    
    Returns a dummy handle (0) during compilation for shape inference.
    The actual handle is only meaningful at runtime.
    """
    return 0  # Dummy handle for compilation

@torch.library.custom_op("spyre::wait", mutates_args=())
def wait(handle: int, tensor: torch.Tensor) -> torch.Tensor:
    """Wait for async communication operation to complete.
    
    Blocks until the WorkSchedule identified by handle has completed,
    then returns the tensor containing the communication result.
    
    C++ implementation in torch_spyre/csrc/spyre_distributed.cpp
    
    Args:
        handle: WorkSchedule handle from broadcast_async()
        tensor: Tensor that was passed to broadcast_async()
        
    Returns:
        torch.Tensor: The input tensor (now contains completed communication result)
        
    Example:
        >>> handle = torch.ops.spyre.broadcast_async(tensor, src=0)
        >>> # ... independent compute here ...
        >>> result = torch.ops.spyre.wait(handle, tensor)
    """
    raise RuntimeError(
        "This should never be called - C++ dispatcher should handle all calls. "
        "Check that spyre_distributed.cpp is compiled and TORCH_LIBRARY_IMPL is registered."
    )

@wait.register_fake
def _(handle: int, tensor: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.compile / AOT Autograd.
    
    Returns the input tensor unchanged during compilation for shape inference.
    """
    return torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device)
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
