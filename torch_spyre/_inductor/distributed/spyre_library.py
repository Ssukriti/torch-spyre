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

# Async broadcast operation - returns immediately
@torch.library.custom_op("spyre::broadcast_async", mutates_args=())
def broadcast_async(x: torch.Tensor, src_rank: int = 0, group_name: str = "default") -> torch.Tensor:
    """Async broadcast operation - returns immediately, communication in background.
    
    C++ implementation in torch_spyre/csrc/distributed/spyre_distributed.cpp
    
    Usage:
        output = torch.ops.spyre.broadcast_async(input, 0, "default")
        torch.ops.spyre.wait_work(output)  # Synchronize before using result
    """
    raise RuntimeError(
        "This should never be called - C++ dispatcher should handle all calls."
    )

@broadcast_async.register_fake
def _(x: torch.Tensor, src_rank: int = 0, group_name: str = "default") -> torch.Tensor:
    """Fake implementation for shape inference during compilation."""
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)


# Wait for async operation to complete
@torch.library.custom_op("spyre::wait_work", mutates_args=())
def wait_work(x: torch.Tensor) -> torch.Tensor:
    """Wait for async operation to complete.
    
    C++ implementation in torch_spyre/csrc/distributed/spyre_distributed.cpp
    
    Usage:
        output = torch.ops.spyre.broadcast_async(input, 0, "default")
        torch.ops.spyre.wait_work(output)  # Block until broadcast completes
    """
    raise RuntimeError(
        "This should never be called - C++ dispatcher should handle all calls."
    )

@wait_work.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    """Fake implementation - pass through the tensor."""
    return x
