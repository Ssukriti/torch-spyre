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
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
