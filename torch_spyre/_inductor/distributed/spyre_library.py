import torch
from torch.library import Library

lib = Library("spyre", "DEF")

# Define all_reduce WITHOUT aliasing annotations to work with AOT Autograd
# AOT Autograd's functionalization doesn't support custom ops with aliasing
lib.define(
    "all_reduce_(Tensor x, str reduce_op='sum', str group_name='default') -> Tensor"
)

# Define broadcast as FUNCTIONAL (no underscore) without aliasing annotations
# This works with AOT Autograd's functionalization pass
lib.define(
    "broadcast(Tensor x, int src_rank=0, str group_name='default') -> Tensor"
)

# Register fake/abstract implementation for AOT Autograd
# This tells AOT what the output shape/dtype will be WITHOUT executing the op
# This keeps the op in the graph for GraphLowering to handle
print(f"[MODULE LOAD] Registering fake implementation for spyre::broadcast")

@torch.library.register_fake("spyre::broadcast")
def spyre_broadcast_fake(x, src_rank=0, group_name="default"):
    """Fake implementation for AOT Autograd shape inference.
    
    This provides metadata (shape, dtype, stride) without executing the broadcast.
    Critically, this prevents AOT from executing and removing the op from the graph.
    """
    print(f"\n{'='*80}")
    print(f"[BROADCAST_FAKE] CALLED FOR SHAPE INFERENCE!")
    print(f"[BROADCAST_FAKE] x.shape={x.shape}, x.dtype={x.dtype}, x.stride()={x.stride()}")
    print(f"[BROADCAST_FAKE] x.device={x.device}, x.type={type(x).__name__}")
    print(f"[BROADCAST_FAKE] Returning fake tensor with same metadata")
    print(f"{'='*80}\n")
    # Return a tensor with the same shape, stride, dtype, and device
    # This tells AOT what the output will look like without actually broadcasting
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)

print(f"[MODULE LOAD] Fake implementation registered for spyre::broadcast")