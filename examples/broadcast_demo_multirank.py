import os
# Disable precompiled headers to avoid openssl dependency
os.environ.setdefault("TORCHINDUCTOR_CPP_WRAPPER_PRECOMPILE_HEADERS", "0")

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


def run_demo():
    device = torch.device("spyre")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    root_rank = 0

    print(f"Rank {rank}/{world_size} using device {device}")

    c10d._register_process_group("default", dist.group.WORLD)

    # Initialize spyre-comms library
    import spyre_comms
    spyre_comms.initialize_library()

    # Create tensor - must be at least 128 bytes for spyre-comms
    # Using 8x8 float32 = 256 bytes (meets minimum requirement)
    if rank == root_rank:
        x = torch.ones(8, 8).to(device) * 42.0
        print(f"Rank {rank} (ROOT) - Initial tensor: {x[0, :4]}")
    else:
        x = torch.zeros(8, 8).to(device)
        #print(f"Rank {rank} - Initial tensor: {x[0, :4]}")

    def fn(t):
        # Perform computation
        y = t + t
        
        # Broadcast from root rank - this will be lowered to spyre.broadcast
        y = torch.ops._c10d_functional.broadcast(y, root_rank, "default")
        y = torch.ops._c10d_functional.wait_tensor(y)
        
        # Continue computation
        z = y * 2
        return z

    #print(f"Rank {rank} - Compiling function...")
    compiled_fn = torch.compile(fn)

    print(f"Rank {rank} - Executing broadcast...")
    out = compiled_fn(x)

    print("\n")
    print(f"Rank {rank} - After broadcast: {out[0, :4]}")
    print(f"\n[Rank {rank}] Output shape: {out.shape}\n")

    # Cleanup
    spyre_comms.finalize_library()
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_demo()


"""
## To run
torchrun --nproc-per-node=2 examples/broadcast_demo_multirank.py

## Expected FX graph lowering
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
Rank 1/2 using device spyre
Rank 0/2 using device spyre
Rank 1 - Executing broadcast...

======SpyreBroadcastFallback.codegen called======
[CODEGEN] Generated: buf1 = torch.ops.spyre.broadcast(buf0, 0, 'default')
Rank 0 (ROOT) - Initial tensor: tensor([42., 42., 42., 42.], device='spyre:0')
Rank 0 - Executing broadcast...

=== FX GRAPH BEFORE LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %y_1 : [num_users=1] = call_function[target=torch.ops._c10d_functional.broadcast](args = (%y, 0, default), kwargs = {})
    %y_2 : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor](args = (%y_1,), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%y_2, 2), kwargs = {})
    return (z,)

=== FX GRAPH LOWERING ===
>> Lowering _c10d_functional.broadcast + wait_tensor → spyre.broadcast

=== FX GRAPH AFTER LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %broadcast_default : [num_users=1] = call_function[target=torch.ops.spyre.broadcast.default](args = (%y, 0, default), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%broadcast_default, 2), kwargs = {})
    return (z,)

======SpyreBroadcastFallback.codegen called======
[CODEGEN] Generated: buf1 = torch.ops.spyre.broadcast(buf0, 0, 'default')


Rank 1 - After broadcast: tensor([168., 168., 168., 168.], device='spyre:0')

[Rank 1] Output shape: torch.Size([8, 8])

Rank 0 - After broadcast: tensor([168., 168., 168., 168.], device='spyre:0')

[Rank 0] Output shape: torch.Size([8, 8])
"""
