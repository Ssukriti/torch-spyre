import os

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
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_demo()

"""
[1000840000@sukriti-2-dev torch-spyre]$ torchrun --nproc-per-node=2 examples/broadcast_demo_multirank.py
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
Rank 1/2 using device spyre
Rank 0/2 using device spyre
Rank 1 - Executing broadcast...

======================================================================
[DIRECT LOWERING] _c10d_functional.broadcast
  → Creating SpyreBroadcastFallback IR node
  → src_rank=0, group_name='default'
  → Will generate: torch.ops.spyre.broadcast(tensor, 0, 'default')
======================================================================

[DIRECT LOWERING] _c10d_functional.wait_tensor → No-op (synchronous broadcast)

======================================================================
[IR CODEGEN] SpyreBroadcastFallback.codegen()
======================================================================
  Input tensor: buf0
  src_rank: 0
  group_name: 'default'

  Generated code:
    buf1 = torch.ops.spyre.broadcast(buf0, 0, 'default')

  This will dispatch to C++ spyre_broadcast_impl() at runtime
======================================================================

Rank 0 (ROOT) - Initial tensor: tensor([42., 42., 42., 42.], device='spyre:0')
Rank 0 - Executing broadcast...

======================================================================
[DIRECT LOWERING] _c10d_functional.broadcast
  → Creating SpyreBroadcastFallback IR node
  → src_rank=0, group_name='default'
  → Will generate: torch.ops.spyre.broadcast(tensor, 0, 'default')
======================================================================

[DIRECT LOWERING] _c10d_functional.wait_tensor → No-op (synchronous broadcast)

======================================================================
[IR CODEGEN] SpyreBroadcastFallback.codegen()
======================================================================
  Input tensor: buf0
  src_rank: 0
  group_name: 'default'

  Generated code:
    buf1 = torch.ops.spyre.broadcast(buf0, 0, 'default')

  This will dispatch to C++ spyre_broadcast_impl() at runtime
======================================================================



Rank 1 - After broadcast: tensor([168., 168., 168., 168.], device='spyre:0')

[Rank 1] Output shape: torch.Size([8, 8])



Rank 0 - After broadcast: tensor([168., 168., 168., 168.], device='spyre:0')

[Rank 0] Output shape: torch.Size([8, 8])
"""
# Made with Bob
