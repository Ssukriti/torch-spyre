#!/usr/bin/env python3
"""Test broadcast operation in eager mode (no torch.compile)."""

import os
import torch
import torch.distributed as dist

# Initialize torch-spyre
import torch_spyre

def test_eager_broadcast():
    """Test broadcast in eager mode without compilation."""
    # Initialize distributed
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Initialize spyre-comms library
    import spyre_comms
    spyre_comms.initialize_library()
    
    # Set device
    device = torch.device("spyre:0")
    
    print(f"\n=== EAGER MODE TEST (Rank {rank}/{world_size}) ===")
    
    # Create tensor (must be at least 128 bytes for spyre-comms)
    # 128 bytes / 4 bytes per float32 = 32 elements minimum
    if rank == 0:
        x = torch.ones(64, device=device) * 42.0
        print(f"Rank {rank} (ROOT) - Initial tensor (first 4): {x[:4]}")
    else:
        x = torch.zeros(64, device=device)
        print(f"Rank {rank} - Initial tensor (first 4): {x[:4]}")
    
    # Call broadcast directly (eager mode, no compilation)
    print(f"Rank {rank} - Calling torch.ops.spyre.broadcast_ directly...")
    result = torch.ops.spyre.broadcast_(x, src_rank=0, group_name="default")
    
    print(f"Rank {rank} - After broadcast (first 4): {result[:4]}")
    print(f"Rank {rank} - Tensor is same object: {result is x}")
    print(f"Rank {rank} - Expected all values to be 42.0")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_eager_broadcast()

# Made with Bob
