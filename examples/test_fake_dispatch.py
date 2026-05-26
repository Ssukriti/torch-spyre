#!/usr/bin/env python3
"""
Test script to diagnose why @register_fake isn't being called for spyre.broadcast
"""

import torch
import torch.distributed as dist
import os

# Initialize torch-spyre
import torch_spyre

print("="*80)
print("Testing Fake Dispatch for spyre.broadcast")
print("="*80)

# Initialize distributed
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

dist.init_process_group(backend='gloo', rank=0, world_size=1)

# Create a simple spyre tensor
device = torch.device('spyre')
x = torch.randn(4, 4, device=device)

print(f"\n1. Testing direct call to spyre.broadcast:")
print(f"   Input: {x.shape}, {x.dtype}, {x.device}")

try:
    result = torch.ops.spyre.broadcast(x, src_rank=0, group_name="default")
    print(f"   Result: {result.shape}, {result.dtype}, {result.device}")
    print(f"   ✓ Direct call succeeded")
except Exception as e:
    print(f"   ✗ Direct call failed: {e}")

print(f"\n2. Testing with torch.compile:")

def broadcast_fn(t):
    return torch.ops.spyre.broadcast(t, src_rank=0, group_name="default")

try:
    compiled_fn = torch.compile(broadcast_fn, backend="inductor")
    print(f"   Compilation started...")
    result = compiled_fn(x)
    print(f"   Result: {result.shape}, {result.dtype}, {result.device}")
    print(f"   ✓ Compiled call succeeded")
except Exception as e:
    print(f"   ✗ Compiled call failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n3. Testing fake tensor mode directly:")
from torch._subclasses.fake_tensor import FakeTensorMode

with FakeTensorMode():
    fake_x = torch.randn(4, 4, device=device)
    print(f"   Fake tensor: {fake_x.shape}, {fake_x.dtype}, {fake_x.device}")
    print(f"   Is fake: {isinstance(fake_x, torch._subclasses.fake_tensor.FakeTensor)}")
    
    try:
        fake_result = torch.ops.spyre.broadcast(fake_x, src_rank=0, group_name="default")
        print(f"   Fake result: {fake_result.shape}, {fake_result.dtype}, {fake_result.device}")
        print(f"   ✓ Fake mode succeeded")
    except Exception as e:
        print(f"   ✗ Fake mode failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("Diagnostic complete")
print("="*80)

dist.destroy_process_group()

# Made with Bob
