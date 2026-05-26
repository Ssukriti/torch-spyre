# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import torch
from torch._inductor import lowering as inductor_lowering
from torch._inductor import ir
from torch_spyre._inductor.lowering import spyre_lowerings
from torch_spyre._inductor.ir import SpyreBroadcastFallback


def register_lowerings():
    """Register Inductor lowerings for C10D distributed operations directly.
    
    This registers lowerings for _c10d_functional.broadcast and wait_tensor
    so they can be lowered by GraphLowering without being rewritten before AOT.
    """
    print(f"[REGISTER_LOWERINGS] Registering C10D lowerings directly")
    
    # Get both the OpOverloadPacket and OpOverload versions
    # The graph may contain either depending on how it was constructed
    c10d_broadcast_packet = torch.ops._c10d_functional.broadcast
    c10d_broadcast_default = torch.ops._c10d_functional.broadcast.default
    c10d_wait_packet = torch.ops._c10d_functional.wait_tensor
    c10d_wait_default = torch.ops._c10d_functional.wait_tensor.default
    
    print(f"[REGISTER_LOWERINGS] c10d_broadcast_packet = {c10d_broadcast_packet}")
    print(f"[REGISTER_LOWERINGS] c10d_broadcast_default = {c10d_broadcast_default}")
    print(f"[REGISTER_LOWERINGS] c10d_wait_packet = {c10d_wait_packet}")
    print(f"[REGISTER_LOWERINGS] c10d_wait_default = {c10d_wait_default}")
    
    # Define the lowering functions
    def lower_c10d_wait_tensor(x):
        print(f"\n{'='*80}")
        print(f"[LOWERING] _c10d_functional.wait_tensor CALLED!")
        print(f"[LOWERING] x={x}")
        print(f"{'='*80}\n")
        # wait_tensor is a no-op for lowering purposes
        # The actual synchronization happens in the broadcast
        return x
    
    def lower_c10d_broadcast(x, src_rank=0, group_name="default"):
        print(f"\n{'='*80}")
        print(f"[LOWERING] _c10d_functional.broadcast CALLED!")
        print(f"[LOWERING] x={x}, src_rank={src_rank}, group_name={group_name}")
        print(f"{'='*80}\n")
        
        x.realize()
        
        # Create the SpyreBroadcastFallback IR node
        result = ir.TensorBox.create(
            SpyreBroadcastFallback(
                c10d_broadcast_default,  # Use .default for the IR node
                x,
                src_rank,
                group_name,
            )
        )
        
        print(f"[LOWERING] Created SpyreBroadcastFallback IR node: {result}")
        return result
    
    # Register for both OpOverloadPacket and OpOverload versions
    # This ensures we catch the op regardless of how it appears in the graph
    inductor_lowering.register_lowering(c10d_wait_packet)(lower_c10d_wait_tensor)
    inductor_lowering.register_lowering(c10d_wait_default)(lower_c10d_wait_tensor)
    inductor_lowering.register_lowering(c10d_broadcast_packet)(lower_c10d_broadcast)
    inductor_lowering.register_lowering(c10d_broadcast_default)(lower_c10d_broadcast)
    
    print(f"[REGISTER_LOWERINGS] Registered C10D lowerings successfully")
    print(f"[REGISTER_LOWERINGS] c10d_broadcast_packet in lowerings: {c10d_broadcast_packet in inductor_lowering.lowerings}")
    print(f"[REGISTER_LOWERINGS] c10d_broadcast_default in lowerings: {c10d_broadcast_default in inductor_lowering.lowerings}")
    print(f"[REGISTER_LOWERINGS] c10d_wait_packet in lowerings: {c10d_wait_packet in inductor_lowering.lowerings}")
    print(f"[REGISTER_LOWERINGS] c10d_wait_default in lowerings: {c10d_wait_default in inductor_lowering.lowerings}")
    
    # Debug: Check object identity
    print(f"\n[REGISTER_LOWERINGS] Object identity check:")
    print(f"  c10d_broadcast_packet id: {id(c10d_broadcast_packet)}")
    print(f"  c10d_broadcast_packet type: {type(c10d_broadcast_packet)}")
    print(f"  c10d_broadcast_packet == torch.ops._c10d_functional.broadcast: {c10d_broadcast_packet == torch.ops._c10d_functional.broadcast}")
    print(f"  c10d_broadcast_packet is torch.ops._c10d_functional.broadcast: {c10d_broadcast_packet is torch.ops._c10d_functional.broadcast}")
    
    # Keep all_reduce as fallback for now
    inductor_lowering.make_fallback(torch.ops.spyre.all_reduce_.default)
