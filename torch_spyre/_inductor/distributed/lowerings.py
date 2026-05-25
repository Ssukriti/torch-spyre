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
from torch._inductor import lowering as lowering
from torch_spyre._inductor.lowering import spyre_lowerings


def register_lowerings():
    """Register Inductor lowerings for Spyre distributed operations."""
    # TEMPORARILY COMMENTED OUT FOR DEBUGGING
    # make_fallback may be overriding our custom lower_spyre_broadcast
    # We want to test if our custom lowering works without the fallback
    
    print(f"[REGISTER_LOWERINGS] Called")
    print(f"[REGISTER_LOWERINGS] TEMPORARILY NOT calling make_fallback for broadcast")
    print(f"[REGISTER_LOWERINGS] Testing if custom lowering works without fallback")
    
    # COMMENTED OUT: This may override our custom lowering
    # lowering.make_fallback(torch.ops.spyre.broadcast.default)
    
    # Keep all_reduce as fallback for now
    lowering.make_fallback(torch.ops.spyre.all_reduce_.default)

# Made with Bob
