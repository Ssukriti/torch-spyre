# FX graph lowering prototype


# Problem Statement 
As workloads scale across multiple AIUs, collective communication becomes a significant component of execution time. While Torch-Spyre can lower compute operations through the compiler stack, communication operations are traditionally handled outside the compiler's optimization domain. 

As a result, communication and compute follow separate execution paths, creating boundaries that can contribute to device idle time and limiting opportunities for communication-aware optimizations.

The long-term goal of this work is to reduce device idle time in multi-rank workloads by bringing collective communication into the Torch-Spyre execution model, enabling future optimizations.

# Phase 1 Goal

Keep collectives inside the Torch-Spyre compilation flow and route them through the native Spyre communication stack.


Instead of:

FX Graph
   ↓
c10d Functional Collective
   ↓
Fallback / external communication path

we now have:

FX Graph
   ↓
Spyre Collective Op
   ↓
Torch-Spyre Compilation
   ↓
C++ Dispatcher
   ↓
spyre-comms

## What Was Implemented
1. Functional Collective Recognition

Intercept functional collective operations in the FX graph.

Example:

_c10d_functional.broadcast
wait_tensor

are replaced with:

spyre::broadcast

2. Compiler Compatibility

Custom Spyre collectives remain visible during:

FX graph transformations
AOT Autograd tracing
Inductor compilation

through custom op registration and fake/meta implementations.

This allows collectives to survive the compilation pipeline rather than being removed or causing graph breaks

3. Runtime Integration

Lowered collectives are dispatched through:

Torch-Spyre
   ↓
C++ Dispatcher
   ↓
spyre-comms API

allowing execution on the native Spyre communication stack.

#### What Phase 1 Proves
Architectural Proof

Functional collectives can be represented as first-class operations in the Torch-Spyre compilation flow.

Communication no longer needs to be treated as an opaque operation outside the compiler stack.

Torch-Spyre and spyre-comms can be connected through a clean lowering boundary:

FX Graph
   ↓
Torch-Spyre Lowering
   ↓
spyre-comms

This validates the overall architecture for future collective support.

This is the prerequisite for future work around:

- asynchronous collectives
- WorkSchedule tracking
- communication/computation overlap
- memory residency hints
- scratchpad-aware scheduling
- reduced device idle time

Phase 1 establishes a compiler-visible path for functional collectives by lowering FX graph communication operations through Torch-Spyre into spyre-comms, proving the architectural foundation required for future asynchronous execution, communication/computation overlap, and reduction of device idle time.

## Implementation details:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PyTorch User Code                                        │
│    torch.ops._c10d_functional.broadcast(x, 0, "default")    │
│    torch.ops._c10d_functional.wait_tensor(x)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FX Pass (fx_pass.py)                                     │
│    Converts: _c10d_functional.broadcast + wait_tensor       │
│           → spyre.broadcast                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Python Custom Op (spyre_library.py)                      │
│    @torch.library.custom_op("spyre::broadcast")             │
│    - Defines schema for torch.compile                       │
│    - Provides fake implementation for shape inference       │
│    - NO runtime logic (delegated to C++)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Inductor Lowering (lowering.py)                          │
│    @register_spyre_lowering(torch.ops.spyre.broadcast)      │
│    Creates IR node: SpyreBroadcastFallback                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. C++ Dispatcher (spyre_distributed.cpp)                   │
│    TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m)                │
│    m.impl("broadcast", &spyre_broadcast_impl);              │
│    - Direct spyre-comms calls                               │
│    - Native tensor access via SpyreTensorImpl               │
└─────────────────────────────────────────────────────────────┘
```

torch_spyre/_inductor/distributed/fx_pass.py: 
Transforms _c10d_functional.broadcast → spyre::broadcast in FX graph

torch_spyre/_inductor/lowering.py: 
Registers lower_spyre_broadcast() to create IR node during graph lowering

torch_spyre/_inductor/ir.py:: 
Defines SpyreBroadcastFallback IR node that generates runtime call

torch_spyre/_inductor/distributed/spyre_library.py: 
Provides fake implementation for shape inference during compilation

torch_spyre/csrc/spyre_distributed.cpp: 
C++ dispatcher implementation using spyre-comms API

examples/broadcast_demo_multirank.py: 
Demo showing broadcast working with torch.compile


# Phase 2 - Add async implementation with broadcast 

collective_async starts WorkSchedule → returns immediately.
wait_tensor / wait_work synchronizes later

A map will contain tensor to workschedule mapping to keep track of work that is pending.
pending_work_map_[output.device_ptr()] = work;

# Future work after Phase 2:

1. Add scratchpad / residency hints - need to design 
2. Expand to more collectives
3. Explore overlap between compute and communication using streams and identify other optimizations