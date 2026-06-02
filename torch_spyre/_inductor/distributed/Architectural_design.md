# Direct c10d Lowering Prototype

## Problem Statement 
As workloads scale across multiple AIUs, collective communication becomes a significant component of execution time. While Torch-Spyre can lower compute operations through the compiler stack, communication operations are traditionally handled outside the compiler's optimization domain. 

As a result, communication and compute follow separate execution paths, creating boundaries that can contribute to device idle time and limiting opportunities for communication-aware optimizations.

The long-term goal of this work is to reduce device idle time in multi-rank workloads by bringing collective communication into the Torch-Spyre execution model, enabling future optimizations.

## Phase 1 Goal

Keep collectives inside the Torch-Spyre compilation flow and route them through the native Spyre communication stack.

Instead of:

```
FX Graph
   ↓
c10d Functional Collective
   ↓
Fallback / external communication path
```

we now have:

```
FX Graph
   ↓
Direct Lowering
   ↓
Torch-Spyre Compilation
   ↓
C++ Dispatcher
   ↓
spyre-comms
```

## What Was Implemented

### 1. Direct c10d Lowering

PyTorch's `_c10d_functional.broadcast` is lowered directly to an IR node without FX graph transformation.

Example:
```python
torch.ops._c10d_functional.broadcast(x, 0, "default")
```

is lowered directly to `SpyreBroadcastFallback` IR node, which generates:
```python
torch.ops.spyre.broadcast(x, 0, "default")
```

### 2. Compiler Compatibility

Custom Spyre collectives remain visible during:
- AOT Autograd tracing
- Inductor compilation

through custom op registration and fake/meta implementations.

This allows collectives to survive the compilation pipeline rather than being removed or causing graph breaks.

### 3. Runtime Integration

Lowered collectives are dispatched through:

```
Torch-Spyre
   ↓
C++ Dispatcher
   ↓
spyre-comms API
```

allowing execution on the native Spyre communication stack.

## What Phase 1 Proves

**Architectural Proof**: Functional collectives can be represented as first-class operations in the Torch-Spyre compilation flow.

Communication no longer needs to be treated as an opaque operation outside the compiler stack.

Torch-Spyre and spyre-comms can be connected through a clean lowering boundary:

```
FX Graph (_c10d_functional.broadcast)
   ↓
Direct Lowering (no FX pass)
   ↓
IR Node (SpyreBroadcastFallback)
   ↓
Generated Code (torch.ops.spyre.broadcast)
   ↓
C++ Dispatcher
   ↓
spyre-comms
```

This validates the overall architecture for future collective support.

This is the prerequisite for future work around:
- asynchronous collectives
- WorkSchedule tracking
- communication/computation overlap
- memory residency hints
- scratchpad-aware scheduling
- reduced device idle time

## Implementation Details

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PyTorch User Code                                        │
│    torch.ops._c10d_functional.broadcast(x, 0, "default")    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Direct Lowering (lowering.py)                            │
│    @register_spyre_lowering(_c10d_functional.broadcast)     │
│    Creates IR node: SpyreBroadcastFallback                  │
│    NO FX pass transformation                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. IR Node (ir.py)                                          │
│    SpyreBroadcastFallback.codegen()                         │
│    Generates: torch.ops.spyre.broadcast(x, 0, "default")    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Custom Op Registration (spyre_library.py)               │
│    @torch.library.custom_op("spyre::broadcast")             │
│    - Defines schema for PyTorch dispatcher                  │
│    - Provides fake implementation for shape inference       │
│    - NO runtime logic (delegated to C++)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. C++ Dispatcher (spyre_distributed.cpp)                   │
│    TORCH_LIBRARY(spyre, m)                                  │
│    TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m)                │
│    m.impl("broadcast", &spyre_broadcast_impl);              │
│    - Direct spyre-comms calls                               │
│    - Native tensor access via SpyreTensorImpl               │
└─────────────────────────────────────────────────────────────┘
```

### Key Files

**torch_spyre/_inductor/lowering.py**: 
Registers `lower_c10d_broadcast_direct()` to create IR node during graph lowering

**torch_spyre/_inductor/ir.py**: 
Defines `SpyreBroadcastFallback` IR node that generates runtime call

**torch_spyre/_inductor/distributed/spyre_library.py**: 
Provides custom op registration and fake implementation for shape inference

**torch_spyre/csrc/distributed/spyre_distributed.cpp**: 
C++ dispatcher implementation using spyre-comms API

**examples/broadcast_demo_multirank.py**: 
Demo showing broadcast working with torch.compile

## Phase 2 - Async Implementation (Future Work)

`collective_async` starts WorkSchedule → returns immediately.
`wait_tensor` / `wait_work` synchronizes later.

A map will contain tensor to workschedule mapping to keep track of work that is pending:
```cpp
pending_work_map_[output.device_ptr()] = work;
```

## Future Work After Phase 2

1. Add scratchpad / residency hints
2. Expand to more collectives (all_reduce, all_gather, reduce_scatter)
3. Explore overlap between compute and communication using streams
4. Identify other optimizations for reduced device idle time