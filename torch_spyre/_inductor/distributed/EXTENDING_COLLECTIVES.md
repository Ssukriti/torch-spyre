# Extending Functional Collective Lowering to All Collectives

This document explains how to extend the broadcast implementation pattern to support all PyTorch distributed collectives (all_reduce, all_gather, reduce_scatter, etc.) with torch.compile.

## Table of Contents
1. [Overview](#overview)
2. [Architecture Pattern](#architecture-pattern)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Collective-Specific Considerations](#collective-specific-considerations)
5. [Testing Strategy](#testing-strategy)
6. [Performance Optimization](#performance-optimization)

---

## Overview

### Current State
✅ **Implemented**: `broadcast`
- FX pass converts `_c10d_functional.broadcast` → `spyre.broadcast`
- Runtime kernel uses spyre-comms
- Works with torch.compile

🚧 **To Implement**: 
- `all_reduce` (sum, prod, min, max, etc.)
- `all_gather` (gather tensors from all ranks)
- `reduce_scatter` (reduce and scatter results)
- `all_to_all` (personalized all-to-all exchange)
- `barrier` (synchronization)

### Why This Pattern Works
The key insight from the broadcast implementation:
1. **`@torch.library.custom_op`** prevents AOT Autograd from executing the op during compilation
2. **FX pass** converts c10d functional ops to spyre ops before Inductor sees them
3. **Lowering registration** tells Inductor how to generate code for spyre ops
4. **IR nodes** emit runtime calls to spyre-comms

---

## Architecture Pattern

### The 4-Layer Stack

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PyTorch User Code                                        │
│    torch.ops._c10d_functional.all_reduce(x, "sum", "default")│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FX Pass (fx_pass.py)                                     │
│    Converts: _c10d_functional.all_reduce → spyre.all_reduce │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Inductor Lowering (lowering.py)                          │
│    @register_spyre_lowering(torch.ops.spyre.all_reduce)     │
│    Creates IR node: SpyreAllReduceFallback                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Runtime Kernel (spyre_library.py)                        │
│    @torch.library.custom_op("spyre::all_reduce")            │
│    Calls spyre-comms: ctx.all_reduce(tensor, reduce_op)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Guide

### Example: Implementing `all_reduce`

#### Step 1: Define Runtime Kernel (`spyre_library.py`)

```python
@torch.library.custom_op("spyre::all_reduce", mutates_args=())
def all_reduce(
    x: torch.Tensor, 
    reduce_op: str = "sum",
    group_name: str = "default"
) -> torch.Tensor:
    """Runtime kernel for spyre.all_reduce.
    
    Performs an all-reduce operation across all ranks in the group.
    
    Args:
        x: Input tensor to reduce
        reduce_op: Reduction operation ("sum", "prod", "min", "max", "avg")
        group_name: Process group name
        
    Returns:
        Tensor with reduced values (same shape as input)
    """
    import spyre_comms
    import torch_spyre
    
    # Clone input to create output buffer
    out = x.clone()
    
    # Get CompositeAddress for spyre-comms
    composite_addr_ptr = torch_spyre._C.get_composite_address_ptr(out)
    
    # Get spyre-comms context
    ctx = spyre_comms.get_world_context()
    
    # Ensure contiguous
    if not out.is_contiguous():
        raise RuntimeError("spyre.all_reduce requires contiguous input")
    
    # Map reduce_op string to spyre_comms enum
    reduce_op_map = {
        "sum": spyre_comms.ReduceOp.SUM,
        "prod": spyre_comms.ReduceOp.PROD,
        "min": spyre_comms.ReduceOp.MIN,
        "max": spyre_comms.ReduceOp.MAX,
        "avg": spyre_comms.ReduceOp.AVG,
    }
    spyre_reduce_op = reduce_op_map.get(reduce_op, spyre_comms.ReduceOp.SUM)
    
    # Create tensor info
    shape = list(out.shape)
    dtype_map = {
        torch.float32: spyre_comms.TensorDataTypeEnum.float32,
        torch.float16: spyre_comms.TensorDataTypeEnum.float16,
        torch.bfloat16: spyre_comms.TensorDataTypeEnum.bfloat16,
        torch.int32: spyre_comms.TensorDataTypeEnum.int32,
        torch.int64: spyre_comms.TensorDataTypeEnum.int64,
    }
    spyre_dtype = dtype_map.get(out.dtype, spyre_comms.TensorDataTypeEnum.float32)
    tensor_info = spyre_comms.TensorInfo(spyre_dtype, spyre_comms.TensorShape(shape))
    
    # Create spyre_comms tensor
    buffer_tensor = spyre_comms.Tensor(tensor_info)
    buffer_tensor.set_spyre_device_address(composite_addr_ptr)
    
    # Execute all_reduce
    work = ctx.all_reduce(buffer_tensor, spyre_reduce_op)
    work.start()
    work.wait()
    
    return out

@all_reduce.register_fake
def _(
    x: torch.Tensor,
    reduce_op: str = "sum", 
    group_name: str = "default"
) -> torch.Tensor:
    """Fake implementation for shape inference during compilation."""
    # All-reduce preserves shape and dtype
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
```

#### Step 2: Add FX Pass Pattern (`fx_pass.py`)

```python
def lower_collectives(gm: fx.GraphModule):
    graph = gm.graph
    rank = _get_rank()

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        # Handle broadcast + wait_tensor pattern
        if node.target == torch.ops._c10d_functional.broadcast:
            # ... existing broadcast code ...
        
        # Handle all_reduce + wait_tensor pattern
        elif node.target == torch.ops._c10d_functional.all_reduce:
            all_reduce_node = node
            
            # Find wait_tensor users
            wait_users = [
                u for u in list(all_reduce_node.users)
                if u.op == "call_function"
                and u.target == torch.ops._c10d_functional.wait_tensor
            ]
            
            if rank == 0:
                print(">> Lowering _c10d_functional.all_reduce + wait_tensor → spyre.all_reduce")
            
            # Extract reduce_op from args
            # _c10d_functional.all_reduce(tensor, reduce_op, group_name)
            input_tensor = all_reduce_node.args[0]
            reduce_op = all_reduce_node.args[1] if len(all_reduce_node.args) > 1 else "sum"
            group_name = all_reduce_node.args[2] if len(all_reduce_node.args) > 2 else "default"
            
            # Create new spyre.all_reduce node
            with graph.inserting_after(all_reduce_node):
                new_node = graph.call_function(
                    torch.ops.spyre.all_reduce.default,
                    args=(input_tensor, reduce_op, group_name),
                    kwargs={},
                )
            
            # Replace wait_tensor with new node
            if wait_users:
                for wait_node in wait_users:
                    wait_node.replace_all_uses_with(new_node)
                    graph.erase_node(wait_node)
            else:
                all_reduce_node.replace_all_uses_with(new_node)
            
            graph.erase_node(all_reduce_node)

    graph.lint()
    gm.recompile()
    return gm
```

#### Step 3: Register Lowering (`lowering.py`)

```python
@register_spyre_lowering(torch.ops.spyre.all_reduce.default)
def lower_spyre_all_reduce(x, reduce_op="sum", group_name="default"):
    """
    Lowering for spyre.all_reduce - generates a fallback call.
    
    Creates an IR node that will emit a runtime call to torch.ops.spyre.all_reduce.
    """
    x.realize()
    return ir.TensorBox.create(
        SpyreAllReduceFallback(
            torch.ops.spyre.all_reduce.default,
            x,
            reduce_op,
            group_name,
        )
    )
```

#### Step 4: Create IR Node (`ir.py`)

```python
class SpyreAllReduceFallback(ir.ExternKernel):
    """IR node for spyre.all_reduce — emits a runtime call."""
    
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        """Generate code to call torch.ops.spyre.all_reduce at runtime."""
        input_name = self.inputs[0].codegen_reference()
        reduce_op, group_name = self.constant_args
        output_name = self.get_name()
        wrapper.writeline(
            f"{output_name} = torch.ops.spyre.all_reduce({input_name}, '{reduce_op}', '{group_name}')"
        )
    
    def should_allocate(self) -> bool:
        return False
    
    def get_mutation_names(self) -> Sequence[str]:
        return []
    
    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()
    
    def __init__(
        self,
        op_overload: torch._ops.OpOverload,
        x: IRNode,
        reduce_op: str,
        group_name: str,
    ) -> None:
        layout = x.get_layout()
        super().__init__(
            None,
            layout,
            [x],
            (reduce_op, group_name),
            python_kernel_name="torch.ops.spyre.all_reduce",
            op_overload=op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
```

#### Step 5: Add Layout Propagation (`propagate_layouts.py`)

```python
elif isinstance(op, SpyreAllReduceFallback):
    # All-reduce preserves input layout
    op.layouts = [generic_layout(op)]
    op.restick_cost_fn = AnyInNode.from_args()
```

#### Step 6: Add Work Division (`work_division.py`)

```python
elif isinstance(op, SpyreAllReduceFallback):
    # Work division not supported on all_reduce kernels
    pass
```

---

## Collective-Specific Considerations

### 1. All-Reduce
**Signature**: `all_reduce(tensor, reduce_op, group) -> tensor`
- **Input/Output**: Same shape
- **Reduce ops**: sum, prod, min, max, avg
- **Special handling**: Need to map string reduce_op to spyre_comms enum

### 2. All-Gather
**Signature**: `all_gather(tensor, group) -> tensor`
- **Input**: `[batch, ...]` on each rank
- **Output**: `[world_size, batch, ...]` (gathered along new dimension)
- **Special handling**: Output shape changes! Need to handle in fake implementation
- **Layout**: May need special layout handling for gathered dimension

```python
@all_gather.register_fake
def _(x: torch.Tensor, group_name: str = "default") -> torch.Tensor:
    # Output has extra dimension for world_size
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    output_shape = [world_size] + list(x.shape)
    return torch.empty(output_shape, dtype=x.dtype, device=x.device)
```

### 3. Reduce-Scatter
**Signature**: `reduce_scatter(tensor, reduce_op, group) -> tensor`
- **Input**: `[world_size, batch, ...]` on each rank
- **Output**: `[batch, ...]` (reduced and scattered)
- **Special handling**: Input is split along first dimension, reduced, then scattered
- **Layout**: Need to handle dimension reduction

```python
@reduce_scatter.register_fake
def _(x: torch.Tensor, reduce_op: str = "sum", group_name: str = "default") -> torch.Tensor:
    # Output loses first dimension (scattered)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if x.shape[0] != world_size:
        raise ValueError(f"First dimension must equal world_size ({world_size})")
    output_shape = list(x.shape[1:])
    return torch.empty(output_shape, dtype=x.dtype, device=x.device)
```

### 4. All-to-All
**Signature**: `all_to_all(tensor, group) -> tensor`
- **Input/Output**: Same shape `[world_size, batch, ...]`
- **Special handling**: Each rank sends different data to each other rank
- **Complexity**: Most complex collective, may need split/concat handling

### 5. Barrier
**Signature**: `barrier(group) -> None`
- **No tensor input/output**: Just synchronization
- **Special handling**: May not need full IR node, could be simpler

---

## Testing Strategy

### 1. Unit Tests
For each collective, create tests similar to `broadcast_demo_multirank.py`:

```python
# test_all_reduce_compile.py
def test_all_reduce_sum():
    def fn(x):
        y = x + 1
        y = torch.ops._c10d_functional.all_reduce(y, "sum", "default")
        y = torch.ops._c10d_functional.wait_tensor(y)
        return y * 2
    
    compiled_fn = torch.compile(fn)
    
    # Each rank starts with different value
    x = torch.ones(8, 8, device="spyre") * rank
    out = compiled_fn(x)
    
    # After all_reduce with sum, should have sum of all ranks
    expected = sum(range(world_size)) + world_size  # (rank+1) summed
    expected = expected * 2  # final multiply
    assert torch.allclose(out, torch.full_like(out, expected))
```

### 2. Integration Tests
Test combinations of collectives:

```python
def test_broadcast_then_all_reduce():
    def fn(x):
        x = torch.ops._c10d_functional.broadcast(x, 0, "default")
        x = torch.ops._c10d_functional.wait_tensor(x)
        x = x + rank  # Each rank adds its rank
        x = torch.ops._c10d_functional.all_reduce(x, "sum", "default")
        x = torch.ops._c10d_functional.wait_tensor(x)
        return x
    
    compiled_fn = torch.compile(fn)
    # Test that both collectives work together
```

### 3. Performance Tests
Compare eager vs compiled performance:

```python
def benchmark_collective(collective_fn, tensor_size):
    # Warmup
    for _ in range(10):
        collective_fn(torch.randn(tensor_size, device="spyre"))
    
    # Measure
    start = time.time()
    for _ in range(100):
        collective_fn(torch.randn(tensor_size, device="spyre"))
    elapsed = time.time() - start
    
    return elapsed / 100  # Average time per call
```

---

## Performance Optimization

### 1. Move to C++ Dispatcher (Future)
Once Python implementation is stable, move to C++ for better performance:

```cpp
// In spyre_distributed.cpp
torch::Tensor spyre_all_reduce_(
    torch::Tensor x,
    const std::string& reduce_op,
    const std::string& group_name
) {
    // Direct CompositeAddress access (no Python overhead)
    auto* composite_addr = get_composite_address(x);
    
    // Call spyre-comms directly
    auto ctx = spyre_comms::get_world_context();
    // ... implementation ...
    
    return x;
}

// Register
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
    m.impl("all_reduce", &spyre_all_reduce_);
}
```

### 2. Fusion Opportunities
Consider fusing collectives with surrounding ops:

```python
# Instead of:
y = x + 1
y = all_reduce(y, "sum")
z = y * 2

# Could fuse to:
y = x + 1
y = all_reduce(y, "sum")  # Fused with multiply
# (y * 2 happens in same kernel)
```

### 3. Overlap Communication and Computation
For large tensors, consider pipelining:
- Split tensor into chunks
- Start communication on chunk 1 while computing chunk 2
- Requires more complex IR nodes

---

## Implementation Checklist

For each new collective:

- [ ] Define `@torch.library.custom_op` in `spyre_library.py`
- [ ] Add `@register_fake` implementation for shape inference
- [ ] Add FX pass pattern in `fx_pass.py`
- [ ] Register lowering in `lowering.py`
- [ ] Create IR node class in `ir.py`
- [ ] Add layout propagation in `propagate_layouts.py`
- [ ] Add work division handling in `work_division.py`
- [ ] Create demo example in `examples/`
- [ ] Write unit tests
- [ ] Update documentation

---

## Common Pitfalls

### 1. Shape Mismatches
**Problem**: Fake implementation returns wrong shape
**Solution**: Carefully compute output shape based on collective semantics

### 2. AOT Execution
**Problem**: Op executes during compilation instead of runtime
**Solution**: Always use `@torch.library.custom_op`, never manual registration

### 3. Non-Contiguous Tensors
**Problem**: spyre-comms requires contiguous tensors
**Solution**: Add contiguity check in runtime kernel, or make contiguous

### 4. Group Name Handling
**Problem**: Group names not properly passed through
**Solution**: Ensure group_name is in constant_args tuple in IR node

### 5. Reduce Op Mapping
**Problem**: String reduce_op not mapped to spyre_comms enum
**Solution**: Create explicit mapping dict in runtime kernel

---

## Next Steps

### Priority Order
1. **all_reduce** - Most commonly used, similar to broadcast
2. **all_gather** - Needed for model parallelism
3. **reduce_scatter** - Inverse of all_gather
4. **all_to_all** - More complex, lower priority
5. **barrier** - Simple but less critical

### Long-term Goals
- Move all collectives to C++ for performance
- Add fusion support for collective + compute
- Support async collectives (non-blocking)
- Add profiling/tracing support
- Optimize for different tensor sizes

---

## References

- PyTorch c10d functional collectives: `torch/distributed/_functional_collectives.py`
- Inductor lowering: `torch/_inductor/lowering.py`
- Custom ops guide: `torch.library` documentation
- spyre-comms API: Internal spyre-comms documentation

---

**Document Version**: 1.0  
**Last Updated**: 2025-05-27  
**Author**: Based on broadcast implementation