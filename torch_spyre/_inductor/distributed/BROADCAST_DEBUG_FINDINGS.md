# Broadcast Compilation Debug Findings

## Problem Statement
Broadcast operation compiles successfully but doesn't execute at runtime:
- Rank 0: Gets 168 ✅ (correct)
- Rank 1: Gets 0 ❌ (should be 168 after broadcast)

## What Works
1. ✅ **Lowering Registration**: `torch.ops.spyre.broadcast.default` is in `spyre_lowerings` and `lowering.lowerings`
2. ✅ **No Decomposition**: No decomposition interfering with the op
3. ✅ **FX Pass**: Successfully converts `_c10d_functional.broadcast` → `torch.ops.spyre.broadcast.default`
4. ✅ **Node in Graph**: Broadcast node exists in FX graph going into Inductor
5. ✅ **IR Node Created**: `SpyreBroadcastFallback` class exists and has `register_buffer/register_operation` calls
6. ✅ **Other Spyre Ops Work**: `spyre.constant` and `spyre.empty` use same pattern and work

## What Doesn't Work
1. ❌ **Lowering Never Called**: `lower_spyre_broadcast()` function never executes
2. ❌ **run_node Never Called**: `GraphLowering.run_node()` never called for broadcast
3. ❌ **GraphLowering.run Never Called**: No `[GRAPH_LOWERING]` prints at all
4. ❌ **Broadcast Doesn't Execute**: Runtime broadcast operation never happens

## Key Observations

### Two Separate Compilations
The output shows TWO distinct compilation passes:

**Pass 1**: Graph WITHOUT broadcast (simple mul)
```
FX graph after lowering
graph():
    %l_args_0_ : torch.Tensor [num_users=1] = placeholder[target=L_args_0_]
    %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%l_args_0_, 42.0), kwargs = {})
    return (mul_tensor,)
[BEFORE INDUCTOR] Graph has broadcast node: False
[AFTER INDUCTOR] Compilation complete
```

**Pass 2**: Graph WITH broadcast
```
FX graph after lowering
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %broadcast_default : [num_users=1] = call_function[target=torch.ops.spyre.broadcast.default](args = (%y, 0, default), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%broadcast_default, 2), kwargs = {})
    return (z,)
[BEFORE INDUCTOR] Graph has broadcast node: True
[BEFORE INDUCTOR] Broadcast node: spyre.broadcast.default, args: (y, 0, 'default')
[AFTER INDUCTOR] Compilation complete
```

### GraphLowering Never Runs
Despite the broadcast node being in the graph:
- NO `[GRAPH_LOWERING]` prints (from `GraphLowering.run()` patch)
- NO `[RUN_NODE]` prints (from `GraphLowering.run_node()` patch)
- This means `GraphLowering` is never instantiated or run for this graph

### Hypothesis
The graph containing the broadcast node is being compiled through a **different code path** that:
1. Doesn't use `GraphLowering`
2. Doesn't call `run_node()`
3. Doesn't look up lowerings
4. Possibly treats custom ops as no-ops or placeholders

## Comparison with Working Ops

`spyre.constant` and `spyre.empty` work because they're called from within passes that explicitly call `graph_lowering.run_node()`:

```python
# From pass_utils.py
empty_tb = graph_lowering.run_node(empty_fx)
const_tb = graph_lowering.run_node(const_fx)
```

But broadcast is in the USER'S graph, not inserted by a pass, so it needs to be handled by Inductor's normal graph processing.

## Next Steps

Need to find:
1. **Where does the graph with broadcast go after `compile_fx`?**
2. **Why doesn't it go through `GraphLowering`?**
3. **Is there a special handler for graphs with custom ops?**
4. **How do we force it through the normal lowering path?**

Possible solutions:
1. Find where Inductor decides to skip `GraphLowering` and force it to run
2. Implement a different integration point (pre-pass that converts broadcast to something Inductor understands)
3. Use AOTAutograd's fallback mechanism properly
4. Convert broadcast to ATen ops that Inductor can handle