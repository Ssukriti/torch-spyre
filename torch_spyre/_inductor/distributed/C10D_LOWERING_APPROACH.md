# Direct C10D Lowering Approach

## Problem
The previous approach of rewriting `_c10d_functional.broadcast` → `torch.ops.spyre.broadcast` before AOT Autograd caused the operation to be executed during tracing and removed from the graph before GraphLowering could process it.

## Solution
Register lowerings for the C10D ops directly in the Inductor lowering table, so GraphLowering can handle them without any FX graph rewriting.

## Changes Made

### 1. torch_spyre/_inductor/distributed/lowerings.py
- **Removed**: Spyre broadcast lowering registration
- **Added**: Direct C10D lowering registration:
  - `@inductor_lowering.register_lowering(torch.ops._c10d_functional.broadcast.default)`
  - `@inductor_lowering.register_lowering(torch.ops._c10d_functional.wait_tensor.default)`
- Both lowerings include extensive debug output to confirm they are called
- `lower_c10d_broadcast` creates `SpyreBroadcastFallback` IR node
- `lower_c10d_wait_tensor` is a passthrough (returns input)

### 2. torch_spyre/_inductor/__init__.py
- **Disabled**: `gm = lower_collectives(gm)` call (line 149 commented out)
- **Added**: Debug output to check for C10D ops in the graph before Inductor
- This allows `_c10d_functional.broadcast` and `wait_tensor` to pass through to GraphLowering unchanged

## Expected Behavior

### Graph Flow:
1. **AOT Autograd**: Sees `_c10d_functional.broadcast` and `wait_tensor` but doesn't execute them (no custom implementation registered for these ops)
2. **GraphLowering**: Receives graph with C10D ops intact
3. **Lowering Phase**: Calls our registered lowerings:
   - `lower_c10d_broadcast` → creates `SpyreBroadcastFallback` IR node
   - `lower_c10d_wait_tensor` → passthrough
4. **Code Generation**: `SpyreBroadcastFallback.codegen()` emits runtime call to `torch.ops.spyre.broadcast_impl`

### Debug Signals to Look For:
```
[REGISTER_LOWERINGS] Registering C10D lowerings directly
[REGISTER_LOWERINGS] c10d_broadcast in lowerings: True
[REGISTER_LOWERINGS] c10d_wait in lowerings: True

[BEFORE INDUCTOR] Checking for C10D ops in graph:
  -> Found C10D op: <OpOverload(op='_c10d_functional.broadcast', overload='default')>
  -> Is broadcast.default: True

[LOWERING] _c10d_functional.broadcast CALLED!
[LOWERING] _c10d_functional.wait_tensor CALLED!
```

## Test Command
```bash
rm -rf ~/.triton/cache /tmp/torchinductor_*
torchrun --nproc-per-node=2 examples/broadcast_demo_multirank.py
```

## Success Criteria
- Both ranks should print the correct broadcasted value (168)
- Debug output should show lowerings being called
- No "NaN" or "0" values on Rank 1

## Fallback Plan
If this doesn't work, we may need to:
1. Register a fake/meta implementation for C10D ops to prevent AOT from trying to execute them
2. Or find a hook point between AOT and GraphLowering to inject the FX pass