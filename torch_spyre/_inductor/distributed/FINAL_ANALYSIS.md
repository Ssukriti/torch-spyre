# Final Analysis: Why Broadcast Compilation Fails

## Problem
C10D ops (`_c10d_functional.broadcast`, `wait_tensor`) are removed by AOT Autograd before GraphLowering, so our registered lowerings are never called.

## Evidence

### 1. Wrapper Applied But Never Called
```
[PATCHES] Before patch, fx_codegen_and_compile id: 139693413857568
[PATCHES] After patch, fx_codegen_and_compile id: 139693442097664  ← ID changed
[PATCHES] Successfully wrapped fx_codegen_and_compile at module load time
```

But we NEVER see: `[WRAPPED FX_CODEGEN HIT]`

This means `fx_codegen_and_compile` is NOT the function being called in this PyTorch version.

### 2. C10D Ops Present Before AOT, Gone After
```
[BEFORE INDUCTOR] Checking for C10D ops in graph:
  -> Found C10D op: _c10d_functional.broadcast  ✓

[CALLING _ORIG] About to call torch._inductor.compile_fx.compile_fx

[AFTER _ORIG] Returned from compile_fx
  (C10D ops are gone)  ✗
```

### 3. Lowerings Registered Correctly
```
[REGISTER_LOWERINGS] c10d_broadcast_packet in lowerings: True  ✓
[REGISTER_LOWERINGS] c10d_broadcast_default in lowerings: True  ✓
```

But never called because ops are removed before lowering phase.

## Root Cause

The compilation path in this PyTorch version does NOT go through `fx_codegen_and_compile`. It uses a different function/path that we haven't identified yet.

## Next Steps

Need to find the actual post-AOT hook point. Candidates:
1. `scheme.codegen_and_compile(...)` - inspect the `scheme` object
2. Different function name in `torch._inductor.compile_fx`
3. Hook into AOT Autograd's `fw_compiler` parameter directly
4. Patch at a different level (e.g., in `_compile_fx_main` or similar)

## Alternative Approach

Since we can't find the post-AOT hook, consider:
1. **Pre-AOT transformation**: Convert C10D ops to a custom op that AOT treats as opaque
2. **Custom AOT config**: Configure AOT to not execute/remove C10D ops
3. **Different compilation mode**: Use a mode that doesn't involve AOT Autograd

The fundamental issue is that we need to intercept the graph AFTER AOT finishes but BEFORE GraphLowering starts, and `fx_codegen_and_compile` is not that point in this PyTorch version.