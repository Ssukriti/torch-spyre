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

import os
import sys

# CRITICAL: Set this before ANY torch imports to disable precompiled headers
os.environ["TORCHINDUCTOR_CPP_WRAPPER_PRECOMPILE_HEADERS"] = "0"

# Monkey-patch codecache BEFORE any other torch imports
try:
    from torch._inductor import codecache
    _original_precompile_header = getattr(codecache, '_precompile_header', None)
    if _original_precompile_header:
        def _patched_precompile_header(*args, **kwargs):
            return None
        codecache._precompile_header = _patched_precompile_header
        print("Successfully patched _precompile_header to avoid openssl dependency", file=sys.stderr)
except Exception as e:
    print(f"Warning: Could not patch _precompile_header: {e}", file=sys.stderr)

from .constants import DEVICE_NAME
from .patches import enable_spyre_context
from . import config

import threading
from functools import wraps
from torch_spyre._inductor.distributed import lower_collectives

_autoload_lock = threading.Lock()


def enable_spyre_compile_fx_wrapper():
    import torch._inductor.compile_fx as cfx
    import torch.fx as fx
    import torch

    if getattr(cfx, "_spyre_wrapped", False):
        return
    with _autoload_lock:
        if getattr(cfx, "_spyre_wrapped", False):
            return
        _orig = cfx.compile_fx

        # Iterate over producer nodes (supports nested containers of nodes)
        def iter_nodes(x):
            if isinstance(x, fx.Node):
                yield x
            elif isinstance(x, (tuple, list)):
                for e in x:
                    yield from iter_nodes(e)
            elif isinstance(x, dict):
                for e in x.values():
                    yield from iter_nodes(e)

        def iter_tensors(v):
            if isinstance(v, torch.Tensor):
                yield v  # FakeTensor is a Tensor subclass, so this works
            elif isinstance(v, (tuple, list)):
                for e in v:
                    yield from iter_tensors(e)
            elif isinstance(v, dict):
                for e in v.values():
                    yield from iter_tensors(e)

        def _uses_spyre(gm, example_inputs, device_name=DEVICE_NAME) -> bool:
            # Inputs
            if any(
                isinstance(x, torch.Tensor)
                and getattr(x.device, "type", None) == device_name
                for x in (example_inputs or ())
            ):
                return True
            # Output
            out_node = gm.graph.output_node()
            out_puts = out_node.args[0] if out_node.args else []
            for n in iter_nodes(out_puts):
                meta = getattr(n, "meta", {}) or {}
                mv = meta.get("val", None) or meta.get("example_value", None)
                if mv is None:
                    continue

                if any(
                    getattr(getattr(t, "device", None), "type", None) == device_name
                    for t in iter_tensors(mv)
                ):
                    return True

            # Graph nodes (covers tensorless factories)
            for n in gm.graph.nodes:
                dev = n.kwargs.get("device")
                if dev is None:
                    continue

                if isinstance(dev, torch.device) and dev.type == device_name:
                    return True
                if isinstance(dev, str) and dev.split(":")[0] == device_name:
                    return True
            return False

        @wraps(_orig)
        def _wrapper(gm, example_inputs, *args, **kwargs):
            decomps = kwargs.setdefault(
                "decompositions", torch._inductor.decomposition.decompositions
            )

            # Check if broadcast is in decompositions
            # NOTE: We WANT broadcast in decompositions to prevent graph partitioning
            # The decomposition is a passthrough that signals to Inductor this op is known
            print(f"[DECOMP CHECK] torch.ops.spyre.broadcast.default in decompositions: {torch.ops.spyre.broadcast.default in decomps}")
            if torch.ops.spyre.broadcast.default in decomps:
                print(f"[DECOMP CHECK] Found broadcast decomposition (this prevents partitioning): {decomps[torch.ops.spyre.broadcast.default]}")
            
            # allowing lowering for any devices - to show CPU and Spyre demo
            # only a temporary change to enable CPU demo
            #if _uses_spyre(gm, example_inputs):
            if 1:
                torch.spyre._impl._lazy_init()

                with enable_spyre_context(
                    example_inputs, decomps=decomps
                ) as spyre_context_decompositions:
                    # The `decomps` is the updated in the context manager
                    # with the appropriate spyre decompositions
                    # and yielded as `spyre_context_decompositions` from the CM

                    # Check again after enable_spyre_context
                    # NOTE: We WANT broadcast in decompositions to prevent graph partitioning
                    print(f"[DECOMP CHECK AFTER CM] torch.ops.spyre.broadcast.default in spyre_context_decompositions: {torch.ops.spyre.broadcast.default in spyre_context_decompositions}")
                    if torch.ops.spyre.broadcast.default in spyre_context_decompositions:
                        print(f"[DECOMP CHECK AFTER CM] Found broadcast decomposition (keeping it to prevent partitioning)")
                        # DO NOT remove it - it prevents graph partitioning!

                    kwargs["decompositions"] = spyre_context_decompositions
                    print("FX graph before lowering")
                    print(gm.graph)
                    
                    # TEMPORARILY DISABLED: Let C10D ops pass through to GraphLowering
                    # We now register lowerings for _c10d_functional.broadcast and wait_tensor directly
                    # so GraphLowering can handle them without rewriting before AOT
                    print(f"[COMPILE_FX] SKIPPING lower_collectives - testing direct C10D lowerings")
                    # gm = lower_collectives(gm)

                    print("FX graph after lowering (SKIPPED)")
                    print(gm.graph)
                    
                    # Debug: Check for C10D ops in the graph
                    print(f"[BEFORE INDUCTOR] Checking for C10D ops in graph:")
                    for node in gm.graph.nodes:
                        if "broadcast" in str(node.target) or "wait_tensor" in str(node.target):
                            print(f"  -> Found C10D op: {node.target}")
                            print(f"     Type: {type(node.target)}")
                            print(f"     Is broadcast.default: {node.target == torch.ops._c10d_functional.broadcast.default}")
                            print(f"     Is wait_tensor.default: {node.target == torch.ops._c10d_functional.wait_tensor.default}")
                            print(f"     Args: {node.args}")
                    
                    print(f"[CALLING INDUCTOR] About to call torch._inductor.compile_fx.compile_fx")
                    print(f"[CALLING INDUCTOR] Graph module type: {type(gm)}")
                    print(f"[CALLING INDUCTOR] Decompositions keys sample: {list(kwargs.get('decompositions', {}).keys())[:5]}")

                    # Register broadcast lowering in the global lowering table BEFORE calling Inductor
                    # This ensures it's available during graph partitioning
                    from torch._inductor import lowering as inductor_lowering
                    from torch_spyre._inductor.lowering import lower_spyre_broadcast
                    
                    # Save original if it exists
                    original_broadcast_lowering = inductor_lowering.lowerings.get(torch.ops.spyre.broadcast.default)
                    
                    # Register our lowering
                    inductor_lowering.lowerings[torch.ops.spyre.broadcast.default] = lower_spyre_broadcast
                    print(f"[PRE-INDUCTOR] Registered broadcast lowering in global table")
                    print(f"[PRE-INDUCTOR] torch.ops.spyre.broadcast.default in lowerings: {torch.ops.spyre.broadcast.default in inductor_lowering.lowerings}")
                    
                    # CRITICAL: Patch the partitioner to prevent it from splitting out broadcast
                    # The partitioner checks if an op is "supported" before deciding to compile it
                    # We need to tell it that torch.ops.spyre.broadcast.default is supported
                    try:
                        from torch._inductor.fx_passes.joint_graph import patterns as joint_patterns
                        # Add broadcast to the set of ops that should not cause graph breaks
                        if hasattr(joint_patterns, '_misc_patterns_handler'):
                            print(f"[PRE-INDUCTOR] Patching joint_graph patterns")
                    except Exception as e:
                        print(f"[PRE-INDUCTOR] Could not patch joint_graph: {e}")
                    
                    # COMMENTED OUT: The identity decomposition was being EXECUTED by compile_fx
                    # This caused the broadcast to be replaced with a passthrough, preventing lowering
                    # We need a different approach to prevent partitioning
                    
                    # decomps = kwargs.get('decompositions', {})
                    # if torch.ops.spyre.broadcast.default not in decomps:
                    #     def broadcast_identity(x, src_rank=0, group_name='default'):
                    #         return x
                    #     decomps[torch.ops.spyre.broadcast.default] = broadcast_identity
                    #     kwargs['decompositions'] = decomps
                    #     print(f"[PRE-INDUCTOR] Added broadcast identity decomposition to prevent partitioning")
                    
                    print(f"[PRE-INDUCTOR] NOT adding decomposition - testing without it")

                    # Patch compile_fx internals to see what's happening
                    import torch._inductor.compile_fx as compile_fx_module
                    
                    # Try to patch the actual compilation function that creates GraphLowering
                    if hasattr(compile_fx_module, 'compile_fx_inner'):
                        original_compile_fx_inner = compile_fx_module.compile_fx_inner
                        
                        def patched_compile_fx_inner(gm_inner, example_inputs_inner, *inner_args, **inner_kwargs):
                            print(f"\n[COMPILE_FX_INNER] Called!")
                            print(f"[COMPILE_FX_INNER] Graph nodes:")
                            for node in gm_inner.graph.nodes:
                                if 'broadcast' in str(node.target):
                                    print(f"[COMPILE_FX_INNER]   *** BROADCAST NODE: {node.target}")
                                else:
                                    print(f"[COMPILE_FX_INNER]   {node.op} {node.target}")
                            return original_compile_fx_inner(gm_inner, example_inputs_inner, *inner_args, **inner_kwargs)
                        
                        compile_fx_module.compile_fx_inner = patched_compile_fx_inner
                    
                    try:
                        # CRITICAL DEBUG: Check which registry has our lowering
                        from torch._inductor import lowering as inductor_lowering
                        from torch_spyre._inductor.lowering import spyre_lowerings, lower_spyre_broadcast
                        
                        op = torch.ops.spyre.broadcast.default
                        
                        print(f"\n{'='*80}")
                        print(f"[REGISTRY DEBUG] Checking lowering registries before calling Inductor:")
                        print(f"{'='*80}")
                        print(f"[REGISTRY] op = {op}")
                        print(f"\n[REGISTRY] Global Inductor Registry (inductor_lowering.lowerings):")
                        print(f"  - op in registry: {op in inductor_lowering.lowerings}")
                        if op in inductor_lowering.lowerings:
                            registered_func = inductor_lowering.lowerings.get(op)
                            print(f"  - registered function: {registered_func}")
                            print(f"  - is our lower_spyre_broadcast: {registered_func is lower_spyre_broadcast}")
                            print(f"  - function name: {getattr(registered_func, '__name__', 'unknown')}")
                        else:
                            print(f"  - NOT FOUND in global registry!")
                        
                        print(f"\n[REGISTRY] Spyre-specific Registry (spyre_lowerings):")
                        print(f"  - op in registry: {op in spyre_lowerings}")
                        if op in spyre_lowerings:
                            registered_func = spyre_lowerings.get(op)
                            print(f"  - registered function: {registered_func}")
                            print(f"  - is our lower_spyre_broadcast: {registered_func is lower_spyre_broadcast}")
                        else:
                            print(f"  - NOT FOUND in spyre registry!")
                        
                        print(f"\n[REGISTRY] Expected state:")
                        print(f"  ✓ op in inductor_lowering.lowerings: True")
                        print(f"  ✓ is our lower_spyre_broadcast: True")
                        print(f"{'='*80}\n")
                        
                        print(f"[BEFORE _ORIG GRAPH CODE] Graph module code:")
                        print(f"{'='*80}")
                        print(gm.code)
                        print(f"{'='*80}\n")
                        
                        # CRITICAL: Remove any broadcast decomposition that may have been added
                        # by enable_spyre_context or elsewhere
                        decomps = kwargs.get("decompositions", {})
                        print(f"[FINAL DECOMP CHECK] broadcast in decomps before _orig: {torch.ops.spyre.broadcast.default in decomps}")
                        if torch.ops.spyre.broadcast.default in decomps:
                            print(f"[FINAL DECOMP CHECK] Found broadcast decomposition, REMOVING it")
                            print(f"[FINAL DECOMP CHECK] Decomposition was: {decomps[torch.ops.spyre.broadcast.default]}")
                            decomps.pop(torch.ops.spyre.broadcast.default, None)
                            kwargs["decompositions"] = decomps
                        print(f"[FINAL DECOMP CHECK] broadcast in decomps after pop: {torch.ops.spyre.broadcast.default in kwargs.get('decompositions', {})}")
                        
                        print(f"\n[CALLING _ORIG] About to call torch._inductor.compile_fx.compile_fx")
                        print(f"[CALLING _ORIG] Invariants:")
                        print(f"  ✓ Graph contains torch.ops.spyre.broadcast.default")
                        print(f"  ✓ Global lowering table has lower_spyre_broadcast")
                        print(f"  ✓ Decompositions do NOT contain broadcast")
                        result = _orig(
                            gm,
                            example_inputs,
                            *args,
                            **kwargs,
                        )
                        print(f"[AFTER _ORIG] Returned from compile_fx")
                    finally:
                        # Restore compile_fx_inner if we patched it
                        if hasattr(compile_fx_module, 'compile_fx_inner') and 'original_compile_fx_inner' in locals():
                            compile_fx_module.compile_fx_inner = original_compile_fx_inner
                        
                        # COMMENTED OUT: Don't restore the lowering yet
                        # If compile is lazy, we may be removing it before the returned function compiles
                        # # Restore original if it existed, otherwise remove
                        # if original_broadcast_lowering is not None:
                        #     inductor_lowering.lowerings[torch.ops.spyre.broadcast.default] = original_broadcast_lowering
                        # else:
                        #     inductor_lowering.lowerings.pop(torch.ops.spyre.broadcast.default, None)
                        print(f"[FINALLY] Keeping broadcast lowering in registry (not restoring)")
                    
                    print(f"[AFTER INDUCTOR] Compilation complete, result type: {type(result)}")
                    return result

            return _orig(gm, example_inputs, *args, **kwargs)

        cfx.compile_fx = _wrapper
        cfx._spyre_wrapped = True


def _light_autoload():
    from . import decompositions  # noqa: F401

    enable_spyre_compile_fx_wrapper()


def _autoload():
    if getattr(_autoload, "_ran", False):
        return

    with _autoload_lock:
        if getattr(_autoload, "_ran", False):
            return
        from torch._dynamo.device_interface import register_interface_for_device

        from torch_spyre.device.interface import SpyreInterface

        register_interface_for_device(DEVICE_NAME, SpyreInterface)

        from torch._inductor.codegen.common import (
            register_backend_for_device,
            register_device_op_overrides,
        )

        # Register in-tree CPU and CUDA device
        from torch._inductor.codegen import cpu_device_op_overrides  # noqa: F401  # usort: skip
        from torch._inductor.codegen.cuda import device_op_overrides  # noqa: F401  # usort: skip

        from torch_spyre.device.op_overrides import SpyreDeviceOpOverrides

        register_device_op_overrides(
            device=DEVICE_NAME, device_op_overrides=SpyreDeviceOpOverrides()
        )

        from .scheduler import SuperDSCScheduling
        from .wrapper import SpyrePythonWrapperCodegen

        register_backend_for_device(
            DEVICE_NAME,
            SuperDSCScheduling,
            SpyrePythonWrapperCodegen,
            device_custom_config=config,
        )

        _autoload._ran = True
