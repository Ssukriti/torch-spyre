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

from contextlib import contextmanager
import os
import sys

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import InputType
from torch._inductor.virtualized import V
from typing import Callable, Optional

# Set this as early as possible to disable precompiled headers
# This avoids the openssl dependency issue
os.environ["TORCHINDUCTOR_CPP_WRAPPER_PRECOMPILE_HEADERS"] = "0"

# Monkey-patch the _precompile_header function to return None (disable precompiled headers)
try:
    from torch._inductor import codecache
    _original_precompile_header = codecache._precompile_header
    
    def _patched_precompile_header(*args, **kwargs):
        # Return None to disable precompiled headers
        return None
    
    codecache._precompile_header = _patched_precompile_header
except Exception as e:
    print(f"Warning: Could not patch _precompile_header: {e}", file=sys.stderr)

# CRITICAL: Patch the scheme class's codegen_and_compile method
# This is the actual post-AOT hook point (scheme.codegen_and_compile)
try:
    import torch._inductor.compile_fx as cfx
    from torch_spyre._inductor.distributed.fx_pass import lower_collectives
    
    # Find the scheme classes - try common ones
    scheme_classes = []
    for attr_name in dir(cfx):
        attr = getattr(cfx, attr_name)
        if isinstance(attr, type) and hasattr(attr, 'codegen_and_compile'):
            scheme_classes.append((attr_name, attr))
            print(f"[PATCHES] Found scheme class: {attr_name}")
    
    if not scheme_classes:
        print(f"[PATCHES] WARNING: No scheme classes found with codegen_and_compile method")
    
    # Patch all found scheme classes
    for class_name, scheme_class in scheme_classes:
        print(f"[PATCHES] Patching {class_name}.codegen_and_compile")
        
        orig_codegen_and_compile = scheme_class.codegen_and_compile
        
        def wrapped_codegen_and_compile(self, gm, example_inputs, *args, **kwargs):
            print(f"\n[SCHEME CODEGEN HIT] {class_name}.codegen_and_compile called!")
            print("[SCHEME CODEGEN] Graph before lower_collectives:")
            print(gm.graph)
            
            has_collective = any(
                "_c10d" in str(n.target)
                for n in gm.graph.nodes
                if hasattr(n, "target")
            )
            
            print(f"[SCHEME CODEGEN] has_collective: {has_collective}")
            
            if has_collective:
                print("[SCHEME CODEGEN] Applying lower_collectives...")
                gm = lower_collectives(gm)
                
                print("\n[SCHEME CODEGEN] After lower_collectives:")
                print(gm.graph)
            
            return orig_codegen_and_compile(self, gm, example_inputs, *args, **kwargs)
        
        scheme_class.codegen_and_compile = wrapped_codegen_and_compile
        print(f"[PATCHES] Successfully patched {class_name}.codegen_and_compile")
    
except Exception as e:
    print(f"[PATCHES] ERROR: Could not patch scheme.codegen_and_compile: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()


@contextmanager
def spyre_data_types():
    saved = torch._prims_common._computation_dtype_map
    torch._prims_common._computation_dtype_map = {
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.complex32: torch.complex32,
    }
    try:
        yield
    finally:
        torch._prims_common._computation_dtype_map = saved


@contextmanager
def enable_spyre_context(
    example_inputs: list[InputType],
    decomps: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
):
    """
    Context manager that sets up the complete Spyre compilation environment.

    This CM configures PyTorch Inductor to compile graphs for the Spyre device by:
      - Enabling Spyre-specific data type handling
      - Activating Spyre lowerings and decompositions
      - Configuring Inductor settings optimized for Spyre
      - Setting up custom pre/post compilation passes
      - Disabling incompatible optimizations (e.g., reduction splitting, permute fusion)

    Args:
        example_inputs: List of example inputs to the graph being compiled. Used to
            set real inputs in the virtualized context for shape inference and
            optimization decisions.
        decomps: Decomposition table to be populated with Spyre-specific
            decompositions. Maps operator overloads to their decomposition implementations.
            This is typically a clone of PyTorch Inductor's global decomposition registry.
    """

    if decomps is None:
        decomps = torch._inductor.decomposition.decompositions

    from torch_spyre._inductor.lowering import enable_spyre_lowerings  # your CM

    # Ensure decorators run (custom ops/decomp/lowerings modules)
    import torch_spyre._inductor.customops  # noqa: F401
    from torch_spyre._inductor.decompositions import (
        enable_spyre_decompositions,
    )

    import torch_spyre._inductor.lowering  # noqa: F401
    from torch_spyre._inductor.choices import SpyreHeuristics
    from torch_spyre._inductor.passes import (
        CustomPreGradPasses,
        CustomPrePasses,
        CustomPostPasses,
        CustomPreFusionPasses,
        CustomPostFusionPasses,
        CustomPreSchedulingPasses,
    )

    # *) Inductor config tweaks (saved/restored)
    new_config = {
        "split_reductions": False,
        "benchmark_harness": False,
        "pre_grad_custom_pass": CustomPreGradPasses(),
        "post_grad_custom_pre_pass": CustomPrePasses(),
        "post_grad_custom_post_pass": CustomPostPasses(),
        "_pre_fusion_custom_pass": CustomPreFusionPasses(),
        "_post_fusion_custom_pass": CustomPostFusionPasses(),
        # Adding this configuration in so as to avoid the optimization of turning small matmuls into non-matmuls
        # found here: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py#L1580
        "unroll_reductions_threshold": 1,
        # Disable fusing of mm + permute/transpose for now.
        "permute_fusion": False,
        "allow_buffer_reuse": False,  # For now, as buffer reuse does not consider stride_map.
    }

    from torch._inductor.ir import Loops

    # Force all operations to be realized when LoopLevel IR is initially constructed
    old_loop = Loops.has_large_inner_fn
    Loops.has_large_inner_fn = lambda self, threshold=None: True

    from torch._inductor.fx_passes import joint_graph

    origin_pass = list(joint_graph.pass_patterns)
    # disable mul_softmax_pattern and div_softmax_pattern for now
    joint_graph.pass_patterns.pop()

    # Inject the pre_scheduling_passes before the Scheduler is constructed,
    # allowing the passes to modify the graph IR (buffers, inputs, constants).
    old_update_scheduler = GraphLowering._update_scheduler

    _pre_scheduling_pass = CustomPreSchedulingPasses()

    def _spyre_update_scheduler(self: GraphLowering) -> None:
        _pre_scheduling_pass(self.operations)
        old_update_scheduler(self)

    GraphLowering._update_scheduler = _spyre_update_scheduler  # type: ignore[method-assign]
    
    # NOTE: fx_codegen_and_compile wrapper is now at module load time (top of file)
    # No need to wrap it again here
    
    # Patch run_node to debug broadcast lowering
    old_run_node = GraphLowering.run_node
    
    def _debug_run_node(self: GraphLowering, node):
        # Log ALL nodes to see if C10D ops reach GraphLowering
        if hasattr(node, 'op'):
            if node.op == 'call_function' and hasattr(node, 'target'):
                target_str = str(node.target)
                # Print ALL call_function nodes
                print(f"[RUN_NODE] op={node.op}, target={target_str}")
                
                # Check if this is a C10D or spyre op
                if 'broadcast' in target_str or 'wait_tensor' in target_str or 'spyre' in target_str or '_c10d' in target_str:
                    print(f"\n[RUN_NODE] *** FOUND COLLECTIVE NODE ***")
                    print(f"[RUN_NODE] Node: {node}")
                    print(f"[RUN_NODE] Target: {node.target}")
                    print(f"[RUN_NODE] Target type: {type(node.target)}")
                    print(f"[RUN_NODE] Target id: {id(node.target)}")
                    print(f"[RUN_NODE] Op: {node.op}")
                    print(f"[RUN_NODE] Args: {node.args}")
                    
                    # Check if it's the OpOverloadPacket or OpOverload
                    import torch
                    if '_c10d_functional.broadcast' in target_str:
                        packet = torch.ops._c10d_functional.broadcast
                        default = torch.ops._c10d_functional.broadcast.default
                        print(f"[RUN_NODE] Comparing with torch.ops._c10d_functional.broadcast:")
                        print(f"[RUN_NODE]   packet id: {id(packet)}, node.target == packet: {node.target == packet}, node.target is packet: {node.target is packet}")
                        print(f"[RUN_NODE]   default id: {id(default)}, node.target == default: {node.target == default}, node.target is default: {node.target is default}")
                    
                    print(f"[RUN_NODE] Checking lowering.lowerings for: {node.target}")
                    import torch._inductor.lowering as lowering
                    print(f"[RUN_NODE] Is in lowering.lowerings? {node.target in lowering.lowerings}")
                    if node.target in lowering.lowerings:
                        print(f"[RUN_NODE] Lowering function: {lowering.lowerings[node.target]}")
                    else:
                        print(f"[RUN_NODE] NOT FOUND in lowerings - will not be lowered!")
                        # Show what IS in lowerings for C10D ops
                        print(f"[RUN_NODE] Checking what C10D keys ARE in lowerings:")
                        for key in lowering.lowerings.keys():
                            if '_c10d' in str(key) or 'broadcast' in str(key):
                                print(f"[RUN_NODE]   Found key: {key}, id: {id(key)}, type: {type(key)}")
        
        result = old_run_node(self, node)
        
        if hasattr(node, 'target') and ('broadcast' in str(node.target) or 'wait_tensor' in str(node.target)):
            print(f"[RUN_NODE] Result type: {type(result)}")
            print(f"[RUN_NODE] Result: {result}\n")
        
        return result
    
    GraphLowering.run_node = _debug_run_node  # type: ignore[method-assign]
    
    # Also patch the graph processing to see ALL nodes being processed
    old_run = GraphLowering.run
    
    def _debug_run(self: GraphLowering, *args):
        print(f"\n[GRAPH_LOWERING] Starting graph lowering")
        print(f"[GRAPH_LOWERING] Graph has {len(list(self.graph.nodes))} nodes")
        print(f"[GRAPH_LOWERING] ALL NODES IN GRAPH:")
        for i, node in enumerate(self.graph.nodes):
            target_str = str(node.target) if hasattr(node, 'target') else 'no target'
            print(f"[GRAPH_LOWERING]   {i}: op={node.op}, target={target_str}")
            if 'broadcast' in target_str and 'spyre' in target_str:
                print(f"[GRAPH_LOWERING]   ^^^ THIS IS THE BROADCAST NODE WE'RE LOOKING FOR!")
        result = old_run(self, *args)
        print(f"[GRAPH_LOWERING] Finished graph lowering\n")
        return result
    
    GraphLowering.run = _debug_run  # type: ignore[method-assign]
    
    # COMMENTED OUT: Special call_function handling for testing
    # We want to verify the normal lowering path works, not a special case
    # old_call_function = GraphLowering.call_function
    #
    # def _patched_call_function(self: GraphLowering, target, args, kwargs):
    #     # Handle torch.ops.spyre.broadcast.default directly
    #     if target == torch.ops.spyre.broadcast.default:
    #         print(f"\n[CALL_FUNCTION] *** HANDLING spyre.broadcast.default ***")
    #         ...
    #         return result
    #
    #     return old_call_function(self, target, args, kwargs)
    #
    # GraphLowering.call_function = _patched_call_function  # type: ignore[method-assign]

    with (
        spyre_data_types(),
        enable_spyre_lowerings(),
        enable_spyre_decompositions(decomps=decomps) as spyre_context_decompositions,
        V.set_real_inputs(example_inputs),
        V.set_choices_handler(SpyreHeuristics()),
        torch._inductor.config.patch(new_config),
    ):
        try:
            yield spyre_context_decompositions
        finally:
            joint_graph.pass_patterns[:] = origin_pass
            Loops.has_large_inner_fn = old_loop
            GraphLowering._update_scheduler = old_update_scheduler  # type: ignore[method-assign]
            GraphLowering.run_node = old_run_node  # type: ignore[method-assign]
            GraphLowering.run = old_run  # type: ignore[method-assign]
            # COMMENTED OUT: old_call_function doesn't exist since we commented out the patch
            # GraphLowering.call_function = old_call_function  # type: ignore[method-assign]
