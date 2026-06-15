# Copyright 2026 The Torch-Spyre Authors.
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

from torch._inductor.scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from torch._inductor.virtualized import V
from torch._inductor.ir import FallbackKernel
from torch_spyre._inductor.logging_utils import _get_env_bool
from .ir import FixedTiledLayout, SpyreBroadcastAsyncFallback
from .constants import SEGMENT_OFFSETS

# TODO: Temporary hook to easily disable
_FUSION_ENABLED = _get_env_bool("SPYRE_INDUCTOR_ENABLE_FUSION", True)


def _max_bundle_tensors() -> int:
    # Until https://github.com/torch-spyre/torch-spyre/issues/827 is completed.
    has_pool = getattr(V.graph, "pool_size", 0) > 0
    return len(SEGMENT_OFFSETS) - (2 if has_pool else 1)


def _make_fused(nodes: list[SchedulerNode]) -> BaseSchedulerNode | None:
    if len(nodes) > 1:
        return FusedSchedulerNode(nodes[0].scheduler, nodes)
    elif len(nodes) == 1:
        return nodes[0]
    return None


def _is_non_intermediate(name: str) -> bool:
    buf = V.graph.get_buffer(name)
    if buf is None or isinstance(buf, FallbackKernel):
        return False
    layout = buf.get_layout()
    return isinstance(layout, FixedTiledLayout) and not layout.allocation


def _is_async_comm_node(node: BaseSchedulerNode) -> bool:
    """
    Check if a node represents an async communication operation.
    
    Async communication operations (broadcast_async, allreduce_async, etc.) need
    special handling during fusion to enable communication-compute overlap.
    
    Args:
        node: Scheduler node to check
        
    Returns:
        True if node is an async communication operation
    """
    # Check if it's a SchedulerNode with SpyreBroadcastAsyncFallback
    if isinstance(node, SchedulerNode) and hasattr(node, 'node'):
        return isinstance(node.node, SpyreBroadcastAsyncFallback)
    
    # Check if it's an ExternKernelSchedulerNode wrapping SpyreBroadcastAsyncFallback
    if hasattr(node, 'node') and isinstance(node.node, SpyreBroadcastAsyncFallback):
        return True
    
    # Fallback: check node name contains 'broadcast_async'
    if hasattr(node, 'get_name'):
        node_name = node.get_name()
        if 'broadcast_async' in str(node_name).lower():
            return True
    
    return False


def _node_depends_on_async_comm(node: BaseSchedulerNode, async_comm_node: BaseSchedulerNode) -> bool:
    """
    Check if a node depends on the output of an async communication operation.
    
    This is critical for creating correct fusion boundaries. A node depends on async
    comm if it reads from the buffer produced by the async comm operation.
    
    IMPORTANT: Must check the BUFFER name, not the scheduler node name!
    - Scheduler node name: "op1"
    - Buffer name: "buf1"
    - Dependent nodes read from "buf1", not "op1"
    
    Args:
        node: Node to check for dependency
        async_comm_node: The async communication node
        
    Returns:
        True if node depends on async_comm_node's output
    """
    if not isinstance(node, SchedulerNode):
        return False
    
    # Extract the buffer name that the async comm node produces
    # For ExternKernelSchedulerNode wrapping a fallback, the wrapped node's name
    # is the buffer name (e.g., "buf1")
    async_comm_buffer = None
    
    if hasattr(async_comm_node, 'node') and hasattr(async_comm_node.node, 'get_name'):
        async_comm_buffer = async_comm_node.node.get_name()
    
    # Fallback: use the scheduler node's name
    if not async_comm_buffer and hasattr(async_comm_node, 'get_name'):
        async_comm_buffer = async_comm_node.get_name()
    
    if not async_comm_buffer:
        return False
    
    # Check if this node reads from the async comm output buffer
    node_reads = {dep.name for dep in node.read_writes.reads}
    return async_comm_buffer in node_reads


def spyre_fuse_nodes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Fuse nodes together to form kernels without changing their order.
    Each kernel will be compiled into a single SuperDSC Bundle.
    Fusion is limited by the following constraints.
     1. We only want to fuse SchedulerNodes (ie, nodes that generate OpSpecs).
     2. A SDSC Bundle can refer to at most 5 unique non-intermediate tensors
        (graph inputs/outputs). Intermediates don't count toward this limit.
     3. Async communication nodes force bundle boundaries for overlap.
    """
    if not _FUSION_ENABLED or len(nodes) == 0:
        return nodes

    print(f"\n{'='*70}")
    print(f"[FUSION] Starting fusion pass with {len(nodes)} nodes")
    print(f"{'='*70}")
    for i, n in enumerate(nodes):
        node_name = n.get_name() if hasattr(n, 'get_name') else str(type(n))
        node_type = type(n).__name__
        is_async = _is_async_comm_node(n)
        
        # Show what this node reads and writes
        reads = writes = "N/A"
        if isinstance(n, SchedulerNode) and hasattr(n, 'read_writes'):
            reads = {dep.name for dep in n.read_writes.reads}
            writes = {dep.name for dep in n.read_writes.writes}
        
        print(f"  {i}. {node_name} ({node_type}) {'← ASYNC COMM!' if is_async else ''}")
        if reads != "N/A":
            print(f"      Reads: {reads}")
            print(f"      Writes: {writes}")
    print()

    max_tensors = _max_bundle_tensors()
    fused_nodes: list[BaseSchedulerNode] = []
    cur_nodes: list[SchedulerNode] = []
    cur_tensors: set[str] = set()
    cur_non_intermediate_count: int = 0
    
    # Track async comm nodes we've seen
    last_async_comm_node: BaseSchedulerNode | None = None
    in_independent_region = False

    for i, n in enumerate(nodes):
        node_name = n.get_name() if hasattr(n, 'get_name') else f"node_{i}"
        
        # STEP 1: Handle async communication nodes
        # These force a bundle boundary to enable communication-compute overlap
        if _is_async_comm_node(n):
            print(f"[FUSION] Async comm node detected: {node_name}, forcing bundle boundary")
            print(f"[FUSION]   Current bundle has {len(cur_nodes)} nodes")
            
            # Flush pre-comm bundle
            if fused := _make_fused(cur_nodes):
                print(f"[FUSION]   Creating PRE-COMM fused node from {len(cur_nodes)} nodes")
                fused_nodes.append(fused)
            
            # Add async comm as separate (unfused) node
            fused_nodes.append(n)
            print(f"[FUSION]   Adding async comm node as separate (unfused)")
            
            # Track this node for dependency checking of subsequent operations
            last_async_comm_node = n
            in_independent_region = True
            print(f"[FUSION]   Entering independent region after async comm")
            
            # Reset bundle state
            cur_nodes = []
            cur_tensors = set()
            cur_non_intermediate_count = 0
            continue
        
        # STEP 2: After async comm, check if this node depends on comm output
        # This determines whether it goes in the independent or dependent bundle
        if in_independent_region and last_async_comm_node:
            depends_on_comm = _node_depends_on_async_comm(n, last_async_comm_node)
            
            if depends_on_comm:
                # This node depends on async comm - force boundary
                print(f"[FUSION] Node {node_name} depends on async comm output, forcing boundary")
                print(f"[FUSION]   Current INDEPENDENT bundle has {len(cur_nodes)} nodes")
                
                # Flush independent bundle
                if fused := _make_fused(cur_nodes):
                    print(f"[FUSION]   Creating INDEPENDENT fused node from {len(cur_nodes)} nodes")
                    fused_nodes.append(fused)
                
                # Start dependent bundle
                in_independent_region = False
                print(f"[FUSION]   Starting DEPENDENT region")
                
                # Reset bundle state (this dependent node will be added below)
                cur_nodes = []
                cur_tensors = set()
                cur_non_intermediate_count = 0
            else:
                # This node is independent - can overlap with communication
                print(f"[FUSION] Node {node_name} is independent of async comm, can overlap")
        
        # Process SchedulerNode (both independent and dependent nodes)
        if isinstance(n, SchedulerNode):
            n_tensors = {dep.name for dep in n.read_writes.reads_and_writes()}
            new_tensors = n_tensors - cur_tensors
            new_non_intermediate = sum(
                1 for t in new_tensors if _is_non_intermediate(t)
            )
            if cur_non_intermediate_count + new_non_intermediate <= max_tensors:
                # Ok to put in the current bundle
                cur_nodes.append(n)
                cur_tensors |= n_tensors
                cur_non_intermediate_count += new_non_intermediate
            else:
                # Would be too many non-intermediate tensors; start a new bundle.
                if fused := _make_fused(cur_nodes):
                    fused_nodes.append(fused)
                cur_nodes = [n]
                cur_tensors = n_tensors
                cur_non_intermediate_count = sum(
                    1 for t in n_tensors if _is_non_intermediate(t)
                )

        else:
            # Other node types (eg Fallback nodes) force a bundle boundary.
            if fused := _make_fused(cur_nodes):
                fused_nodes.append(fused)
            fused_nodes.append(n)
            cur_nodes = []
            cur_tensors = set()
            cur_non_intermediate_count = 0

    if fused := _make_fused(cur_nodes):
        fused_nodes.append(fused)

    return fused_nodes
