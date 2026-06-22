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

"""
Reorder operations to enable communication-compute overlap.

This pass runs before fusion and scheduling to reorder IR operations such that:
1. Async communication operations are scheduled early
2. Independent compute operations are placed between comm start and dependent ops
3. This allows independent compute to overlap with communication
"""

import logging
from typing import List, Set
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Operation
from .ir import SpyreBroadcastAsyncFallback
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("reorder_async_comm")


def get_operation_reads(op: Operation) -> Set[str]:
    """Get the set of buffer names that this operation reads."""
    reads = set()
    if hasattr(op, 'get_read_names'):
        reads.update(op.get_read_names())
    elif hasattr(op, 'get_read_writes'):
        rw = op.get_read_writes()
        reads.update(dep.name for dep in rw.reads)
    return reads


def get_operation_writes(op: Operation) -> Set[str]:
    """Get the set of buffer names that this operation writes."""
    writes = set()
    if hasattr(op, 'get_name'):
        name = op.get_name()
        if name:
            writes.add(name)
    elif hasattr(op, 'get_read_writes'):
        rw = op.get_read_writes()
        writes.update(dep.name for dep in rw.writes)
    return writes


def is_async_comm_operation(op: Operation) -> bool:
    """Check if an operation is an async communication operation."""
    return isinstance(op, SpyreBroadcastAsyncFallback)


def is_independent_of(op: Operation, comm_ops: List[Operation]) -> bool:
    """
    Check if an operation is independent of all async communication operations.
    
    An operation is independent if it doesn't read from any buffers written by comm ops.
    """
    comm_writes = set()
    for comm_op in comm_ops:
        comm_writes.update(get_operation_writes(comm_op))
    
    op_reads = get_operation_reads(op)
    
    # Operation is independent if it doesn't read from comm outputs
    return not bool(comm_writes & op_reads)


def reorder_for_async_overlap(graph: GraphLowering) -> None:
    """
    Reorder operations to enable communication-compute overlap.
    
    Strategy:
    1. Identify async communication operations
    2. Find operations that are independent of the comm operations
    3. Reorder so that: [pre-comm ops] -> [comm ops] -> [independent ops] -> [dependent ops]
    
    This allows independent operations to execute while communication is in flight.
    
    Args:
        graph: GraphLowering object containing operations in graph.operations
    """
    operations = graph.operations
    
    logger.info("Starting reorder_for_async_overlap")
    logger.info(f"Total operations: {len(operations)}")
    
    # Debug: Print all operations
    for i, op in enumerate(operations):
        op_name = op.get_name() if hasattr(op, 'get_name') else str(type(op))
        op_type = type(op).__name__
        logger.debug(f"  {i}. {op_name} ({op_type})")
    
    # Phase 1: Identify async communication operations
    async_comm_ops = [op for op in operations if is_async_comm_operation(op)]
    
    if not async_comm_ops:
        # No async comm operations, nothing to reorder
        logger.info("No async comm operations found, skipping reorder")
        return
    
    logger.info(f"Found {len(async_comm_ops)} async comm operations")
    for op in async_comm_ops:
        logger.debug(f"  - {op.get_name()}")
    logger.info("Reordering to enable communication-compute overlap")
    
    # Phase 2: Find the position of the first async comm operation
    first_comm_idx = min(operations.index(op) for op in async_comm_ops)
    last_comm_idx = max(operations.index(op) for op in async_comm_ops)
    
    logger.debug(f"first_comm_idx={first_comm_idx}, last_comm_idx={last_comm_idx}")
    
    # Phase 3: Separate operations into categories
    pre_comm_ops = operations[:first_comm_idx]
    comm_ops = async_comm_ops
    post_comm_ops = operations[last_comm_idx + 1:]
    
    logger.debug(f"pre_comm_ops: {len(pre_comm_ops)}, comm_ops: {len(comm_ops)}, post_comm_ops: {len(post_comm_ops)}")
    
    # Phase 4: Separate post-comm operations into independent and dependent
    independent_ops = []
    dependent_ops = []
    
    logger.debug(f"Analyzing {len(post_comm_ops)} post-comm operations...")
    for op in post_comm_ops:
        op_name = op.get_name() if hasattr(op, 'get_name') else str(type(op))
        if is_independent_of(op, comm_ops):
            independent_ops.append(op)
            logger.debug(f"Independent op (can overlap): {op_name}")
        else:
            dependent_ops.append(op)
            logger.debug(f"Dependent op (after comm): {op_name}")
    
    # Phase 5: Reconstruct operations list in optimal order
    # Order: pre-comm -> comm -> independent -> dependent
    new_order = pre_comm_ops + comm_ops + independent_ops + dependent_ops
    
    logger.info(f"Reordering: pre({len(pre_comm_ops)}) -> comm({len(comm_ops)}) -> independent({len(independent_ops)}) -> dependent({len(dependent_ops)})")
    
    # Phase 6: Update the operations list in-place
    operations.clear()
    operations.extend(new_order)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Final operation order:")
        for i, op in enumerate(operations):
            op_name = op.get_name() if hasattr(op, 'get_name') else str(type(op))
            op_type = "ASYNC_COMM" if op in comm_ops else \
                      "INDEPENDENT" if op in independent_ops else \
                      "DEPENDENT" if op in dependent_ops else "PRE_COMM"
            logger.debug(f"  {i+1}. [{op_type}] {op_name}")
    
    if independent_ops:
        logger.info(f"SUCCESS: {len(independent_ops)} operations can overlap with communication!")
    else:
        logger.info("WARNING: No independent operations found to overlap with communication")

# Made with Bob
