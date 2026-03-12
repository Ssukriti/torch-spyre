import torch
import torch.fx as fx
import operator
from .lowerings import LOWERED_OPS

PASSTHROUGH_METHODS = {"clone", "copy_", "detach", "view", "reshape", "contiguous"}


def is_wait(node):
    return node.op == "call_function" and node.target == torch.ops._c10d_functional.wait_tensor


def is_passthrough(node):
    return (node.op == "call_method" and node.target in PASSTHROUGH_METHODS) or (
        node.op == "call_function" and node.target == operator.getitem
    )


def collect_chain(start):
    stack = [start]
    chain = set()
    while stack:
        node = stack.pop()
        for user in list(node.users):
            if is_wait(user) or is_passthrough(user):
                if user not in chain:
                    chain.add(user)
                    stack.append(user)
    return chain


def lower_collectives(gm: fx.GraphModule):
    graph = gm.graph
    nodes_to_delete = set()

    for node in list(graph.nodes):
        # Skip lowering wait_tensor itself; only lower the collective
        if node.op != "call_function" or node.target not in LOWERED_OPS:
            continue
        if node.target == torch.ops._c10d_functional.wait_tensor:
            continue  # <--- skip double lowering

        collective = node
        chain = collect_chain(collective)

        # Detect wait_tensor nodes in the chain
        wait_nodes = [n for n in chain if is_wait(n)]
        has_wait = len(wait_nodes) > 0

        # Insert async collective
        with graph.inserting_after(collective):
            async_node = graph.call_function(
                LOWERED_OPS[collective.target],
                args=collective.args,
                kwargs=collective.kwargs,
            )
            async_node.meta = dict(collective.meta)

        final_node = async_node

        # Replace old wait_tensor nodes with async handle
        for n in wait_nodes:
            n.replace_all_uses_with(async_node)
            nodes_to_delete.add(n)

        # Insert single spyre_wait if any wait_tensor existed
        if has_wait:
            with graph.inserting_after(async_node):
                wait_node = graph.call_function(
                    LOWERED_OPS[torch.ops._c10d_functional.wait_tensor],
                    args=(async_node,),
                )
                wait_node.meta = dict(async_node.meta)
                final_node = wait_node

        # Redirect all downstream uses (passthrough ops etc.)
        collective.replace_all_uses_with(final_node)
        for n in chain:
            n.replace_all_uses_with(final_node)
            nodes_to_delete.add(n)
        nodes_to_delete.add(collective)

    # Safe node deletion
    changed = True
    while changed:
        changed = False
        for n in list(nodes_to_delete):
            if len(n.users) == 0:
                graph.erase_node(n)
                nodes_to_delete.remove(n)
                changed = True

    graph.lint()
    gm.recompile()
    return gm