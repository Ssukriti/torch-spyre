import torch
import torch.fx as fx
import operator

from .lowerings import LOWERED_OPS


# Tensor materialization / passthrough ops
PASSTHROUGH_METHODS = {
    "clone",
    "copy_",
    "detach",
    "view",
    "reshape",
    "contiguous",
}


def is_wait(node):
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor
    )


def is_passthrough(node):
    return (
        (node.op == "call_method" and node.target in PASSTHROUGH_METHODS)
        or (node.op == "call_function" and node.target == operator.getitem)
    )


def collect_chain(start):
    """
    Walk downstream nodes that do not change collective semantics.
    """
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
    """
    Collapse functional c10d collectives into Spyre collectives.

    Handles patterns like:

        all_reduce
            ↓
        wait_tensor
            ↓
        clone/view/reshape/contiguous
            ↓
        compute

    into:

        spyre_all_reduce
            ↓
        compute
    """

    graph = gm.graph
    nodes_to_delete = set()

    for node in list(graph.nodes):

        if node.op != "call_function":
            continue

        if node.target not in LOWERED_OPS:
            continue

        collective = node

        # ---------------------------------
        # Find async collective chain
        # ---------------------------------

        chain = collect_chain(collective)

        # ---------------------------------
        # Insert Spyre collective
        # ---------------------------------

        with graph.inserting_after(collective):
            new_node = graph.call_function(
                LOWERED_OPS[collective.target],
                args=collective.args,
                kwargs=collective.kwargs,
            )

            # preserve meta for Inductor
            new_node.meta = dict(collective.meta)

        # ---------------------------------
        # Redirect outputs
        # ---------------------------------

        replaced = False

        for n in chain:
            if len(n.users) > 0:
                n.replace_all_uses_with(new_node)
                nodes_to_delete.add(n)
                replaced = True

        if not replaced:
            collective.replace_all_uses_with(new_node)

        nodes_to_delete.add(collective)

    # ---------------------------------
    # SAFE NODE DELETION
    # ---------------------------------

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