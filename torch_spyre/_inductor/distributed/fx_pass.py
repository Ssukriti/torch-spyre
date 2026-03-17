import torch
import torch.fx as fx


def lower_collectives(gm: fx.GraphModule):
    graph = gm.graph

    print("\n=== FX GRAPH BEFORE LOWERING ===")
    print(graph)

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        if node.target != torch.ops._c10d_functional.all_reduce:
            continue

        all_reduce_node = node

        wait_users = [
            u for u in list(all_reduce_node.users)
            if u.op == "call_function"
            and u.target == torch.ops._c10d_functional.wait_tensor
        ]

        print(">> Lowering _c10d_functional.all_reduce + wait_tensor → spyre.all_reduce")

        with graph.inserting_after(all_reduce_node):
            new_node = graph.call_function(
                torch.ops.spyre.all_reduce,
                args=all_reduce_node.args,
                kwargs=all_reduce_node.kwargs,
            )

        if wait_users:
            for wait_node in wait_users:
                wait_node.replace_all_uses_with(new_node)
                graph.erase_node(wait_node)
        else:
            all_reduce_node.replace_all_uses_with(new_node)

        graph.erase_node(all_reduce_node)

    print("\n=== FX GRAPH AFTER LOWERING ===")
    print(graph)

    graph.lint()
    gm.recompile()
    return gm