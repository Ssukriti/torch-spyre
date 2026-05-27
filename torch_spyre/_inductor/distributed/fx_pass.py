import torch
import torch.fx as fx


def lower_collectives(gm: fx.GraphModule):
    graph = gm.graph

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        # Handle broadcast + wait_tensor pattern
        if node.target == torch.ops._c10d_functional.broadcast:
            broadcast_node = node

            wait_users = [
                u for u in list(broadcast_node.users)
                if u.op == "call_function"
                and u.target == torch.ops._c10d_functional.wait_tensor
            ]
            print("\n=== FX GRAPH BEFORE LOWERING ===")
            print(graph)

            print(">> Lowering _c10d_functional.broadcast + wait_tensor → spyre.broadcast")

            with graph.inserting_after(broadcast_node):
                new_node = graph.call_function(
                    torch.ops.spyre.broadcast.default,
                    args=broadcast_node.args,
                    kwargs=broadcast_node.kwargs,
                )

            if wait_users:
                for wait_node in wait_users:
                    wait_node.replace_all_uses_with(new_node)
                    graph.erase_node(wait_node)
            else:
                broadcast_node.replace_all_uses_with(new_node)

            graph.erase_node(broadcast_node)

            print("\n=== FX GRAPH AFTER LOWERING ===")
            print(graph)

    graph.lint()
    gm.recompile()
    return gm