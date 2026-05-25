import torch
import torch.fx as fx


def lower_collectives(gm: fx.GraphModule):
    graph = gm.graph

    print("\n=== FX GRAPH BEFORE LOWERING ===")
    print(graph)

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        # Handle all_reduce + wait_tensor pattern
        if node.target == torch.ops._c10d_functional.all_reduce:
            all_reduce_node = node

            wait_users = [
                u for u in list(all_reduce_node.users)
                if u.op == "call_function"
                and u.target == torch.ops._c10d_functional.wait_tensor
            ]

            print(">> Lowering _c10d_functional.all_reduce + wait_tensor → spyre.all_reduce_")

            with graph.inserting_after(all_reduce_node):
                new_node = graph.call_function(
                    torch.ops.spyre.all_reduce_.default,
                    args=all_reduce_node.args,
                    kwargs=all_reduce_node.kwargs,
                )
                print(f"   Created node with target: {new_node.target}")

            if wait_users:
                for wait_node in wait_users:
                    wait_node.replace_all_uses_with(new_node)
                    graph.erase_node(wait_node)
            else:
                all_reduce_node.replace_all_uses_with(new_node)

            graph.erase_node(all_reduce_node)

        # Handle broadcast + wait_tensor pattern
        elif node.target == torch.ops._c10d_functional.broadcast:
            broadcast_node = node

            wait_users = [
                u for u in list(broadcast_node.users)
                if u.op == "call_function"
                and u.target == torch.ops._c10d_functional.wait_tensor
            ]

            print(">> Lowering _c10d_functional.broadcast + wait_tensor → spyre.broadcast")
            print(f"   Original broadcast_node.args: {broadcast_node.args}")
            print(f"   Original broadcast_node.kwargs: {broadcast_node.kwargs}")

            with graph.inserting_after(broadcast_node):
                new_node = graph.call_function(
                    torch.ops.spyre.broadcast.default,
                    args=broadcast_node.args,
                    kwargs=broadcast_node.kwargs,
                )
                print(f"   Created node with target: {new_node.target}")
                print(f"   New node type: {type(new_node.target)}")
                print(f"   New node.args: {new_node.args}")
                print(f"   New node.kwargs: {new_node.kwargs}")
                print(f"   Is target the OpOverload? {new_node.target is torch.ops.spyre.broadcast.default}")

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