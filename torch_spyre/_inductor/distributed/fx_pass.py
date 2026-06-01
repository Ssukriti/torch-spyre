import torch
import torch.fx as fx


def _get_rank():
    """Get current rank, return 0 if not in distributed context."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def lower_collectives(gm: fx.GraphModule, use_async: bool = True):
    """
    Lower collective operations to Spyre ops.
    
    Args:
        gm: FX GraphModule to transform
        use_async: If True, lower to async ops (broadcast_async + wait).
                   If False, lower to synchronous ops (broadcast).
    """
    graph = gm.graph
    rank = _get_rank()

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
            
            if rank == 0:
                print("\n=== FX GRAPH BEFORE LOWERING ===")
                print(graph)
                print("\n=== FX GRAPH LOWERING ===")
                if use_async:
                    print(">> Lowering _c10d_functional.broadcast + wait_tensor → spyre.broadcast_async + spyre.wait")
                else:
                    print(">> Lowering _c10d_functional.broadcast + wait_tensor → spyre.broadcast")

            if use_async:
                # Lower to async pattern: broadcast_async + wait
                with graph.inserting_after(broadcast_node):
                    # Create broadcast_async call (returns handle)
                    async_node = graph.call_function(
                        torch.ops.spyre.broadcast_async.default,
                        args=broadcast_node.args,
                        kwargs=broadcast_node.kwargs,
                    )
                
                # Replace wait_tensor with spyre.wait
                if wait_users:
                    for wait_node in wait_users:
                        with graph.inserting_after(async_node):
                            # spyre.wait takes (handle, tensor) and returns tensor
                            new_wait = graph.call_function(
                                torch.ops.spyre.wait.default,
                                args=(async_node, broadcast_node.args[0]),  # handle, tensor
                            )
                        wait_node.replace_all_uses_with(new_wait)
                        graph.erase_node(wait_node)
                else:
                    # No wait_tensor found - insert wait immediately after broadcast_async
                    with graph.inserting_after(async_node):
                        new_wait = graph.call_function(
                            torch.ops.spyre.wait.default,
                            args=(async_node, broadcast_node.args[0]),
                        )
                    broadcast_node.replace_all_uses_with(new_wait)
                
                graph.erase_node(broadcast_node)
            else:
                # Lower to synchronous pattern: broadcast
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

            if rank == 0:
                print("\n=== FX GRAPH AFTER LOWERING ===")
                print(graph)

    graph.lint()
    gm.recompile()
    return gm