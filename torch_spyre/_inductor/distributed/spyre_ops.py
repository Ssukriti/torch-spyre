import torch
import torch.distributed as dist
import torch._dynamo


@torch._dynamo.disable
def spyre_all_reduce_async(x, reduce_op="sum", group_name="default"):
    print("Spyre lowering triggered for all_reduce_async")

    # CPU tensors needed for gloo backend. detach was needed
    # to prevent autograd errors 
    cpu_tensor = torch.clone(x).detach().cpu().contiguous()

    reduce_map = {
        "sum": dist.ReduceOp.SUM,
        "avg": dist.ReduceOp.AVG,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }

    op = reduce_map.get(str(reduce_op), dist.ReduceOp.SUM)

    dist.all_reduce(cpu_tensor, op=op)

    return {
        "cpu_tensor": cpu_tensor,
        "device": x.device,
    }


@torch._dynamo.disable
def spyre_wait(handle):
    print("Spyre wait called")

    cpu_tensor = handle["cpu_tensor"]
    device = handle["device"]

    spyre_tensor = torch.empty_like(cpu_tensor, device=device)
    spyre_tensor.copy_(cpu_tensor)

    return spyre_tensor