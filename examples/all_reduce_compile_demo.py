import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

dist.init_process_group(
    backend="gloo",
    rank=0,
    world_size=1,
)
# Register the group name expected by functional collectives
c10d._register_process_group("default", dist.group.WORLD)
pg = dist.group.WORLD
DEVICE = torch.device("spyre")

def fn(x):

    y = x + x

    y = torch.ops._c10d_functional.all_reduce(
        y,
        "sum",
        "default",
    )
    y = torch.ops._c10d_functional.wait_tensor(y)
    y = y.clone()

    return y + y


x = torch.randn(4,4).to(DEVICE)

compiled_fn = torch.compile(fn)

out = compiled_fn(x)

print("Output:", out)

####
# export MASTER_ADDR=localhost
# export MASTER_PORT=29500
# python3 examples/all_reduce_compile_demo.py