import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


def run_demo():
    device = torch.device("cpu")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size} using device {device}")

    c10d._register_process_group("default", dist.group.WORLD)

    x = torch.randn(4, 4, device=device)

    def fn(t):
        y = t + t
        y = torch.ops._c10d_functional.all_reduce(y, "sum", "default")
        y = torch.ops._c10d_functional.wait_tensor(y)
        y = y.clone()
        return y + y

    compiled_fn = torch.compile(fn)
    out = compiled_fn(x)

    print(f"\n[Rank {rank}] Output:\n{out}\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_demo()