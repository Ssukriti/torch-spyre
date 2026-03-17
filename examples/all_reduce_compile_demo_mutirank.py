
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.distributed.distributed_c10d as c10d

# def run_demo():
#     # Each rank uses a different Spyre device - defined by runtime based on RANK
#     device = torch.device(f"spyre")

#     # Initialize process group
#     dist.init_process_group(
#         backend="gloo"
#     )
#     comm_size = dist.get_world_size()
#     comm_rank = dist.get_rank()
#     print(f"Rank {comm_rank} using device {device}")

#     # Register group name expected by functional collectives
#     c10d._register_process_group("default", dist.group.WORLD)

#     # Create Spyre tensor on rank-specific device
#     x = torch.randn(4, 4).to(device)

#     # Function to compile
#     def fn(t):
#         y = t + t
#         y = torch.ops._c10d_functional.all_reduce(y, "sum", "default")
#         y = torch.ops._c10d_functional.wait_tensor(y)
#         y = y.clone()
#         return y + y

#     compiled_fn = torch.compile(fn)

#     # Run function
#     out = compiled_fn(x)

#     print(f"\n[Rank {comm_rank}] Output:\n{out}\n")

#     # FX graph before lowering
#     gm = torch._dynamo.export(fn, x, aten_graph=True)
#     print(f"[Rank {comm_rank}] FX graph before lowering:\n{gm.graph}\n")

#     # FX graph after lowering
#     gm_lowered = torch._dynamo.export(fn, x, aten_graph=True, backend="inductor")
#     print(f"[Rank {comm_rank}] FX graph after lowering:\n{gm_lowered.graph}\n")

#     # Cleanup
#     if dist.is_initialized():
#         dist.destroy_process_group()

#     del out, x


# if __name__ == "__main__":
#     run_demo()

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

def run_demo():
    device = torch.device("spyre")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size} using device {device}")

    c10d._register_process_group("default", dist.group.WORLD)

    x = torch.randn(4, 4).to(device)

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