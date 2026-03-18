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


"""
## To run
torchrun --nproc-per-node=2 examples/all_reduce_demo_multirank_cpu.py

## Fx graph lowering succesful

=== FX GRAPH BEFORE LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %y_1 : [num_users=1] = call_function[target=torch.ops._c10d_functional.all_reduce](args = (%y, sum, default), kwargs = {})
    %y_2 : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor](args = (%y_1,), kwargs = {})
    %y_3 : [num_users=1] = call_method[target=clone](args = (%y_2,), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=operator.add](args = (%y_3, %y_3), kwargs = {})
    return (add_1,)
>> Lowering _c10d_functional.all_reduce + wait_tensor → spyre.all_reduce

=== FX GRAPH AFTER LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %all_reduce_1 : [num_users=1] = call_function[target=torch.ops.spyre.all_reduce](args = (%y, sum, default), kwargs = {})
    %y_3 : [num_users=1] = call_method[target=clone](args = (%all_reduce_1,), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=operator.add](args = (%y_3, %y_3), kwargs = {})
    return (add_1,)

## Succesful output

Spyre all_reduce called on CPU
Spyre all_reduce called on CPU

[Rank 0] Output:
tensor([[-6.3130, -3.9648, -0.3072,  1.5623],
        [-4.5307, -1.9031, -4.9918,  2.7599],
        [11.4037,  9.2552, -0.9911, -1.5492],
        [-4.5383, -3.9433,  9.4816, -7.9006]])


[Rank 1] Output:
tensor([[-6.3130, -3.9648, -0.3072,  1.5623],
        [-4.5307, -1.9031, -4.9918,  2.7599],
        [11.4037,  9.2552, -0.9911, -1.5492],
        [-4.5383, -3.9433,  9.4816, -7.9006]])

"""