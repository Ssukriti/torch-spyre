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

"""
## To run
torchrun --nproc-per-node=2 examples/all_reduce_compile_demo_multirank.py

## FX graph lowering successful

=== FX GRAPH BEFORE LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %y_1 : [num_users=1] = call_function[target=torch.ops._c10d_functional.all_reduce](args = (%y, sum, default), kwargs = {})
    %y_2 : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor](args = (%y_1,), kwargs = {})
    %y_3 : [num_users=1] = call_method[target=clone](args = (%y_2,), kwargs = {})
    return (y_3,)
>> Lowering _c10d_functional.all_reduce + wait_tensor → spyre.all_reduce

=== FX GRAPH AFTER LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %all_reduce_1 : [num_users=1] = call_function[target=torch.ops.spyre.all_reduce](args = (%y, sum, default), kwargs = {})
    %y_3 : [num_users=1] = call_method[target=clone](args = (%all_reduce_1,), kwargs = {})
    return (y_3,)


## The crash 

Spyre all_reduce called
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1] Backend compiler exception
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]   Explanation: Backend compiler `inductor` failed with c10d.allreduce_.default
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1] 
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     While executing %all_reduce_1 : [num_users=1] = call_function[target=torch.ops.spyre.all_reduce](args = (%y, sum, default), kwargs = {})
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     Original traceback:
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     None
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs). Adding a graph break.
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]   Hint: Report an issue to the backend compiler repo.
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1] 
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]   Developer debug context: Backend: inductor
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     Exception:c10d.allreduce_.default
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1] 
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     While executing %all_reduce_1 : [num_users=1] = call_function[target=torch.ops.spyre.all_reduce](args = (%y, sum, default), kwargs = {})
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     Original traceback:
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     None
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]     Traceback:
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]       File "/home/senuser/dt_inductor/torch-spyre/examples/all_reduce_compile_demo_multirank.py", line 23, in fn
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]         return y + y
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1] 
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1] 
[rank1]:W0318 17:49:19.235000 45168 torch/_dynamo/exc.py:593] [0/0_1]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0219.html

...exception:

Spyre kernel placeholder for _c10d all_reduce
Spyre kernel placeholder for _c10d all_reduce
Spyre kernel placeholder for _c10d wait_tensor
Spyre kernel placeholder for _c10d wait_tensor
corrupted size vs. prev_size
Signal Received: 6 (Aborted)
Signal Received from pid=45167 
*** BACKTRACE ***
"""