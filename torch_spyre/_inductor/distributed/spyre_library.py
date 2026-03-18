import torch

@torch.library.custom_op("spyre::all_reduce", mutates_args=())
def all_reduce(x: torch.Tensor, reduce_op: str = "sum", group_name: str = "default") -> torch.Tensor:
    raise NotImplementedError("spyre::all_reduce should dispatch to a registered runtime kernel")


@torch.library.register_fake("spyre::all_reduce")
def _(x, reduce_op="sum", group_name="default"):
    print("FAKE spyre custom_op called")
    return torch.empty_like(x, device="meta")