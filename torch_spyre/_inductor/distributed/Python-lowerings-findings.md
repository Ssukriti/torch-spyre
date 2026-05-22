## Python binding path 

### Results in no-op in compile path . Needs C++ dispatcher.

Findings:
- Graph does lower 
_c10d_functional.broadcast + wait_tensor
→ torch.ops.spyre.broadcast_

- With eager mode python op works 
torch.ops.spyre.broadcast_(x, src_rank=0, group_name="default")

- Compiled mode does not call the Python impl
In torch.compile / Inductor, the graph contains torch.ops.spyre.broadcast_, but the Python spyre_broadcast_impl() prints never appear. Rank 1 remains unchanged.

That means Inductor is not emitting a runtime call back into the Python implementation.

Conclusion :
For production, the op should be a real Torch-Spyre C++ dispatcher op

spyre::broadcast_(Tensor(a!) x, int src_rank, str group_name) -> Tensor(a!)

implemented in C++, so Inductor/runtime can preserve and execute it as a side-effecting op.


```
Output of compile path 

>> Lowering _c10d_functional.broadcast + wait_tensor → spyre.broadcast_

=== FX GRAPH AFTER LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %broadcast_ : [num_users=1] = call_function[target=torch.ops.spyre.broadcast_](args = (%y, 0, default), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%broadcast_, 2), kwargs = {})
    return (z,)
FX graph after lowering
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %broadcast_ : [num_users=1] = call_function[target=torch.ops.spyre.broadcast_](args = (%y, 0, default), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%broadcast_, 2), kwargs = {})
    return (z,)
Rank 1 - After broadcast: tensor([0., 0., 0., 0.], device='spyre:0')

[Rank 1] Output shape: torch.Size([8, 8])

Rank 0 - After broadcast: tensor([168., 168., 168., 168.], device='spyre:0')

[Rank 0] Output shape: torch.Size([8, 8])
```


### Python binding path is fragile for production

It requires Python to manually do low-level runtime plumbing:

composite_addr_ptr = torch_spyre._C.get_composite_address_ptr(x)
buffer_tensor.set_spyre_device_address(composite_addr_ptr)

That exposes internal device-address/lifetime details at Python level.

Torch-Spyre C++ already owns the knowledge of:

at::Tensor → Spyre storage → flex::CompositeAddress

spyre-comms only needs the address. Python should not own this interop boundary.