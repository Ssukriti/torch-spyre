[1000840000@sukriti-dev torch-spyre]$ torchrun --nproc-per-node=2 examples/broadcast_demo_multirank.py
/usr/local/lib64/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Successfully patched _precompile_header to avoid openssl dependency
[MODULE LOAD] Registering fake implementation for spyre::broadcast
[MODULE LOAD] Fake implementation registered for spyre::broadcast
[MODULE LOAD] Registered lowering for torch.ops.spyre.broadcast.default
[MODULE LOAD] Lowering function: <function lower_spyre_broadcast at 0x7fcb92df3600>
[MODULE LOAD] Is in spyre_lowerings? True

[MODULE LOAD DEBUG] Checking registries:
  1. spyre_lowerings[torch.ops.spyre.broadcast.default] = <function lower_spyre_broadcast at 0x7fcb92df3600>
  2. lowering.lowerings[torch.ops.spyre.broadcast.default] = NOT FOUND
[MODULE LOAD DEBUG] If lowering.lowerings shows 'NOT FOUND', GraphLowering won't find it!

[REGISTER_LOWERINGS] Called
[REGISTER_LOWERINGS] TEMPORARILY NOT calling make_fallback for broadcast
[REGISTER_LOWERINGS] Testing if custom lowering works without fallback
/usr/local/lib64/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/usr/local/lib64/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Successfully patched _precompile_header to avoid openssl dependency
Successfully patched _precompile_header to avoid openssl dependency
[MODULE LOAD] Registering fake implementation for spyre::broadcast
[MODULE LOAD] Registering fake implementation for spyre::broadcast
[MODULE LOAD] Fake implementation registered for spyre::broadcast
[MODULE LOAD] Fake implementation registered for spyre::broadcast
[MODULE LOAD] Registered lowering for torch.ops.spyre.broadcast.default
[MODULE LOAD] Lowering function: <function lower_spyre_broadcast at 0x7fc5880b2f20>
[MODULE LOAD] Is in spyre_lowerings? True

[MODULE LOAD DEBUG] Checking registries:
  1. spyre_lowerings[torch.ops.spyre.broadcast.default] = <function lower_spyre_broadcast at 0x7fc5880b2f20>
  2. lowering.lowerings[torch.ops.spyre.broadcast.default] = NOT FOUND
[MODULE LOAD DEBUG] If lowering.lowerings shows 'NOT FOUND', GraphLowering won't find it!

[REGISTER_LOWERINGS] Called
[REGISTER_LOWERINGS] TEMPORARILY NOT calling make_fallback for broadcast
[REGISTER_LOWERINGS] Testing if custom lowering works without fallback
[MODULE LOAD] Registered lowering for torch.ops.spyre.broadcast.default
[MODULE LOAD] Lowering function: <function lower_spyre_broadcast at 0x7f1a26caef20>
[MODULE LOAD] Is in spyre_lowerings? True

[MODULE LOAD DEBUG] Checking registries:
  1. spyre_lowerings[torch.ops.spyre.broadcast.default] = <function lower_spyre_broadcast at 0x7f1a26caef20>
  2. lowering.lowerings[torch.ops.spyre.broadcast.default] = NOT FOUND
[MODULE LOAD DEBUG] If lowering.lowerings shows 'NOT FOUND', GraphLowering won't find it!

[REGISTER_LOWERINGS] Called
[REGISTER_LOWERINGS] TEMPORARILY NOT calling make_fallback for broadcast
[REGISTER_LOWERINGS] Testing if custom lowering works without fallback
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
Rank 0/2 using device spyre
Rank 1/2 using device spyre
Rank 1 - Initial tensor: tensor([0., 0., 0., 0.], device='spyre:0')
Rank 1 - Compiling function...
Rank 1 - Executing broadcast...
Spyre kernel placeholder for _c10d broadcast
Spyre kernel placeholder for _c10d wait_tensor
[DECOMP CHECK] torch.ops.spyre.broadcast.default in decompositions: False
[DECOMP CHECK] torch.ops.spyre.broadcast.default in decompositions: False
[ENABLE_SPYRE_LOWERINGS] Registering 30 spyre lowerings
[ENABLE_SPYRE_LOWERINGS] Looking for torch.ops.spyre.broadcast.default: True
[SPYRE LOWERING TABLE ENTRY] spyre.broadcast.default -> <function lower_spyre_broadcast at 0x7f1a26caef20>
[ENABLE_SPYRE_LOWERINGS] After registration, torch.ops.spyre.broadcast.default in lowering.lowerings: True
[DECOMP CHECK AFTER CM] torch.ops.spyre.broadcast.default in spyre_context_decompositions: False
FX graph before lowering
graph():
    %l_args_0_ : torch.Tensor [num_users=1] = placeholder[target=L_args_0_]
    %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%l_args_0_, 42.0), kwargs = {})
    return (mul_tensor,)

=== FX GRAPH BEFORE LOWERING ===
graph():
    %l_args_0_ : torch.Tensor [num_users=1] = placeholder[target=L_args_0_]
    %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%l_args_0_, 42.0), kwargs = {})
    return (mul_tensor,)

=== FX GRAPH AFTER LOWERING ===
graph():
    %l_args_0_ : torch.Tensor [num_users=1] = placeholder[target=L_args_0_]
    %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%l_args_0_, 42.0), kwargs = {})
    return (mul_tensor,)
FX graph after lowering
graph():
    %l_args_0_ : torch.Tensor [num_users=1] = placeholder[target=L_args_0_]
    %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%l_args_0_, 42.0), kwargs = {})
    return (mul_tensor,)
[BEFORE INDUCTOR] Graph has broadcast node: False
[CALLING INDUCTOR] About to call torch._inductor.compile_fx.compile_fx
[CALLING INDUCTOR] Graph module type: <class 'torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl'>
[CALLING INDUCTOR] Decompositions keys sample: [<OpOverload(op='aten.addcdiv', overload='default')>, <OpOverload(op='aten.addcdiv', overload='out')>, <OpOverload(op='aten.addcdiv_', overload='default')>, <OpOverload(op='aten.addcmul', overload='default')>, <OpOverload(op='aten.addcmul', overload='out')>]
[PRE-INDUCTOR] Registered broadcast lowering in global table
[PRE-INDUCTOR] torch.ops.spyre.broadcast.default in lowerings: True
[PRE-INDUCTOR] NOT adding decomposition - testing without it

================================================================================
[REGISTRY DEBUG] Checking lowering registries before calling Inductor:
================================================================================
[REGISTRY] op = spyre.broadcast.default

[REGISTRY] Global Inductor Registry (inductor_lowering.lowerings):
  - op in registry: True
  - registered function: <function lower_spyre_broadcast at 0x7f1a26caef20>
  - is our lower_spyre_broadcast: True
  - function name: lower_spyre_broadcast

[REGISTRY] Spyre-specific Registry (spyre_lowerings):
  - op in registry: True
  - registered function: <function lower_spyre_broadcast at 0x7f1a26caef20>
  - is our lower_spyre_broadcast: True

[REGISTRY] Expected state:
  ✓ op in inductor_lowering.lowerings: True
  ✓ is our lower_spyre_broadcast: True
================================================================================

[BEFORE _ORIG GRAPH CODE] Graph module code:
================================================================================



def forward(self, L_args_0_ : torch.Tensor):
    l_args_0_ = L_args_0_
    mul_tensor = torch.ops.aten.mul.Tensor(l_args_0_, 42.0);  l_args_0_ = None
    return (mul_tensor,)
    
================================================================================

[FINAL DECOMP CHECK] broadcast in decomps before _orig: False
[FINAL DECOMP CHECK] broadcast in decomps after pop: False

[CALLING _ORIG] About to call torch._inductor.compile_fx.compile_fx
[CALLING _ORIG] Invariants:
  ✓ Graph contains torch.ops.spyre.broadcast.default
  ✓ Global lowering table has lower_spyre_broadcast
  ✓ Decompositions do NOT contain broadcast
[ENABLE_SPYRE_LOWERINGS] Registering 30 spyre lowerings
[ENABLE_SPYRE_LOWERINGS] Looking for torch.ops.spyre.broadcast.default: True
[SPYRE LOWERING TABLE ENTRY] spyre.broadcast.default -> <function lower_spyre_broadcast at 0x7fc5880b2f20>
[ENABLE_SPYRE_LOWERINGS] After registration, torch.ops.spyre.broadcast.default in lowering.lowerings: True
[DECOMP CHECK AFTER CM] torch.ops.spyre.broadcast.default in spyre_context_decompositions: False
FX graph before lowering
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %y_1 : [num_users=1] = call_function[target=torch.ops._c10d_functional.broadcast](args = (%y, 0, default), kwargs = {})
    %y_2 : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor](args = (%y_1,), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%y_2, 2), kwargs = {})
    return (z,)

=== FX GRAPH BEFORE LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %y_1 : [num_users=1] = call_function[target=torch.ops._c10d_functional.broadcast](args = (%y, 0, default), kwargs = {})
    %y_2 : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor](args = (%y_1,), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%y_2, 2), kwargs = {})
    return (z,)
>> Lowering _c10d_functional.broadcast + wait_tensor → spyre.broadcast
   Original broadcast_node.args: (y, 0, 'default')
   Original broadcast_node.kwargs: {}
   Created node with target: spyre.broadcast.default
   New node type: <class 'torch._ops.OpOverload'>
   New node.args: (y, 0, 'default')
   New node.kwargs: {}
   Is target the OpOverload? True

=== FX GRAPH AFTER LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %broadcast_default : [num_users=1] = call_function[target=torch.ops.spyre.broadcast.default](args = (%y, 0, default), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%broadcast_default, 2), kwargs = {})
    return (z,)
FX graph after lowering
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %broadcast_default : [num_users=1] = call_function[target=torch.ops.spyre.broadcast.default](args = (%y, 0, default), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%broadcast_default, 2), kwargs = {})
    return (z,)
[BEFORE INDUCTOR] Graph has broadcast node: True
[BEFORE INDUCTOR] Broadcast node: spyre.broadcast.default, args: (y, 0, 'default')
[CALLING INDUCTOR] About to call torch._inductor.compile_fx.compile_fx
[CALLING INDUCTOR] Graph module type: <class 'torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl'>
[CALLING INDUCTOR] Decompositions keys sample: [<OpOverload(op='aten.addcdiv', overload='default')>, <OpOverload(op='aten.addcdiv', overload='out')>, <OpOverload(op='aten.addcdiv_', overload='default')>, <OpOverload(op='aten.addcmul', overload='default')>, <OpOverload(op='aten.addcmul', overload='out')>]
[PRE-INDUCTOR] Registered broadcast lowering in global table
[PRE-INDUCTOR] torch.ops.spyre.broadcast.default in lowerings: True
[PRE-INDUCTOR] NOT adding decomposition - testing without it

================================================================================
[REGISTRY DEBUG] Checking lowering registries before calling Inductor:
================================================================================
[REGISTRY] op = spyre.broadcast.default

[REGISTRY] Global Inductor Registry (inductor_lowering.lowerings):
  - op in registry: True
  - registered function: <function lower_spyre_broadcast at 0x7fc5880b2f20>
  - is our lower_spyre_broadcast: True
  - function name: lower_spyre_broadcast

[REGISTRY] Spyre-specific Registry (spyre_lowerings):
  - op in registry: True
  - registered function: <function lower_spyre_broadcast at 0x7fc5880b2f20>
  - is our lower_spyre_broadcast: True

[REGISTRY] Expected state:
  ✓ op in inductor_lowering.lowerings: True
  ✓ is our lower_spyre_broadcast: True
================================================================================

[BEFORE _ORIG GRAPH CODE] Graph module code:
================================================================================



def forward(self, L_t_ : torch.Tensor):
    l_t_ = L_t_
    y = l_t_ + l_t_;  l_t_ = None
    broadcast_default = torch.ops.spyre.broadcast.default(y, 0, 'default');  y = None
    z = broadcast_default * 2;  broadcast_default = None
    return (z,)
    
================================================================================

[FINAL DECOMP CHECK] broadcast in decomps before _orig: False
[FINAL DECOMP CHECK] broadcast in decomps after pop: False

[CALLING _ORIG] About to call torch._inductor.compile_fx.compile_fx
[CALLING _ORIG] Invariants:
  ✓ Graph contains torch.ops.spyre.broadcast.default
  ✓ Global lowering table has lower_spyre_broadcast
  ✓ Decompositions do NOT contain broadcast

================================================================================
[BROADCAST_IMPL] KERNEL CALLED!
[BROADCAST_IMPL] x.device=spyre:0, src_rank=0
[BROADCAST_IMPL] x.type=FunctionalTensor
[BROADCAST_IMPL] x.dispatch_keys=DispatchKeySet(PrivateUse1, Python, ADInplaceOrView, AutogradPrivateUse1, AutocastPrivateUse1, PythonTLSSnapshot)
[BROADCAST_IMPL] FUNCTIONALIZATION PATH: Detected FunctionalTensor
[BROADCAST_IMPL] Returning x.clone() to preserve op in graph

================================================================================
[BROADCAST_IMPL] KERNEL CALLED!
[BROADCAST_IMPL] x.device=spyre:0, src_rank=0
[BROADCAST_IMPL] x.type=FunctionalTensor
[BROADCAST_IMPL] x.dispatch_keys=DispatchKeySet(PrivateUse1, Python, ADInplaceOrView, AutogradPrivateUse1, AutocastPrivateUse1, PythonTLSSnapshot)
[BROADCAST_IMPL] FUNCTIONALIZATION PATH: Detected FunctionalTensor
[BROADCAST_IMPL] Returning x.clone() to preserve op in graph

[GRAPH_LOWERING] Starting graph lowering
[GRAPH_LOWERING] Graph has 5 nodes
[GRAPH_LOWERING] ALL NODES IN GRAPH:
[GRAPH_LOWERING]   0: op=placeholder, target=arg0_1
[GRAPH_LOWERING]   1: op=call_function, target=aten.add.Tensor
[GRAPH_LOWERING]   2: op=call_function, target=spyre.constant.default
[GRAPH_LOWERING]   3: op=call_function, target=aten.mul.Tensor
[GRAPH_LOWERING]   4: op=output, target=output

[GRAPH_LOWERING] Starting graph lowering
[GRAPH_LOWERING] Graph has 4 nodes
[GRAPH_LOWERING] ALL NODES IN GRAPH:
[GRAPH_LOWERING]   0: op=placeholder, target=arg0_1
[GRAPH_LOWERING]   1: op=call_function, target=spyre.constant.default
[GRAPH_LOWERING]   2: op=call_function, target=aten.mul.Tensor
[GRAPH_LOWERING]   3: op=output, target=output

[RUN_NODE] *** FOUND SPYRE/BROADCAST NODE ***
[RUN_NODE] Node: py_const
[RUN_NODE] Target: spyre.constant.default
[RUN_NODE] Op: call_function
[RUN_NODE] Args: (42.0, torch.float32, device(type='spyre'))
[RUN_NODE] Checking lowering.lowerings for: spyre.constant.default
[RUN_NODE] Is in lowering.lowerings? True
[RUN_NODE] Lowering function: <function lower_constant at 0x7f1a26c9b1a0>

[RUN_NODE] *** FOUND SPYRE/BROADCAST NODE ***
[RUN_NODE] Node: py_const
[RUN_NODE] Target: spyre.constant.default
[RUN_NODE] Op: call_function
[RUN_NODE] Args: (2, torch.float32, device(type='spyre'))
[RUN_NODE] Checking lowering.lowerings for: spyre.constant.default
[RUN_NODE] Is in lowering.lowerings? True
[RUN_NODE] Lowering function: <function lower_constant at 0x7fc5880c71a0>
[GRAPH_LOWERING] Finished graph lowering

[GRAPH_LOWERING] Finished graph lowering

[SPYRE SCHEDULER] codegen_node called with node type: <class 'torch._inductor.scheduler.SchedulerNode'>
[SPYRE SCHEDULER] Processing 1 nodes
[SPYRE SCHEDULER] codegen_node called with node type: <class 'torch._inductor.scheduler.FusedSchedulerNode'>
[SPYRE SCHEDULER] Processing 2 nodes
[AFTER _ORIG] Returned from compile_fx
[FINALLY] Keeping broadcast lowering in registry (not restoring)
[AFTER INDUCTOR] Compilation complete, result type: <class 'function'>
Rank 0 (ROOT) - Initial tensor: tensor([42., 42., 42., 42.], device='spyre:0')
Rank 0 - Compiling function...
Rank 0 - Executing broadcast...
Spyre kernel placeholder for _c10d broadcast
Spyre kernel placeholder for _c10d wait_tensor
[DECOMP CHECK] torch.ops.spyre.broadcast.default in decompositions: False
[ENABLE_SPYRE_LOWERINGS] Registering 30 spyre lowerings
[ENABLE_SPYRE_LOWERINGS] Looking for torch.ops.spyre.broadcast.default: True
[SPYRE LOWERING TABLE ENTRY] spyre.broadcast.default -> <function lower_spyre_broadcast at 0x7f1a26caef20>
[ENABLE_SPYRE_LOWERINGS] After registration, torch.ops.spyre.broadcast.default in lowering.lowerings: True
[DECOMP CHECK AFTER CM] torch.ops.spyre.broadcast.default in spyre_context_decompositions: False
FX graph before lowering
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %y_1 : [num_users=1] = call_function[target=torch.ops._c10d_functional.broadcast](args = (%y, 0, default), kwargs = {})
    %y_2 : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor](args = (%y_1,), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%y_2, 2), kwargs = {})
    return (z,)

=== FX GRAPH BEFORE LOWERING ===
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %y_1 : [num_users=1] = call_function[target=torch.ops._c10d_functional.broadcast](args = (%y, 0, default), kwargs = {})
    %y_2 : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor](args = (%y_1,), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%y_2, 2), kwargs = {})
    return (z,)
>> Lowering _c10d_functional.broadcast + wait_tensor → spyre.broadcast
   Original broadcast_node.args: (y, 0, 'default')
   Original broadcast_node.kwargs: {}
   Created node with target: spyre.broadcast.default
   New node type: <class 'torch._ops.OpOverload'>
   New node.args: (y, 0, 'default')
   New node.kwargs: {}
   Is target the OpOverload? True

=== FX GRAPH AFTER LOWERING ===
[AFTER _ORIG] Returned from compile_fxgraph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %broadcast_default : [num_users=1] = call_function[target=torch.ops.spyre.broadcast.default](args = (%y, 0, default), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%broadcast_default, 2), kwargs = {})
    return (z,)

[FINALLY] Keeping broadcast lowering in registry (not restoring)
[AFTER INDUCTOR] Compilation complete, result type: <class 'function'>
FX graph after lowering
graph():
    %l_t_ : torch.Tensor [num_users=1] = placeholder[target=L_t_]
    %y : [num_users=1] = call_function[target=operator.add](args = (%l_t_, %l_t_), kwargs = {})
    %broadcast_default : [num_users=1] = call_function[target=torch.ops.spyre.broadcast.default](args = (%y, 0, default), kwargs = {})
    %z : [num_users=1] = call_function[target=operator.mul](args = (%broadcast_default, 2), kwargs = {})
    return (z,)
[BEFORE INDUCTOR] Graph has broadcast node: True
[BEFORE INDUCTOR] Broadcast node: spyre.broadcast.default, args: (y, 0, 'default')
[CALLING INDUCTOR] About to call torch._inductor.compile_fx.compile_fx
[CALLING INDUCTOR] Graph module type: <class 'torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl'>
[CALLING INDUCTOR] Decompositions keys sample: [<OpOverload(op='aten.addcdiv', overload='default')>, <OpOverload(op='aten.addcdiv', overload='out')>, <OpOverload(op='aten.addcdiv_', overload='default')>, <OpOverload(op='aten.addcmul', overload='default')>, <OpOverload(op='aten.addcmul', overload='out')>]
[PRE-INDUCTOR] Registered broadcast lowering in global table
[PRE-INDUCTOR] torch.ops.spyre.broadcast.default in lowerings: True
[PRE-INDUCTOR] NOT adding decomposition - testing without it

================================================================================
[REGISTRY DEBUG] Checking lowering registries before calling Inductor:
================================================================================
[REGISTRY] op = spyre.broadcast.default

[REGISTRY] Global Inductor Registry (inductor_lowering.lowerings):
  - op in registry: True
  - registered function: <function lower_spyre_broadcast at 0x7f1a26caef20>
  - is our lower_spyre_broadcast: True
  - function name: lower_spyre_broadcast

[REGISTRY] Spyre-specific Registry (spyre_lowerings):
  - op in registry: True
  - registered function: <function lower_spyre_broadcast at 0x7f1a26caef20>
  - is our lower_spyre_broadcast: True

[REGISTRY] Expected state:
  ✓ op in inductor_lowering.lowerings: True
  ✓ is our lower_spyre_broadcast: True
================================================================================

[BEFORE _ORIG GRAPH CODE] Graph module code:
================================================================================



def forward(self, L_t_ : torch.Tensor):
    l_t_ = L_t_
    y = l_t_ + l_t_;  l_t_ = None
    broadcast_default = torch.ops.spyre.broadcast.default(y, 0, 'default');  y = None
    z = broadcast_default * 2;  broadcast_default = None
    return (z,)
    
================================================================================

[FINAL DECOMP CHECK] broadcast in decomps before _orig: False
[FINAL DECOMP CHECK] broadcast in decomps after pop: False

[CALLING _ORIG] About to call torch._inductor.compile_fx.compile_fx
[CALLING _ORIG] Invariants:
  ✓ Graph contains torch.ops.spyre.broadcast.default
  ✓ Global lowering table has lower_spyre_broadcast
  ✓ Decompositions do NOT contain broadcast
Rank 1 - After broadcast: tensor([0., 0., 0., 0.], device='spyre:0')

[Rank 1] Output shape: torch.Size([8, 8])

[AFTER _ORIG] Returned from compile_fx
[FINALLY] Keeping broadcast lowering in registry (not restoring)
[AFTER INDUCTOR] Compilation complete, result type: <class 'function'>
Rank 0 - After broadcast: tensor([168., 168., 168., 168.], device='spyre:0')

[Rank 0] Output shape: torch.Size([8, 8])

[10