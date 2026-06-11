# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Sequence

from sympy import Expr
import torch
from torch._inductor.utils import ir_dataclass
from torch._inductor.ir import (
    FixedLayout,
    IRNode,
    Reduction,
    ReductionHint,
    TensorBox,
)
from torch_spyre._C import SpyreTensorLayout

from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen,
)
from torch._inductor.virtualized import V
import sympy
from torch.utils._ordered_set import OrderedSet
import torch._inductor.ir as ir


@ir_dataclass
class SpyreReduction(Reduction):
    """
    This class extends Reduction with an op_info to enable spyre-specific information
    to be passed from lowering to codegen for reduction operations.

    We believe this is needed because reduction operations do not go through the same
    virtualized ops API as pointwise operations do after lowering.
    TODO: validate this belief.
    """

    op_info: Any

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        reduction_type,
        op_info=None,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: Optional[IRNode] = None,
    ) -> TensorBox:
        return TensorBox.create(
            SpyreReduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=inner_fn,
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
                op_info=op_info,
            )
        )


class FixedTiledLayout(FixedLayout):
    """
    A Tensor layout for a tensor that is on a Spyre device.
    It augments FixedLayout (the "host" tensor layout) with
    the device tensor layout and the information needed to map between them.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: list[Expr],
        stride: list[Expr],
        device_layout: SpyreTensorLayout,
    ) -> None:
        super().__init__(device, dtype, size, stride)
        self.device_layout: SpyreTensorLayout = device_layout
        self.allocation: dict[str, Any] = {}

    def __str__(self) -> str:
        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}, device_layout={self.device_layout})"
        )

    __repr__ = __str__


class SpyreConstantFallback(ir.ExternKernel):
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.generate_const_tensor_fallback(self)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence[str]:
        return []

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self, op_overload: torch._ops.OpOverload, value, dtype, device
    ) -> None:
        cpp_kernel_name = "aoti_torch_constant"
        layout = FixedLayout(device, dtype, [], [])
        super().__init__(
            None,
            layout,
            [],
            (value,),
            python_kernel_name="torch.ops.spyre.constant",
            cpp_kernel_name=cpp_kernel_name,
            op_overload=op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class SpyreEmptyFallback(ir.ExternKernel):
    """IR node for spyre.empty — emits spyre_empty_with_layout via make_buffer_allocation.

    should_allocate() returns True so the wrapper calls make_buffer_allocation.
    SpyrePythonWrapperCodegen.make_buffer_allocation emits
    spyre_empty_with_layout(size, stride, dtype, device_layout) when the layout is
    a FixedTiledLayout; the placeholder FixedLayout set at construction time must be
    replaced with a FixedTiledLayout before codegen runs (lower_pad_sequence does
    this immediately after calling run_node).  If the layout is never upgraded the
    wrapper falls back to the generic CPU allocator, which is incorrect on Spyre.
    codegen() is a no-op because the allocation IS the result — there is no
    separate kernel call.
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        pass

    def should_allocate(self) -> bool:
        return True

    def get_mutation_names(self) -> Sequence[str]:
        return []

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self,
        op_overload: torch._ops.OpOverload,
        size: list[Expr],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        stride = ir.FlexibleLayout.contiguous_strides(size)
        layout = FixedLayout(device, dtype, size, stride)
        super().__init__(
            None,
            layout,
            [],
            (),
            op_overload=op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

class SpyreBroadcastAsyncFallback(ir.ExternKernel):
    """IR node for spyre.broadcast_async — emits a runtime call to async broadcast.
    
    This starts the broadcast operation asynchronously and returns immediately,
    allowing computation to proceed while communication is in progress.
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        """Generate code to call torch.ops.spyre.broadcast_async at runtime."""
        print(f"\n{'='*70}")
        print(f"[IR CODEGEN] SpyreBroadcastAsyncFallback.codegen()")
        print(f"{'='*70}")

        # Get input tensor name
        input_tensor = self.inputs[0]
        input_name = input_tensor.codegen_reference()

        # Get constant args (src_rank, group_name)
        src_rank, group_name = self.constant_args

        print(f"  Input tensor: {input_name}")
        print(f"  src_rank: {src_rank}")
        print(f"  group_name: '{group_name}'")

        # Generate the async call
        output_name = self.get_name()
        generated_code = f"{output_name} = torch.ops.spyre.broadcast_async({input_name}, {src_rank}, '{group_name}')"
        
        print(f"\n  Generated code:")
        print(f"    {generated_code}")
        print(f"\n  This will dispatch to C++ spyre_broadcast_async_impl() at runtime")
        print(f"  Communication starts immediately, returns without blocking")
        print(f"{'='*70}\n")
        
        wrapper.writeline(generated_code)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence[str]:
        return []

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self,
        op_overload: torch._ops.OpOverload,
        x: IRNode,
        src_rank: int,
        group_name: str,
    ) -> None:
        # Async broadcast returns a tensor with the same layout as input
        layout = x.get_layout()
        super().__init__(
            None,
            layout,
            [x],
            (src_rank, group_name),
            python_kernel_name="torch.ops.spyre.broadcast_async",
            op_overload=op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class SpyreWaitWorkFallback(ir.ExternKernel):
    """IR node for spyre.wait_work — emits a runtime call to synchronize async operation.
    
    This blocks until the async broadcast operation completes.
    
    IMPORTANT: This node must NOT be fused into Spyre kernel partitions because:
    1. It's an ExternKernel that generates Python wrapper code
    2. Spyre kernel codegen expects all buffers in actuals list
    3. This creates buf2 outside the kernel, causing 'buf2 not in list' error
    """
    
    @property
    def group(self):
        """Force this node into its own scheduling group to prevent fusion."""
        return (type(self), "wait_work_barrier")

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        """Generate code to call torch.ops.spyre.wait_work at runtime."""
        print(f"\n{'='*70}")
        print(f"[IR CODEGEN] SpyreWaitWorkFallback.codegen()")
        print(f"{'='*70}")

        # Get input tensor name (the tensor from broadcast_async)
        input_tensor = self.inputs[0]
        input_name = input_tensor.codegen_reference()

        print(f"  Input tensor: {input_name}")

        # Generate the wait call
        output_name = self.get_name()
        generated_code = f"{output_name} = torch.ops.spyre.wait_work({input_name})"
        
        print(f"\n  Generated code:")
        print(f"    {generated_code}")
        print(f"\n  This will dispatch to C++ spyre_wait_work_impl() at runtime")
        print(f"  Blocks until async broadcast completes")
        print(f"{'='*70}\n")
        
        wrapper.writeline(generated_code)

    def should_allocate(self) -> bool:
        # Don't allocate - wait_work returns the same tensor (in-place)
        return False
    
    def has_side_effect(self) -> bool:
        # Mark as having side effects to prevent incorrect optimizations
        return True

    def get_mutation_names(self) -> Sequence[str]:
        """MUTATION HINT: Tell Inductor that this operation mutates the input.
        
        This forces Inductor to understand the read-after-write dependency
        across the wait boundary, preventing incorrect reordering.
        """
        if self.inputs and hasattr(self.inputs[0], "get_name"):
            return [self.inputs[0].get_name()]
        return []

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self,
        op_overload: torch._ops.OpOverload,
        x: IRNode,
    ) -> None:
        # Wait returns the same tensor (pass-through)
        layout = x.get_layout()
        super().__init__(
            None,
            layout,
            [x],
            (),  # No constant args
            python_kernel_name="torch.ops.spyre.wait_work",
            op_overload=op_overload,
        )
        # 1. Register the buffer with Inductor's graph tracking
        self.name = V.graph.register_buffer(self)
        
        # NOTE: We do NOT add alias hints here because Spyre's kernel codegen
        # expects all buffers to be in the actuals list. The mutation hint
        # in get_mutation_names() is sufficient for dependency tracking.
            
        V.graph.register_operation(self)
