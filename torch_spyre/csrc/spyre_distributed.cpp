// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>

// Include flex first to avoid tracing issues
#include <flex/flex.hpp>

#include "spyre_tensor_impl.h"
#include "spyre_storage_impl.h"
#include "spyre_allocator.h"
#include "module.h"

// Include spyre-comms main header
#include <spyre_comms.hpp>

namespace torch_spyre {

// Forward declare the function from spyre namespace
namespace spyre {
  uintptr_t get_composite_address_ptr(const at::Tensor& tensor);
}

at::Tensor spyre_broadcast_(
    at::Tensor& x,
    int64_t src_rank,
    const std::string& group_name) {
  
  std::cout << "[C++ DISPATCHER] spyre_broadcast_ called! src_rank=" << src_rank << std::endl;
  
  TORCH_CHECK(x.device().is_privateuseone(),
              "spyre::broadcast_ expects Spyre tensor, got ", x.device());
  
  // Get spyre-comms world context
  auto comms_ctx = spyre_comms::get_world_context();
  TORCH_CHECK(comms_ctx != nullptr, "spyre-comms world context not initialized");
  
  // Ensure tensor is contiguous
  if (!x.is_contiguous()) {
    x = x.contiguous();
  }
  
  // Get tensor shape
  auto shape = x.sizes().vec();
  std::vector<int64_t> shape_vec(shape.begin(), shape.end());
  
  // Map torch dtype to spyre_comms dtype
  spyre_comms::TensorDataTypeEnum spyre_dtype;
  if (x.scalar_type() == at::ScalarType::Float) {
    spyre_dtype = spyre_comms::TensorDataTypeEnum::float32;
  } else if (x.scalar_type() == at::ScalarType::Half) {
    spyre_dtype = spyre_comms::TensorDataTypeEnum::float16;
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    spyre_dtype = spyre_comms::TensorDataTypeEnum::bfloat16;
  } else if (x.scalar_type() == at::ScalarType::Int) {
    spyre_dtype = spyre_comms::TensorDataTypeEnum::int32;
  } else if (x.scalar_type() == at::ScalarType::Long) {
    spyre_dtype = spyre_comms::TensorDataTypeEnum::int64;
  } else {
    TORCH_CHECK(false, "Unsupported dtype for spyre::broadcast_: ", x.scalar_type());
  }
  
  // Create spyre_comms tensor
  spyre_comms::TensorShape tensor_shape(shape_vec);
  spyre_comms::TensorInfo tensor_info(spyre_dtype, tensor_shape);
  spyre_comms::Tensor buffer_tensor(tensor_info);
  
  // Get CompositeAddress from Spyre tensor
  // We need the actual CompositeAddress pointer, not the uintptr_t
  auto* spyre_impl = static_cast<::spyre::SpyreTensorImpl*>(x.unsafeGetTensorImpl());
  auto& storage = spyre_impl->storage();
  auto* storage_ctx = static_cast<::spyre::SharedOwnerCtx*>(storage.data_ptr().get_context());
  auto* composite_addr = &storage_ctx->composite_addr;
  
  // Set the Spyre device address (no CPU copy)
  buffer_tensor.SetSpyreDeviceAddress(composite_addr);
  
  // Execute broadcast
  auto work = comms_ctx->broadcast(buffer_tensor, static_cast<spyre_comms::process_id_t>(src_rank));
  work->start();
  work->wait();
  
  // Return the same tensor (in-place operation)
  return x;
}

} // namespace torch_spyre

// Register the custom op with PyTorch dispatcher
// COMMENTED OUT FOR PYTHON BINDING APPROACH
// Uncomment for C++ dispatcher approach
/*
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("broadcast_", torch_spyre::spyre_broadcast_);
}
*/

// Made with Bob
