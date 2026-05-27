/*
 * Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/ScalarType.h>

#include <flex/flex.hpp>
#include <spyre_comms.hpp>
#include <spyre_comms_tensor.hpp>

#include <memory>
#include <string>
#include <vector>

#include "logging.h"
#include "spyre_allocator.h"
#include "spyre_tensor_impl.h"

namespace spyre {

// Helper to convert PyTorch ScalarType to spyre_comms TensorDataTypeEnum
spyre_comms::TensorDataTypeEnum torch_dtype_to_spyre_comms(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return spyre_comms::TensorDataTypeEnum::float32;
    case c10::ScalarType::Double:
      return spyre_comms::TensorDataTypeEnum::float64;
    case c10::ScalarType::Half:
      return spyre_comms::TensorDataTypeEnum::float16;
    case c10::ScalarType::BFloat16:
      return spyre_comms::TensorDataTypeEnum::bfloat16;
    case c10::ScalarType::Int:
      return spyre_comms::TensorDataTypeEnum::int32;
    case c10::ScalarType::Long:
      return spyre_comms::TensorDataTypeEnum::int64;
    case c10::ScalarType::Short:
      return spyre_comms::TensorDataTypeEnum::int16;
    case c10::ScalarType::Char:
      return spyre_comms::TensorDataTypeEnum::int8;
    case c10::ScalarType::Byte:
      return spyre_comms::TensorDataTypeEnum::uint8;
    case c10::ScalarType::Bool:
      return spyre_comms::TensorDataTypeEnum::boolean;
    default:
      TORCH_CHECK(false, "Unsupported dtype for spyre_comms: ", dtype);
  }
}

// Helper to get CompositeAddress from a Spyre tensor
const flex::CompositeAddress* get_composite_address(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "Tensor must be on Spyre device for distributed operations");
  
  TORCH_CHECK(tensor.is_contiguous(),
              "Tensor must be contiguous for distributed operations");
  
  auto* spyre_impl = static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  TORCH_CHECK(spyre_impl != nullptr, "SpyreTensorImpl is null");
  
  auto& storage = spyre_impl->storage();
  auto* data_ptr = storage.data_ptr().get();
  TORCH_CHECK(data_ptr != nullptr, "Storage data pointer is null");
  
  auto* ctx = static_cast<SharedOwnerCtx*>(storage.data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null");
  
  return &ctx->composite_addr;
}

// Broadcast implementation using spyre-comms C++ API
at::Tensor spyre_broadcast_impl(
    const at::Tensor& input,
    int64_t src_rank,
    const std::string& group_name) {
  
  DEBUGINFO("spyre::broadcast called with src_rank=", src_rank, "group=", group_name);
  
  // Get world context from spyre-comms
  auto context = spyre_comms::get_world_context();
  TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  
  // Create output tensor (same shape and dtype as input)
  at::Tensor output = at::empty_like(input);
  
  // Get CompositeAddress for the output tensor
  const flex::CompositeAddress* device_addr = get_composite_address(output);
  TORCH_CHECK(device_addr != nullptr, "Failed to get CompositeAddress from output tensor");
  
  // Convert PyTorch tensor metadata to spyre_comms format
  spyre_comms::TensorDataTypeEnum dtype = torch_dtype_to_spyre_comms(input.scalar_type());
  
  // Convert shape to vector<int64_t>
  std::vector<int64_t> shape_vec;
  for (int64_t i = 0; i < input.dim(); i++) {
    shape_vec.push_back(input.size(i));
  }
  spyre_comms::TensorShape shape(shape_vec);
  
  // Create TensorInfo
  spyre_comms::TensorInfo tensor_info(dtype, shape);
  
  // Create spyre_comms Tensor with device address
  spyre_comms::Tensor buffer_tensor(tensor_info);
  buffer_tensor.SetSpyreDeviceAddress(device_addr);
  
  // Copy input to output if we're the source rank
  int current_rank = context->getRank();
  if (current_rank == src_rank) {
    output.copy_(input);
  }
  
  // Perform broadcast using spyre-comms C++ API
  auto work_schedule = context->broadcast(buffer_tensor, static_cast<spyre_comms::process_id_t>(src_rank));
  TORCH_CHECK(work_schedule != nullptr, "Broadcast operation failed to create work schedule");
  
  // Start and wait for completion (synchronous for now)
  // All processes must call start() before any can proceed
  work_schedule->start();
  work_schedule->wait();
  
  return output;
}

// Register the implementation with PyTorch's dispatcher
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("broadcast", &spyre_broadcast_impl);
}

}  // namespace spyre

// Made with Bob
