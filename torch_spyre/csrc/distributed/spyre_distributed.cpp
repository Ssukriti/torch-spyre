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
#include <unordered_map>
#include <mutex>

#include "../logging.h"
#include "../spyre_allocator.h"
#include "../spyre_tensor_impl.h"

namespace spyre {

// Structure to hold pending async work
// Keeps both the WorkSchedule and the output tensor to prevent premature deallocation
struct PendingWork {
  std::shared_ptr<spyre_comms::WorkSchedule> work;
  at::Tensor output;  // Keeps storage alive until wait completes
};

// Global map to track pending WorkSchedules for async operations
// Key: tensor data pointer (stable across calls), Value: PendingWork
static std::unordered_map<void*, PendingWork> pending_work_map_;
static std::mutex work_map_mutex_;

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

// Async broadcast implementation - returns immediately
at::Tensor spyre_broadcast_async_impl(
    const at::Tensor& input,
    int64_t src_rank,
    const std::string& group_name) {
  
  DEBUGINFO("spyre::broadcast_async called with src_rank=", src_rank, ", group=", group_name);
  
  // Get world context
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library();
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  }

  // Create output tensor
  at::Tensor output = at::empty_like(input);

  // Get stable data pointer for map key (this is guaranteed stable)
  void* data_ptr = output.data_ptr();
  TORCH_CHECK(data_ptr != nullptr, "Output tensor has null data pointer");

  // Get CompositeAddress for spyre-comms
  const flex::CompositeAddress* device_addr = get_composite_address(output);
  TORCH_CHECK(device_addr != nullptr, "Failed to get CompositeAddress from output tensor");

  // Convert PyTorch tensor metadata to spyre_comms format
  spyre_comms::TensorDataTypeEnum dtype = torch_dtype_to_spyre_comms(input.scalar_type());
  
  std::vector<int64_t> shape_vec;
  for (int64_t i = 0; i < input.dim(); i++) {
    shape_vec.push_back(input.size(i));
  }
  spyre_comms::TensorShape shape(shape_vec);
  spyre_comms::TensorInfo tensor_info(dtype, shape);

  // Create spyre_comms Tensor with device address
  spyre_comms::Tensor buffer_tensor(tensor_info);
  buffer_tensor.SetSpyreDeviceAddress(device_addr);

  // Copy input to output if we're the source rank
  int current_rank = context->getRank();
  if (current_rank == src_rank) {
    output.copy_(input);
  }

  // Start broadcast (non-blocking)
  auto work_schedule = context->broadcast(buffer_tensor, static_cast<spyre_comms::process_id_t>(src_rank));
  TORCH_CHECK(work_schedule != nullptr, "Broadcast operation failed to create work schedule");
  
  work_schedule->start();  // Start but DON'T wait
  
  // Store WorkSchedule AND output tensor in map for later synchronization
  // The output tensor reference keeps the storage alive until wait completes
  // Use data_ptr as key (stable) and std::move to transfer ownership of unique_ptr
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    pending_work_map_[data_ptr] = PendingWork{
        std::move(work_schedule),
        output  // Keep tensor alive
    };
    DEBUGINFO("Stored PendingWork at data_ptr=", data_ptr, ", pending_work_map size=", pending_work_map_.size());
  }

  return output;  // Return immediately without waiting
}

// Wait for async operation to complete
at::Tensor spyre_wait_work_impl(const at::Tensor& tensor) {
  DEBUGINFO("spyre::wait_work called");
  
  // Get stable data pointer for map lookup
  void* data_ptr = tensor.data_ptr();
  TORCH_CHECK(data_ptr != nullptr, "Tensor has null data pointer");
  
  // Find and wait on WorkSchedule
  PendingWork pending;
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    auto it = pending_work_map_.find(data_ptr);
    TORCH_CHECK(it != pending_work_map_.end(),
                "No pending async work found for tensor at data_ptr=", data_ptr,
                ". wait_work must be called on a tensor returned from broadcast_async.");
    
    pending = std::move(it->second);
    pending_work_map_.erase(it);
    DEBUGINFO("Found PendingWork at data_ptr=", data_ptr, ", waiting... pending_work_map size=", pending_work_map_.size());
  }
  
  // Wait for communication to complete
  pending.work->wait();
  DEBUGINFO("WorkSchedule wait completed");
  
  // Return the output tensor (which we kept alive)
  return pending.output;
}


}  // namespace spyre

// Define the spyre namespace and operations
TORCH_LIBRARY(spyre, m) {
  m.def("broadcast_async(Tensor input, int src_rank, str group_name) -> Tensor");
  m.def("wait_work(Tensor tensor) -> Tensor");
}

// Register the implementations with PyTorch's dispatcher
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("broadcast_async", &spyre::spyre_broadcast_async_impl);
  m.impl("wait_work", &spyre::spyre_wait_work_impl);
}

// Made with Bob