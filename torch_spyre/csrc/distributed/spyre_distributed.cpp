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

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "logging.h"
#include "spyre_allocator.h"
#include "spyre_tensor_impl.h"

namespace spyre {

// ============================================================================
// WorkSchedule Handle Management for Async Operations
// ============================================================================

// Global registry for managing WorkSchedule handles across async operations.
// This is necessary because:
// 1. spyre-comms returns unique_ptr<WorkSchedule> (ownership transfer)
// 2. PyTorch can't hold C++ unique_ptrs directly
// 3. We need to keep WorkSchedules alive between broadcast_async() and wait() calls
static std::unordered_map<int64_t, std::unique_ptr<spyre_comms::WorkSchedule>> g_work_schedules;
static std::atomic<int64_t> g_next_handle{1};
static std::mutex g_work_mutex;

// Register a WorkSchedule and return a unique handle
int64_t register_work_schedule(std::unique_ptr<spyre_comms::WorkSchedule> work) {
  std::lock_guard<std::mutex> lock(g_work_mutex);
  int64_t handle = g_next_handle.fetch_add(1);
  g_work_schedules[handle] = std::move(work);
  return handle;
}

// Wait on a WorkSchedule by handle and clean up
void wait_work_schedule(int64_t handle) {
  std::unique_ptr<spyre_comms::WorkSchedule> work;
  {
    std::lock_guard<std::mutex> lock(g_work_mutex);
    auto it = g_work_schedules.find(handle);
    TORCH_CHECK(it != g_work_schedules.end(),
                "Invalid WorkSchedule handle: ", handle);
    work = std::move(it->second);
    g_work_schedules.erase(it);
  }
  // Wait outside the lock to avoid holding it during potentially long operation
  work->wait();
}


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
  // If not initialized, initialize it first
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library();
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr,
                "Failed to get spyre-comms world context even after initialization. "
                "Make sure spyre-comms is properly configured and MPI/distributed environment is set up.");
  }

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

  // Start and wait for completion (synchronous)
  work_schedule->start();
  work_schedule->wait();

  return output;
}

// ============================================================================
// Async Broadcast Operations
// ============================================================================

// Async broadcast: starts communication and returns handle immediately
c10::SymInt spyre_broadcast_async_impl(const at::Tensor& input, int64_t src_rank) {
  TORCH_CHECK(input.is_privateuseone(), "Input tensor must be on Spyre device");
  
  // Auto-initialize spyre-comms if needed
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library();
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr,
                "Failed to get spyre-comms world context even after initialization. "
                "Make sure spyre-comms is properly configured and MPI/distributed environment is set up.");
  }

  // Get CompositeAddress for the input tensor
  const flex::CompositeAddress* device_addr = get_composite_address(input);
  TORCH_CHECK(device_addr != nullptr, "Failed to get CompositeAddress from input tensor");

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

  // Create and start WorkSchedule (non-blocking)
  auto work_schedule = context->broadcast(buffer_tensor, static_cast<spyre_comms::process_id_t>(src_rank));
  TORCH_CHECK(work_schedule != nullptr, "Broadcast operation failed to create work schedule");
  
  work_schedule->start();  // Non-blocking: launches communication
  
  // Register WorkSchedule and return handle as SymInt
  int64_t handle = register_work_schedule(std::move(work_schedule));
  
  return c10::SymInt(handle);
}

// Wait for async operation to complete
at::Tensor spyre_wait_impl(c10::SymInt handle, const at::Tensor& tensor) {
  // Wait for the WorkSchedule to complete (extract int64_t from SymInt)
  wait_work_schedule(handle.guard_int(__FILE__, __LINE__));
  
  // Return the tensor (now contains completed communication result)
  return tensor;
}

// Register the implementations with PyTorch's dispatcher
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("broadcast", &spyre_broadcast_impl);
  m.impl("broadcast_async", &spyre_broadcast_async_impl);
  m.impl("wait", &spyre_wait_impl);
}

}  // namespace spyre

// Made with Bob