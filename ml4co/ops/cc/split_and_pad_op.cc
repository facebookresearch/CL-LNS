// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#include "ml4co/ops/cc/split_and_pad_op.h"

#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <algorithm>
#include <cstring>

namespace ml4co {

namespace {

template <typename T>
void SplitAndPadCPUKernelImpl(const torch::Tensor& X,
                              const torch::Tensor& sizes,
                              const torch::Tensor& strides, T value,
                              torch::Tensor& Y) {
  const int64_t outer_size = Y.size(0);
  const int64_t inner_size = Y.size(1);
  const T* X_data = X.data_ptr<T>();
  const int64_t* sizes_data = sizes.data_ptr<int64_t>();
  const int64_t* strides_data = strides.data_ptr<int64_t>();
  T* Y_data = Y.data_ptr<T>();
  at::parallel_for(0, outer_size, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t cur_size = sizes_data[i];
      const T* X_ptr = i == 0 ? X_data : X_data + strides_data[i - 1];
      T* Y_ptr = Y_data + i * inner_size;
      std::memcpy(Y_ptr, X_ptr, cur_size * sizeof(T));
      std::fill(Y_ptr + cur_size, Y_ptr + inner_size, value);
    }
  });
}

template <typename T>
void SplitAndPadBackwardCPUKernelImpl(const torch::Tensor& dY,
                                      const torch::Tensor& sizes,
                                      const torch::Tensor& strides,
                                      torch::Tensor& dX) {
  const int64_t outer_size = dY.size(0);
  const int64_t inner_size = dY.size(1);
  const T* dY_data = dY.data_ptr<T>();
  const int64_t* sizes_data = sizes.data_ptr<int64_t>();
  const int64_t* strides_data = strides.data_ptr<int64_t>();
  T* dX_data = dX.data_ptr<T>();
  at::parallel_for(0, outer_size, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t cur_size = sizes_data[i];
      const T* dY_ptr = dY_data + i * inner_size;
      T* dX_ptr = i == 0 ? dX_data : dX_data + strides_data[i - 1];
      std::memcpy(dX_ptr, dY_ptr, cur_size * sizeof(T));
    }
  });
}

}  // namespace

torch::Tensor SplitAndPad(const torch::Tensor& input,
                          const torch::Tensor& sizes, int64_t feature_size,
                          double value) {
  TORCH_CHECK(input.dim() == 1);
  TORCH_CHECK(sizes.dim() == 1);
  const torch::Tensor strides = sizes.cumsum(/*dim=*/0);
  const int64_t outer_size = sizes.numel();
  const int64_t inner_size =
      std::max(feature_size, sizes.max().item<int64_t>());
  torch::Tensor output =
      torch::empty({outer_size, inner_size}, input.options());
  if (input.device().is_cpu()) {
    SplitAndPadCPUKernel(input.contiguous(), sizes, strides, value, output);
  } else if (input.device().is_cuda()) {
#ifndef ML4CO_CPU_ONLY
    SplitAndPadCUDAKernel(input.contiguous(), sizes, strides, value, output);
#else   // ML4CO_CPU_ONLY
    TORCH_CHECK(false, "CUDA is not available.");
#endif  // ML4CO_CPU_ONLY
  } else {
    TORCH_CHECK(false, "Unsupported device.")
  }
  return output;
}

torch::Tensor SplitAndPadBackward(const torch::Tensor& grad_output,
                                  const torch::Tensor& sizes) {
  TORCH_CHECK(grad_output.dim() == 2);
  TORCH_CHECK(sizes.dim() == 1);
  const torch::Tensor strides = sizes.cumsum(/*dim=*/0);
  const int64_t input_size = sizes.sum().item<int64_t>();
  torch::Tensor grad_input = torch::empty({input_size}, grad_output.options());
  if (grad_output.device().is_cpu()) {
    SplitAndPadBackwardCPUKernel(grad_output.contiguous(), sizes, strides,
                                 grad_input);
  } else if (grad_output.device().is_cuda()) {
#ifndef ML4CO_CPU_ONLY
    SplitAndPadBackwardCUDAKernel(grad_output.contiguous(), sizes, strides,
                                  grad_input);
#else   // ML4CO_CPU_ONLY
    TORCH_CHECK(false, "CUDA is not available.");
#endif  // ML4CO_CPU_ONLY
  } else {
    TORCH_CHECK(false, "Unsupported device.");
  }
  return grad_input;
}

void SplitAndPadCPUKernel(const torch::Tensor& X, const torch::Tensor& sizes,
                          const torch::Tensor& strides, double value,
                          torch::Tensor& Y) {
  AT_DISPATCH_ALL_TYPES(X.scalar_type(), "SplitAndPadCPUKernel", [&]() {
    SplitAndPadCPUKernelImpl<scalar_t>(X, sizes, strides,
                                       static_cast<scalar_t>(value), Y);
  });
}

void SplitAndPadBackwardCPUKernel(const torch::Tensor& dY,
                                  const torch::Tensor& sizes,
                                  const torch::Tensor& strides,
                                  torch::Tensor& dX) {
  AT_DISPATCH_ALL_TYPES(
      dY.scalar_type(), "SplitAndPadBackwardCPUKernel", [&]() {
        SplitAndPadBackwardCPUKernelImpl<scalar_t>(dY, sizes, strides, dX);
      });
}

}  // namespace ml4co
