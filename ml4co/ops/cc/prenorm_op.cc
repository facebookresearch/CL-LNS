// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#include "ml4co/ops/cc/prenorm_op.h"

#include <cmath>
#include <cstring>
#include <tuple>
#include <vector>

namespace ml4co {

namespace {

template <typename T>
void PrenormCPUKernelImpl(const torch::Tensor& X, int64_t running_m0,
                          const torch::Tensor& running_m1,
                          const torch::Tensor& running_m2, T eps,
                          torch::Tensor& saved_m1, torch::Tensor& saved_m2,
                          torch::Tensor& scale, torch::Tensor& bias) {
  const int64_t outer_size = X.size(0);
  const int64_t inner_size = X.size(1);
  const T* X_data = X.data_ptr<T>();
  const T* running_m1_data = running_m1.data_ptr<T>();
  const T* running_m2_data = running_m2.data_ptr<T>();
  T* saved_m1_data = saved_m1.data_ptr<T>();
  T* saved_m2_data = saved_m2.data_ptr<T>();
  T* scale_data = scale.data_ptr<T>();
  T* bias_data = bias.data_ptr<T>();

  std::memcpy(saved_m1_data, running_m1_data, inner_size * sizeof(T));
  std::memcpy(saved_m2_data, running_m2_data, inner_size * sizeof(T));
  for (int64_t i = 0; i < outer_size; ++i) {
    const T c = T(1) / static_cast<T>(running_m0 + i + 1);
    const T* X_ptr = X_data + i * inner_size;
    for (int64_t j = 0; j < inner_size; ++j) {
      const T delta = X_ptr[j] - saved_m1_data[j];
      saved_m1_data[j] += delta * c;
      saved_m2_data[j] += delta * (X_ptr[j] - saved_m1_data[j]);
    }
  }
  const T cnt = running_m0 + outer_size;
  for (int64_t i = 0; i < inner_size; ++i) {
    const T rstd =
        T(1) / std::sqrt(saved_m2_data[i] / static_cast<T>(cnt) + eps);
    scale_data[i] = rstd;
    bias_data[i] = -rstd * saved_m1_data[i];
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Prenorm(
    const torch::Tensor& input, int64_t running_m0,
    const torch::Tensor& running_m1, torch::Tensor& running_m2, double eps) {
  TORCH_CHECK(input.dim() == 2);
  const int64_t inner_size = input.size(1);
  TORCH_CHECK(running_m1.numel() == inner_size);
  TORCH_CHECK(running_m2.numel() == inner_size);
  torch::Tensor saved_m1 = torch::empty({inner_size}, input.options());
  torch::Tensor saved_m2 = torch::empty({inner_size}, input.options());
  torch::Tensor scale = torch::empty({inner_size}, input.options());
  torch::Tensor bias = torch::empty({inner_size}, input.options());
  if (input.device().is_cpu()) {
    PrenormCPUKernel(input.contiguous(), running_m0, running_m1, running_m2,
                     eps, saved_m1, saved_m2, scale, bias);
  } else if (input.device().is_cuda()) {
#ifndef ML4CO_CPU_ONLY
    PrenormCUDAKernel(input.contiguous(), running_m0, running_m1, running_m2,
                      eps, saved_m1, saved_m2, scale, bias);
#else   // ML4CO_CPU_ONLY
    TORCH_CHECK(false, "CUDA is not available.");
#endif  // ML4CO_CPU_ONLY
  } else {
    TORCH_CHECK(false, "Unsupported device.")
  }
  return std::make_tuple(saved_m1, saved_m2, scale, bias);
}

void PrenormCPUKernel(const torch::Tensor& X, int64_t running_m0,
                      const torch::Tensor& running_m1,
                      const torch::Tensor& running_m2, double eps,
                      torch::Tensor& saved_m1, torch::Tensor& saved_m2,
                      torch::Tensor& scale, torch::Tensor& bias) {
  AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "PrenormCPUKernel", [&]() {
    PrenormCPUKernelImpl<scalar_t>(X, running_m0, running_m1, running_m2,
                                   static_cast<scalar_t>(eps), saved_m1,
                                   saved_m2, scale, bias);
  });
}

}  // namespace ml4co
