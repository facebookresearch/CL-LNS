// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#include "ml4co/ops/cc/prenorm_op.h"

#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>

#include <tuple>

#include "ml4co/ops/cc/cuda_utils.h"

namespace ml4co {

namespace {

template <typename T>
__global__ void MergeMomentsCUDAKernel(int64_t size, T eps,
                                       const int64_t m0_add, const T* m1_add,
                                       const T* m2_add, int64_t m0, T* m1,
                                       T* m2, T* scale, T* bias) {
  using T_ACC = at::acc_type<T, true>;
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) {
    return;
  }
  const int64_t n = m0 + m0_add;
  const T_ACC c =
      n == 0 ? T_ACC(0) : static_cast<T_ACC>(m0_add) / static_cast<T_ACC>(n);
  const T_ACC delta = static_cast<T_ACC>(m1_add[i]) - static_cast<T_ACC>(m1[i]);
  m1[i] = static_cast<T_ACC>(m1[i]) + c * delta;
  m2[i] = static_cast<T_ACC>(m2[i]) * static_cast<T_ACC>(m0) +
          static_cast<T_ACC>(m2_add[i]) +
          delta * delta * c * static_cast<T_ACC>(m0);
  const T_ACC rsqrt = c10::cuda::compat::rsqrt(static_cast<T_ACC>(m2[i]) /
                                                   static_cast<T_ACC>(n) +
                                               static_cast<T_ACC>(eps));
  scale[i] = rsqrt;
  bias[i] = -rsqrt * static_cast<T_ACC>(m1[i]);
}

template <typename T>
void PrenormCUDAKernelImpl(const torch::Tensor& X, int64_t running_m0,
                           const torch::Tensor& running_m1,
                           const torch::Tensor& running_m2, double eps,
                           torch::Tensor& saved_m1, torch::Tensor& saved_m2,
                           torch::Tensor& scale, torch::Tensor& bias) {
  std::tie(saved_m2, saved_m1) =
      torch::var_mean(X, /*dim=*/{0}, /*unbiased=*/false, /*keepdim=*/false);

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

  const int64_t outer_size = X.size(0);
  const int64_t inner_size = X.size(1);
  const int64_t B = at::divup(inner_size, kCUDANumThreads);
  const T* X_data = X.data_ptr<T>();
  const T* running_m1_data = running_m1.data_ptr<T>();
  const T* running_m2_data = running_m2.data_ptr<T>();
  T* saved_m1_data = saved_m1.data_ptr<T>();
  T* saved_m2_data = saved_m2.data_ptr<T>();
  T* scale_data = scale.data_ptr<T>();
  T* bias_data = bias.data_ptr<T>();
  MergeMomentsCUDAKernel<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
      inner_size, eps, running_m0, running_m1_data, running_m2_data, outer_size,
      saved_m1_data, saved_m2_data, scale_data, bias_data);
  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void PrenormCUDAKernel(const torch::Tensor& X, int64_t running_m0,
                       const torch::Tensor& running_m1,
                       const torch::Tensor& running_m2, double eps,
                       torch::Tensor& saved_m1, torch::Tensor& saved_m2,
                       torch::Tensor& scale, torch::Tensor& bias) {
  AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "PrenormCUDAKernel", [&]() {
    PrenormCUDAKernelImpl<scalar_t>(X, running_m0, running_m1, running_m2, eps,
                                    saved_m1, saved_m2, scale, bias);
  });
}

}  // namespace ml4co
