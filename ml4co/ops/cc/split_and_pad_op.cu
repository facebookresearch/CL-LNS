// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#include "ml4co/ops/cc/split_and_pad_op.h"

#include <ATen/Parallel.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAStream.h>

#include "ml4co/ops/cc/cuda_utils.h"

namespace ml4co {

namespace {

template <typename T>
__global__ void SplitAndPadLargeFeatureCUDAKernel(const T* X,
                                                  const int64_t* sizes,
                                                  const int64_t* strides,
                                                  int64_t inner_size, T value,
                                                  T* Y) {
  const int64_t row = blockIdx.x;
  const int64_t cur_size = sizes[row];
  const T* X_ptr = row == 0 ? X : X + strides[row - 1];
  T* Y_ptr = Y + row * inner_size;
  for (int64_t col = threadIdx.x; col < inner_size; col += blockDim.x) {
    Y_ptr[col] = col < cur_size ? X_ptr[col] : value;
  }
}

template <typename T>
__global__ void SplitAndPadSmallFeatureCUDAKernel(
    const T* X, const int64_t* sizes, const int64_t* strides,
    int64_t outer_size, int64_t inner_size, T value, T* Y) {
  const int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row < outer_size) {
    const int64_t cur_size = sizes[row];
    const T* X_ptr = row == 0 ? X : X + strides[row - 1];
    T* Y_ptr = Y + row * inner_size;
    for (int64_t col = threadIdx.x; col < inner_size; col += blockDim.x) {
      Y_ptr[col] = col < cur_size ? X_ptr[col] : value;
    }
  }
}

template <typename T>
__global__ void SplitAndPadBackwardLargeFeatureCUDAKernel(
    const T* dY, const int64_t* sizes, const int64_t* strides,
    int64_t inner_size, T* dX) {
  const int64_t row = blockIdx.x;
  const int64_t cur_size = sizes[row];
  const T* dY_ptr = dY + row * inner_size;
  T* dX_ptr = row == 0 ? dX : dX + strides[row - 1];
  for (int64_t col = threadIdx.x; col < cur_size; col += blockDim.x) {
    dX_ptr[col] = dY_ptr[col];
  }
}

template <typename T>
__global__ void SplitAndPadBackwardSmallFeatureCUDAKernel(
    const T* dY, const int64_t* sizes, const int64_t* strides,
    int64_t outer_size, int64_t inner_size, T* dX) {
  const int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row < outer_size) {
    const int64_t cur_size = sizes[row];
    const T* dY_ptr = dY + row * inner_size;
    T* dX_ptr = row == 0 ? dX : dX + strides[row - 1];
    for (int64_t col = threadIdx.x; col < cur_size; col += blockDim.x) {
      dX_ptr[col] = dY_ptr[col];
    }
  }
}

template <typename T>
void SplitAndPadCUDAKernelImpl(const torch::Tensor& X,
                               const torch::Tensor& sizes,
                               const torch::Tensor& strides, double value,
                               torch::Tensor& Y) {
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t outer_size = Y.size(0);
  const int64_t inner_size = Y.size(1);
  const T* X_data = X.data_ptr<T>();
  const int64_t* sizes_data = sizes.data_ptr<int64_t>();
  const int64_t* strides_data = strides.data_ptr<int64_t>();
  T* Y_data = Y.data_ptr<T>();

  if (inner_size >= kCUDANumThreads) {
    SplitAndPadLargeFeatureCUDAKernel<T>
        <<<outer_size, kCUDANumThreads, 0, cuda_stream>>>(
            X_data, sizes_data, strides_data, inner_size, value, Y_data);
  } else {
    constexpr int64_t kThreadX = kCUDAWarpSize;
    constexpr int64_t kThreadY = kCUDAWarpSize / 2;
    const int64_t B = at::divup(outer_size, kThreadY);
    SplitAndPadSmallFeatureCUDAKernel<T>
        <<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
            X_data, sizes_data, strides_data, outer_size, inner_size, value,
            Y_data);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void SplitAndPadBackwardCUDAKernelImpl(const torch::Tensor& dY,
                                       const torch::Tensor& sizes,
                                       const torch::Tensor& strides,
                                       torch::Tensor& dX) {
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t outer_size = dY.size(0);
  const int64_t inner_size = dY.size(1);
  const T* dY_data = dY.data_ptr<T>();
  const int64_t* sizes_data = sizes.data_ptr<int64_t>();
  const int64_t* strides_data = strides.data_ptr<int64_t>();
  T* dX_data = dX.data_ptr<T>();

  if (inner_size >= kCUDANumThreads) {
    SplitAndPadBackwardLargeFeatureCUDAKernel<T>
        <<<outer_size, kCUDANumThreads, 0, cuda_stream>>>(
            dY_data, sizes_data, strides_data, inner_size, dX_data);
  } else {
    constexpr int64_t kThreadX = kCUDAWarpSize;
    constexpr int64_t kThreadY = kCUDAWarpSize / 2;
    const int64_t B = at::divup(outer_size, kThreadY);
    SplitAndPadBackwardSmallFeatureCUDAKernel<T>
        <<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
            dY_data, sizes_data, strides_data, outer_size, inner_size, dX_data);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void SplitAndPadCUDAKernel(const torch::Tensor& X, const torch::Tensor& sizes,
                           const torch::Tensor& strides, double value,
                           torch::Tensor& Y) {
  AT_DISPATCH_ALL_TYPES(X.scalar_type(), "SplitAndPadCUDAKernel", [&]() {
    SplitAndPadCUDAKernelImpl<scalar_t>(X, sizes, strides,
                                        static_cast<scalar_t>(value), Y);
  });
}

void SplitAndPadBackwardCUDAKernel(const torch::Tensor& dY,
                                   const torch::Tensor& sizes,
                                   const torch::Tensor& strides,
                                   torch::Tensor& dX) {
  AT_DISPATCH_ALL_TYPES(
      dY.scalar_type(), "SplitAndPadBackwardCUDAKernel", [&]() {
        SplitAndPadBackwardCUDAKernelImpl<scalar_t>(dY, sizes, strides, dX);
      });
}

}  // namespace ml4co
