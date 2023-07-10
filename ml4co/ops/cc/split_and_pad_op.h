// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#pragma once

#include <torch/torch.h>

namespace ml4co {

torch::Tensor SplitAndPad(const torch::Tensor& input,
                          const torch::Tensor& sizes, int64_t feature_size,
                          double value);

torch::Tensor SplitAndPadBackward(const torch::Tensor& grad_output,
                                  const torch::Tensor& sizes);

void SplitAndPadCPUKernel(const torch::Tensor& X, const torch::Tensor& sizes,
                          const torch::Tensor& strides, double value,
                          torch::Tensor& Y);

void SplitAndPadBackwardCPUKernel(const torch::Tensor& dY,
                                  const torch::Tensor& sizes,
                                  const torch::Tensor& strides,
                                  torch::Tensor& dX);

#ifndef ML4CO_CPU_ONLY

void SplitAndPadCUDAKernel(const torch::Tensor& X, const torch::Tensor& sizes,
                           const torch::Tensor& strides, double value,
                           torch::Tensor& Y);

void SplitAndPadBackwardCUDAKernel(const torch::Tensor& dY,
                                   const torch::Tensor& sizes,
                                   const torch::Tensor& strides,
                                   torch::Tensor& dX);

#endif  // ML4CO_CPU_ONLY

}  // namespace ml4co
