// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#pragma once

#include <ATen/core/TensorBody.h>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>

#include <tuple>

namespace ml4co {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Prenorm(
    const torch::Tensor& input, int64_t running_m0,
    const torch::Tensor& running_m1, torch::Tensor& running_m2, double eps);

void PrenormCPUKernel(const torch::Tensor& X, int64_t running_m0,
                      const torch::Tensor& running_m1,
                      const torch::Tensor& running_m2, double eps,
                      torch::Tensor& saved_m1, torch::Tensor& saved_m2,
                      torch::Tensor& scale, torch::Tensor& bias);

#ifndef ML4CO_CPU_ONLY

void PrenormCUDAKernel(const torch::Tensor& X, int64_t running_m0,
                       const torch::Tensor& running_m1,
                       const torch::Tensor& running_m2, double eps,
                       torch::Tensor& saved_m1, torch::Tensor& saved_m2,
                       torch::Tensor& scale, torch::Tensor& bias);

#endif  // ML4CO_CPU_ONLY

}  // namespace ml4co
