// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#pragma once

#include <ATen/cuda/DeviceUtils.cuh>

namespace ml4co {

constexpr int kCUDANumThreads = 256;
constexpr int kCUDAWarpSize = C10_WARP_SIZE;

}  // namespace ml4co
