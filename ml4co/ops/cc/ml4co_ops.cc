// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#include <torch/torch.h>

#include "ml4co/ops/cc/prenorm_op.h"
#include "ml4co/ops/cc/split_and_pad_op.h"

namespace ml4co {

TORCH_LIBRARY(ml4co_ops, m) {
  m.def("split_and_pad", SplitAndPad);
  m.def("split_and_pad_backward", SplitAndPadBackward);

  m.def("prenorm", Prenorm);
}

}  // namespace ml4co
