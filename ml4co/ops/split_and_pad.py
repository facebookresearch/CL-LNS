# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import torch

import ml4co

torch.ops.load_library(
    os.path.join(os.path.dirname(os.path.dirname(ml4co.__file__)),
                 "libml4co_ops.so"))


class SplitAndPadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sizes, feature_size=0, padding_value=0.0):
        ctx.save_for_backward(sizes)
        return torch.ops.ml4co_ops.split_and_pad(input, sizes, feature_size,
                                                 padding_value)

    @staticmethod
    def backward(ctx, grad_output):
        sizes, = ctx.saved_tensors
        grad_input = torch.ops.ml4co_ops.split_and_pad_backward(
            grad_output, sizes)
        return grad_input, None, None, None
