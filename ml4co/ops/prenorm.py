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


class PrenormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_m0, running_m1, running_m2, eps=0.0):
        return torch.ops.ml4co_ops.prenorm(input, running_m0, running_m1,
                                           running_m2, eps)

    @staticmethod
    def backward(ctx, grad_m1, grad_m2, grad_scale, grad_bias):
        raise NotImplementedError
