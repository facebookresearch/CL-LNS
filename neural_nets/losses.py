# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch


class LogScoreLoss(torch.nn.Module):
    """
    Loss function to weight sample loss by confidence in the target value
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("eps", torch.tensor([1e-6]))

    def weight(self, input, target):
        max_tgt = torch.max(target, dim=-1, keepdim=True).values
        return torch.maximum(input, target) / max_tgt

    def forward(self, input, target):        
        # Avoid division by zero
        target = torch.maximum(target, self.eps)
        main_loss = torch.log(input / target).abs()
        # Handle predictions smaller than eps
        neg_domain = (input / target - self.eps).abs() + torch.log(self.eps).abs()
        loss = torch.where(input / target < self.eps, neg_domain, main_loss)
        assert not torch.isnan(loss).any()

        weighted = loss * self.weight(input, target)
        assert not torch.isnan(weighted).any()

        return weighted.mean()



class LinearScoreLoss(torch.nn.Module):
    """
    Loss function to weight sample loss by confidence in the target value
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("eps", torch.tensor([1e-6]))

    def weight(self, input, target):
        max_tgt = torch.max(target, dim=-1, keepdim=True).values
        return torch.maximum(input, target) / max_tgt

    def forward(self, input, target):
        # Avoid division by zero
        target = torch.maximum(target, self.eps)
        loss = (input - target).abs() / target

        weighted = loss * self.weight(input, target)
        return weighted.mean()
