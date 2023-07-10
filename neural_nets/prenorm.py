# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch


class PrenormOld(torch.nn.Module):
    def __init__(self, num_features, shift=True, scale=True, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.register_buffer("eps", torch.tensor([eps], requires_grad=False))
        self.reset_normalization()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError("expected a 2D input (got {}D input)".format(input.dim()))

    def freeze_normalization(self):
        self.frozen = torch.tensor([True], dtype=torch.bool)

    def reset_normalization(self):
        self.register_buffer("running_mean", torch.zeros([num_features]))
        self.register_buffer("running_var", torch.ones([num_features]))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long, requires_grad=False))
        self.register_buffer("frozen", torch.tensor([False], dtype=torch.bool, requires_grad=False))
        
    # The input format should be [batch, features]
    def forward(self, input):
        self._check_input_dim(input)
        
        running_mean = self.running_mean
        if self.training and not self.frozen:
            batch_mean = torch.mean(input, dim=[0])
            batch_var = torch.mean((input - batch_mean).pow(2), dim=[0])

            if self.num_batches_tracked < 10000:
                momentum = (
                    float(self.num_batches_tracked) / (self.num_batches_tracked + 1)
                )
            else:
                momentum = 0.9999

            self.num_batches_tracked += 1
            running_mean = (
                momentum * self.running_mean + (1.0 - momentum) * batch_mean
            ).detach()
            self.running_mean = running_mean.detach() 
            self.running_var = (
                momentum * self.running_var + (1.0 - momentum) * batch_var
            ).detach()

        return (input - self.running_mean) / torch.sqrt(
            torch.max(self.eps, self.running_var)
        )
 

class Prenorm(torch.nn.Module):
    def __init__(self, num_features, shift=True, scale=True, preserve_features=[]):
        super().__init__()

        self.num_features = num_features
        self.preserve_features = preserve_features

        self.register_buffer("avg", torch.zeros([num_features], dtype=torch.double))
        self.register_buffer("var", torch.zeros([num_features], dtype=torch.double))
        self.register_buffer("count", torch.zeros([1]))
        self.register_buffer("frozen", torch.tensor([False], dtype=torch.bool, requires_grad=False))

        if shift:
            self.register_buffer("shift", torch.zeros([num_features]))
        else:
            self.shift = None
        if scale:
            self.register_buffer("scale", torch.ones([num_features]))
        else:
            self.scale = None

    def freeze_normalization(self):
        self.frozen = torch.tensor([True], dtype=torch.bool).detach()

    def reset_normalization(self):
        self.avg.zero_()
        self.var.zero_()
        self.count.zero_()
        self.count += 1
        self.frozen.zero_()
        
    def forward(self, input):
        if self.training and not self.frozen:
            # Online mean and variance estimation from Chan et al
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            assert len(input.shape) == 2
            assert self.num_features == input.shape[-1], f"Expected input dimension of size {self.num_features}, got {input.shape[-1]}."

            with torch.no_grad():
                assert not torch.isnan(input).any()
                assert not torch.isnan(self.var).any()
                assert not torch.isnan(self.scale).any()
                assert not torch.isnan(self.count).any()

                sample_count = float(input.shape[0])
                sample_var, sample_avg = torch.var_mean(input.to(torch.float64), dim=0)

                assert not torch.isnan(sample_avg).any()
                assert not torch.isnan(sample_var).any()

                delta = sample_avg - self.avg

                assert self.count + sample_count > 0
                m2 = (self.var * self.count + sample_var * sample_count + torch.square(delta) * self.count * sample_count / (
                    self.count + sample_count))
                assert not torch.isnan(m2).any()

                self.avg = (self.avg * self.count + sample_avg * sample_count) / (self.count + sample_count)
                assert not torch.isnan(self.avg).any()

                self.count += sample_count
                self.var = m2 / self.count

                if self.shift is not None:
                    self.shift = -self.avg.to(torch.float32)
                    assert not torch.isnan(self.shift).any()

                if self.scale is not None:
                    var = torch.where(torch.eq(self.var, 0), self.var.new_ones([self.num_features]), self.var)
                    assert not torch.isnan(var).any()
                    #assert not torch.isinf(var).any()
                    assert (var > 0).all()
                    self.scale = torch.rsqrt(var).to(torch.float32)
                    assert not torch.isnan(self.scale).any()

            for f in self.preserve_features:
                self.shift[f] = 0.0
                self.scale[f] = 1.0

        output = input
        if self.shift is not None:
            output = output + self.shift
        if self.scale is not None:
            output = output * self.scale

        assert not torch.any(torch.isnan(output))

        return output



