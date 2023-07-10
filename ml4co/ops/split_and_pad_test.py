# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import hypothesis
import hypothesis.strategies as st
import unittest

import torch
import torch.nn.functional as F

from ml4co.ops.split_and_pad import SplitAndPadFunction


def split_and_pad_ref(input, sizes, feature_size=0, padding_value=0):
    feature_size = max(feature_size, sizes.max().item())
    inputs = input.split(sizes.detach().cpu().tolist())
    outputs = [
        F.pad(x, (0, feature_size - x.size(0)), "constant", padding_value)
        for x in inputs
    ]
    return torch.stack(outputs, dim=0)


class SplitAndPadFunctionTest(unittest.TestCase):
    def setUp(self):
        self.split_and_pad = SplitAndPadFunction.apply

    @hypothesis.given(
        batch_size=st.integers(1, 200),
        inner_size=st.integers(10, 500),
        feature_size=st.sampled_from([0, 500]),
        padding_value=st.floats(min_value=-10.0, max_value=10.0),
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]))
    @hypothesis.settings(deadline=None)
    def test_forward(self, batch_size, inner_size, feature_size, padding_value,
                     device):
        sizes = torch.randint(low=1,
                              high=inner_size,
                              size=(batch_size, ),
                              device=device)
        input_size = sizes.sum().item()
        x = torch.randn(input_size, device=device)
        y = self.split_and_pad(x, sizes, feature_size, padding_value)
        y_ref = split_and_pad_ref(x, sizes, feature_size, padding_value)
        torch.testing.assert_allclose(y, y_ref)

    @hypothesis.given(
        batch_size=st.integers(1, 100),
        inner_size=st.integers(10, 500),
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]))
    @hypothesis.settings(deadline=None)
    def test_backward(self, batch_size, inner_size, device):
        sizes = torch.randint(low=1,
                              high=inner_size,
                              size=(batch_size, ),
                              device=device)
        input_size = sizes.sum().item()
        x = torch.randn(input_size, device=device)
        x_ref = x.detach().clone()
        x.requires_grad_(True)
        x_ref.requires_grad_(True)
        y = self.split_and_pad(x, sizes)
        y_ref = split_and_pad_ref(x_ref, sizes)

        dy = torch.randn_like(y)
        y.backward(dy)
        y_ref.backward(dy)
        torch.testing.assert_allclose(x.grad, x_ref.grad)


if __name__ == "__main__":
    unittest.main()
