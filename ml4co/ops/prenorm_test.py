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

from ml4co.ops.prenorm import PrenormFunction


def prenorm_ref(input, running_m0, running_m1, running_m2, eps):
    m0 = input.size(0)
    m2, m1 = torch.var_mean(input, dim=0, unbiased=False)

    n = m0 + running_m0
    c = 0 if n == 0 else running_m0 / n
    delta = running_m1 - m1
    m1 += c * delta
    m2 = m2 * m0 + running_m2 + delta.square() * c * m0

    scale = (m2 / n + eps).rsqrt()
    bias = -scale * m1

    return m1, m2, scale, bias


class PrenormFunctionTest(unittest.TestCase):
    def setUp(self):
        self.prenorm = PrenormFunction.apply

    @hypothesis.given(
        outer_size=st.integers(2, 100),
        inner_size=st.integers(1, 200),
        running_m0=st.integers(0, 10),
        eps=st.floats(min_value=0, max_value=1e-3),
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]))
    @hypothesis.settings(deadline=None)
    def test_prenorm(self, outer_size, inner_size, running_m0, eps, device):
        x = torch.randn(outer_size, inner_size, device=device)
        if running_m0 == 0:
            running_m1 = torch.zeros((inner_size, ), device=device)
            running_m2 = torch.zeros((inner_size, ), device=device)
        else:
            running_m1 = torch.randn((inner_size, ), device=device)
            running_m2 = torch.randn((inner_size, ), device=device)

        m1, m2, scale, bias = self.prenorm(x, running_m0, running_m1,
                                           running_m2, eps)
        m1_ref, m2_ref, scale_ref, bias_ref = prenorm_ref(
            x, running_m0, running_m1, running_m2, eps)

        torch.testing.assert_allclose(m1, m1_ref)
        torch.testing.assert_allclose(m2, m2_ref)
        torch.testing.assert_allclose(scale, scale_ref)
        torch.testing.assert_allclose(bias, bias_ref)


if __name__ == "__main__":
    unittest.main()
