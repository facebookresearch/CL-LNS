# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import ecole.typing


class DualBound(ecole.typing.InformationFunction):
    def __init__(self):
        super().__init__()

    def before_reset(self, model):
        super().before_reset(model)

    def extract(self, model, done):
        m = model.as_pyscipopt()
        dual_bound = m.getDualbound()
        return dual_bound

class Gap(ecole.typing.InformationFunction):
    def __init__(self):
        super().__init__()

    def before_reset(self, model):
        super().before_reset(model)

    def extract(self, model, done):
        m = model.as_pyscipopt()
        gap = m.getGap()
        return gap
