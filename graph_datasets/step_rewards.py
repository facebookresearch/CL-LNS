# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import ecole.typing
import competition.common.rewards as competition_rewards


# Returns the relative improvement in dual bound since the last step
class Dual(ecole.typing.RewardFunction):
    def __init__(self):
        self.parameters = competition_rewards.IntegralParameters()
        super().__init__(wall=True, bound_function=lambda model: (
            self.parameters.offset,
            self.parameters.initial_primal_bound))

    def set_parameters(self, objective_offset=None, initial_primal_bound=None, initial_dual_bound=None):
        self.parameters = competition_rewards.IntegralParameters(
            offset=objective_offset,
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound)

    def before_reset(self, model):
        self.parameters.fetch_values(model)  
        super().before_reset(model)
        self.last_dual_bound = self.parameters.initial_dual_bound

    def extract(self, model, done):
        if done:
            return 0

        m = model.as_pyscipopt()
        dual_bound = m.getDualbound()
        reward = abs(dual_bound - self.last_dual_bound) / abs(self.last_dual_bound)
        self.last_dual_bound = dual_bound

        return reward


# Returns the relative improvement in the primal/dual gap since the last step                                                                                                                                                                               
class PrimalDualGrap(ecole.typing.RewardFunction):
    def __init__(self):
        self.parameters = competition_rewards.IntegralParameters()
        super().__init__(wall=True, bound_function=lambda model: (
            self.parameters.offset,
            self.parameters.initial_primal_bound))

    def set_parameters(self, objective_offset=None, initial_primal_bound=None, initial_dual_bound=None):
        self.parameters = competition_rewards.IntegralParameters(
            offset=objective_offset,
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound)

    def before_reset(self, model):
        self.parameters.fetch_values(model)
        super().before_reset(model)
        self.last_gap = abs(self.parameters.initial_dual_bound - self.parameters.initial_primal_bound) / min(abs(self.parameters.initial_dual_bound), abs(self.parameters.initial_primal_bound))

    def extract(self, model, done):
        if done:
            return 0

        m = model.as_pyscipopt()
        gap = m.getGap()
        reward = (self.last_gap - gap) / self.last_gap
        self.last_gap = gap

        return reward
