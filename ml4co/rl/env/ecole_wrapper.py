# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import ecole
import ilp_model

import numpy as np
import torch

from typing import Any, Callable, Optional, Tuple

import rloptim.utils.data_utils as data_utils

from rloptim.core.types import Tensor, NestedTensor
from rloptim.envs.env import Env
from rloptim.envs.wrappers import TimeLimitWrapper

import competition.common.environments as competition_env
import competition.common.rewards as competition_rewards

from instance_loader import InstanceLoader


class EcoleWrapper(Env):
    def __init__(self,
                 dataset: str,
                 problem_set: str,
                 observation_function: ecole.observation.NodeBipartite,
                 timeout: int = 900):
        super(EcoleWrapper, self).__init__()

        self._env = None
        self._dataset = dataset
        self._problem_set = problem_set
        self._observation_function = observation_function
        self._timeout = timeout

        self._instance_loader = InstanceLoader(dataset_loc=self._dataset,
                                               load_metadata=True)

    @property
    def env(self) -> Optional[ecole.environment.Environment]:
        return self._env

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def problem_set(self) -> str:
        return self._problem_set

    @property
    def observation_function(self) -> ecole.observation.NodeBipartite:
        return self._observation_function

    def reset(self, **kwargs) -> NestedTensor:
        instance_data = self._instance_loader(self._problem_set)
        if isinstance(instance_data, ecole.core.scip.Model):
            instance = instance_data
            model = ilp_model.Model(instance)
            model.find_initial_solution()
            bounds = model.get_primal_dual_bounds()
            initial_primal_bound = bounds[0]
            initial_dual_bound = bounds[1]
        else:
            instance, metadata = instance_data
            initial_primal_bound = metadata["primal_bound"]
            initial_dual_bound = metadata["dual_bound"]

        reward_function = competition_rewards.TimeLimitDualIntegral()
        reward_function.set_parameters(
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound,
            objective_offset=0)

        self._env = competition_env.Branching(
            time_limit=self._timeout,
            observation_function=(self._observation_function,
                                  ecole.observation.Khalil2016(
                                      pseudo_candidates=True)),
            reward_function=-reward_function,
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime()
            })

        obs, action_set, reward, done, _ = self._env.reset(
            instance, objective_limit=initial_primal_bound)

        obs = self._parse_obs(obs, action_set)
        action_set = torch.from_numpy(action_set.astype(np.int64))
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        return {
            "obs": obs,
            "action_set": action_set,
            "reward": reward,
            "done": done
        }

    def step(self, action: NestedTensor) -> NestedTensor:
        if isinstance(action, dict):
            action = action["action"]
        action = data_utils.to_numpy(action)
        obs, action_set, reward, done, _ = self._env.step(action)

        obs = self._parse_obs(obs, action_set)
        action_set = torch.from_numpy(action_set.astype(np.int64))
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done])

        return {
            "obs": obs,
            "action_set": action_set,
            "reward": reward,
            "done": done
        }

    def _parse_obs(self, obs: Any, action_set: Any) -> Tuple[torch.Tensor]:
        bgo, khalil = obs
        bgo.add_khalil_features(khalil, action_set)
        bgo.check_features()
        obs = (bgo.row_features, bgo.edge_features.indices,
               bgo.edge_features.values, bgo.column_features)
        return obs

    def close(self) -> None:
        if self._env is not None:
            self._env.close()

    def seed(self, seed: Optional[int] = None) -> None:
        if self._env is not None:
            self._env.seed(seed)
