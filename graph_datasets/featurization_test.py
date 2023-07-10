# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import unittest
import ecole
import torch
import torch_geometric
import numpy as np
import pyscipopt
import graph_datasets.bipartite_graph as bg
import graph_datasets.bipartite_graph_dataset as bgd
import graph_datasets.bipartite_graph_observations as bgo
import ilp_solver
import os
import time

def advance_to_root_node(model, branching):
    """Utility to advance a model to the root node."""
    if branching:
        dyn = ecole.dynamics.BranchingDynamics()
        #print("BranchingDynamics")
    else:
        dyn = ecole.dynamics.PrimalSearchDynamics()
        #print("PrimalSearchDynamics")

    model = dyn.reset_dynamics(model)
    return model

def make_obs(obs_func, model, branching=True):
    """Utility function to extract observation on root node."""
    start = time.monotonic()
    if isinstance(obs_func, tuple):
        for f in obs_func:
            f.before_reset(model)
    else:
        obs_func.before_reset(model)
    stop = time.monotonic()

    advance_to_root_node(model, branching)

    stop = time.monotonic()
    if isinstance(obs_func, tuple):
        rslt = []
        for f in obs_func:
            rslt.append(f.extract(model, False))
        return rslt
    else:
        return obs_func.extract(model, False)


def disable_all(solver):
    solver.disable_presolver()
    solver.disable_cuts()
    solver.disable_heuristics()


class FeaturizationTest(unittest.TestCase):
    def setUp(self):
        # Create a small ILP problem
        solver1 = ilp_solver.ILPSolver(engine="scip")
        x1 = solver1.create_integer_var("x1")
        x2 = solver1.create_integer_var("x2")
        solver1.add_constraint(10 * x1 + 15 * x2 >= 100.23)
        solver1.add_constraint(20 * x1 + 16 * x2 >= 161.8)
        solver1.add_constraint(17 * x1 + 11 * x2 >= 129.42)

        # Minimize the objective
        solver1.set_objective_function(80 * x1 + 95 * x2, maximize=False)
        disable_all(solver1)
        scip_model = solver1.as_scip_model()
        self.model1 = ecole.scip.Model.from_pyscipopt(scip_model)
        #self.model1.transform_prob()
        
        solver2 = ilp_solver.ILPSolver(engine="scip")
        x1 = solver2.create_integer_var("x1")
        x2 = solver2.create_integer_var("x2")
        solver2.add_constraint(20 * x1 + 30 * x2 <= 200)
        solver2.add_constraint(40 * x1 + 32 * x2 <= 320)
        solver2.add_constraint(34 * x1 + 22 * x2 <= 260)

        # Minimize the objective
        solver2.set_objective_function(80 * x1 + 95 * x2, maximize=True)
        disable_all(solver2)
        scip_model = solver2.as_scip_model()
        self.model2 = ecole.scip.Model.from_pyscipopt(scip_model)
        #self.model2.transform_prob()

        solver3 = ilp_solver.ILPSolver(engine="scip")
        x0 = solver3.create_integer_var("x0")
        x1 = solver3.create_integer_var("x1")
        x2 = solver3.create_integer_var("x2")
        solver3.add_constraint(20 * x1 + 30 * x2 >= 200)
        solver3.add_constraint(40 * x1 + 32 * x2 >= 320)
        solver3.add_constraint(34 * x1 + 22 * x2 >= 260)
        solver3.add_constraint(2 * x0 + 3 * x1 == 12)

        # Minimize the objective
        solver3.set_objective_function(87.3 * x1 + 93.2 * x2, maximize=False)
        disable_all(solver3)
        scip_model = solver3.as_scip_model()
        self.model3 = ecole.scip.Model.from_pyscipopt(scip_model)
        #self.model3.transform_prob()


    def testBranchingFeaturization(self):
        observation = make_obs(bgo.BipartiteGraphObservations(), self.model1)
        #print("VARS1: " + str(observation.column_features), flush=True)
        self.assertEqual(str(observation.column_features),
        """tensor([[-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  8.4211e-01,
          5.8809e+00,  6.4414e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  5.8809e+00,
          8.8086e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00],
        [-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  1.0000e+00,
          2.7614e+00,  7.6491e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  2.7614e+00,
          7.6143e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00]])""")
        #print("CNS1: " + str(observation.row_features))
        self.assertEqual(str(observation.row_features),
        """tensor([[-1.0023e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -5.5598e+00, -9.9375e-01,  1.0000e+00, -1.9779e-03,  0.0000e+00],
        [-1.6180e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -6.3172e+00, -9.8082e-01,  1.0000e+00, -5.6137e-04,  0.0000e+00],
        [-1.2942e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -6.3916e+00, -9.5634e-01,  0.0000e+00,  0.0000e+00,  1.6667e-01]])""")
        #print("EDGES1: " + str(observation.edge_features.indices))
        self.assertEqual(str(observation.edge_features.indices),
        """tensor([[0, 0, 1, 1, 2, 2],
        [0, 1, 0, 1, 0, 1]])""")
        #print("EDGE VALS1: " + str(observation.edge_features.values))
        self.assertEqual(str(observation.edge_features.values),
        """tensor([[-10.0000,  -0.0998],
        [-15.0000,  -0.1497],
        [-20.0000,  -0.1236],
        [-16.0000,  -0.0989],
        [-17.0000,  -0.1314],
        [-11.0000,  -0.0850]])""")

        observation = make_obs(bgo.BipartiteGraphObservations(), self.model2)
        #print("VARS2: " + str(observation.column_features), flush=True)
        self.assertEqual(str(observation.column_features),
        """tensor([[-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18, -8.4211e-01,
          5.7143e+00, -6.4414e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  5.7143e+00,
          7.1429e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00],
        [-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18, -1.0000e+00,
          2.8571e+00, -7.6491e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  2.8571e+00,
          8.5714e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00]])""")
        #print("CNS2: " + str(observation.row_features))
        self.assertEqual(str(observation.row_features),
        """tensor([[ 2.0000e+02,  1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
          5.5470e+00, -9.9375e-01,  1.0000e+00, -4.9448e-04,  0.0000e+00],
        [ 3.2000e+02,  1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
          6.2470e+00, -9.8082e-01,  1.0000e+00, -1.4034e-04,  0.0000e+00],
        [ 2.6000e+02,  1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
          6.4202e+00, -9.5634e-01,  0.0000e+00, -0.0000e+00,  1.6667e-01]])""")
        #print("EDGES2: " + str(observation.edge_features.indices))
        self.assertEqual(str(observation.edge_features.indices),
        """tensor([[0, 0, 1, 1, 2, 2],
        [0, 1, 0, 1, 0, 1]])""")
        #print("EDGE VALS2: " + str(observation.edge_features.values), flush=True)
        self.assertEqual(str(observation.edge_features.values),
        """tensor([[20.0000,  0.1000],
        [30.0000,  0.1500],
        [40.0000,  0.1250],
        [32.0000,  0.1000],
        [34.0000,  0.1308],
        [22.0000,  0.0846]])""")

        observation = make_obs(bgo.BipartiteGraphObservations(), self.model3)
        #print("VARS3: " + str(observation.column_features), flush=True)
        self.assertEqual(str(observation.column_features),
        """tensor([[-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  0.0000e+00,
         -2.7931e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00, -2.7931e+00,
          2.0690e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00],
        [-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  9.3670e-01,
          5.8621e+00,  6.8363e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  5.8621e+00,
          8.6207e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00],
        [-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  1.0000e+00,
          2.7586e+00,  7.2983e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  2.7586e+00,
          7.5862e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00]])""")
        #print("CNS3: " + str(observation.row_features))
        self.assertEqual(str(observation.row_features),
        """tensor([[-2.0000e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -5.5470e+00, -9.8646e-01,  1.0000e+00, -4.6740e-04,  0.0000e+00],
        [-3.2000e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -6.2470e+00, -9.8975e-01,  0.0000e+00,  0.0000e+00,  1.6667e-01],
        [-2.6000e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -6.4202e+00, -9.7044e-01,  1.0000e+00, -2.5171e-04,  0.0000e+00],
        [-1.2000e+01, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -3.3282e+00, -5.6881e-01,  1.0000e+00,  0.0000e+00,  1.6667e-01],
        [ 1.2000e+01,  1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
          3.3282e+00,  5.6881e-01,  1.0000e+00, -0.0000e+00,  1.6667e-01]])""")
        #print("EDGES3: " + str(observation.edge_features.indices))
        self.assertEqual(str(observation.edge_features.indices),
        """tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 2, 1, 2, 1, 2, 0, 1, 0, 1]])""")
        #print("EDGE VALS3: " + str(observation.edge_features.values))
        self.assertEqual(str(observation.edge_features.values),
        """tensor([[-20.0000,  -0.1000],
        [-30.0000,  -0.1500],
        [-40.0000,  -0.1250],
        [-32.0000,  -0.1000],
        [-34.0000,  -0.1308],
        [-22.0000,  -0.0846],
        [ -2.0000,  -0.1667],
        [ -3.0000,  -0.2500],
        [  2.0000,   0.1667],
        [  3.0000,   0.2500]])""")

    def testPrimalSearchFeatures(self):
        observation = make_obs(bgo.BipartiteGraphObservations(), self.model3, branching=False)
        #print("VARS: " + str(observation.column_features), flush=True)
        self.assertEqual(str(observation.column_features),
        """tensor([[-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  0.0000e+00,
         -2.7931e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00, -2.7931e+00,
          2.0690e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00],
        [-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  9.3670e-01,
          5.8621e+00,  6.8363e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  5.8621e+00,
          8.6207e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00],
        [-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  1.0000e+00,
          2.7586e+00,  7.2983e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  2.7586e+00,
          7.5862e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00]])""")
        #print("CNS: " + str(observation.row_features))
        self.assertEqual(str(observation.row_features),
        """tensor([[-2.0000e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -5.5470e+00, -9.8646e-01,  1.0000e+00, -4.6740e-04,  0.0000e+00],
        [-3.2000e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -6.2470e+00, -9.8975e-01,  0.0000e+00,  0.0000e+00,  1.6667e-01],
        [-2.6000e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -6.4202e+00, -9.7044e-01,  1.0000e+00, -2.5171e-04,  0.0000e+00],
        [-1.2000e+01, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -3.3282e+00, -5.6881e-01,  1.0000e+00,  0.0000e+00,  1.6667e-01],
        [ 1.2000e+01,  1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
          3.3282e+00,  5.6881e-01,  1.0000e+00, -0.0000e+00,  1.6667e-01]])""")
        #print("EDGES: " + str(observation.edge_features.indices))
        self.assertEqual(str(observation.edge_features.indices),
        """tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 2, 1, 2, 1, 2, 0, 1, 0, 1]])""")
        #print("EDGE VALS: " + str(observation.edge_features.values))
        self.assertEqual(str(observation.edge_features.values),
        """tensor([[-20.0000,  -0.1000],
        [-30.0000,  -0.1500],
        [-40.0000,  -0.1250],
        [-32.0000,  -0.1000],
        [-34.0000,  -0.1308],
        [-22.0000,  -0.0846],
        [ -2.0000,  -0.1667],
        [ -3.0000,  -0.2500],
        [  2.0000,   0.1667],
        [  3.0000,   0.2500]])""")

    def testKhalilFeaturization(self):
        observation, khalil = make_obs((bgo.BipartiteGraphObservations(), ecole.observation.Khalil2016()), self.model1)
        branching_vars = np.array([0, 1])
        observation.add_khalil_features(khalil, branching_vars)

        print("VARS: " + str(observation.column_features), flush=True)
        self.assertEqual(str(observation.column_features),
        """tensor([[-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  8.4211e-01,
          5.8809e+00,  6.4414e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  5.8809e+00,
          8.8086e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00,  1.6000e+01,  1.6000e+01,
          0.0000e+00,  3.0000e+00,  2.0000e+00,  0.0000e+00,  2.0000e+00,
          2.0000e+00,  3.0000e+00,  1.5667e+01,  4.1899e+00,  1.0000e+01,
          2.0000e+01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  1.1914e-01,  1.1914e-01,  1.1914e-01,  8.8086e-01,
          1.3526e-01,  1.0000e+00,  1.0495e-01,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  2.0000e+00,  0.0000e+00,  2.0000e+00,
          2.0000e+00,  5.0000e-01,  5.0000e-01,  5.0000e-01,  1.0000e+00,
         -1.0000e+00, -1.1610e-01, -9.0719e-02,  4.0000e-01,  6.0714e-01,
          1.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,
          0.0000e+00,  2.0000e+00,  3.0000e+01,  1.5000e+01,  3.5355e+00,
          1.0000e+01,  2.0000e+01,  6.7778e-02,  9.5556e-01,  4.7778e-01,
          5.4997e-02,  4.0000e-01,  5.5556e-01,  6.7778e-02,  9.5556e-01,
          4.7778e-01,  5.4997e-02,  4.0000e-01,  5.5556e-01,  1.2429e+00,
          1.6000e+01,  8.0000e+00,  6.0609e-01,  7.1429e+00,  8.8571e+00],
        [-9.2234e+18,  9.2234e+18, -9.2234e+18,  9.2234e+18,  1.0000e+00,
          2.7614e+00,  7.6491e-01,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  2.7614e+00,
          7.6143e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  0.0000e+00,  1.9000e+01,  1.9000e+01,
          0.0000e+00,  3.0000e+00,  2.0000e+00,  0.0000e+00,  2.0000e+00,
          2.0000e+00,  3.0000e+00,  1.4000e+01,  2.1602e+00,  1.1000e+01,
          1.6000e+01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  2.3857e-01,  2.3857e-01,  2.3857e-01,  7.6143e-01,
          3.1332e-01,  1.0000e+00,  1.8166e-01,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  2.0000e+00,  0.0000e+00,  2.0000e+00,
          2.0000e+00,  5.0000e-01,  5.0000e-01,  5.0000e-01,  1.0000e+00,
         -1.0000e+00, -1.3017e-01, -7.8336e-02,  3.9286e-01,  6.0000e-01,
          1.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,
          0.0000e+00,  2.0000e+00,  3.1000e+01,  1.5500e+01,  3.5355e-01,
          1.5000e+01,  1.6000e+01,  6.7778e-02,  1.0444e+00,  5.2222e-01,
          5.4997e-02,  4.4444e-01,  6.0000e-01,  6.7778e-02,  1.0444e+00,
          5.2222e-01,  5.4997e-02,  4.4444e-01,  6.0000e-01,  1.2429e+00,
          1.9000e+01,  9.5000e+00,  2.6769e+00,  5.7143e+00,  1.3286e+01]])""")
        print("CNS: " + str(observation.row_features))
        self.assertEqual(str(observation.row_features),
        """tensor([[-1.0023e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -5.5598e+00, -9.9375e-01,  1.0000e+00, -1.9779e-03,  0.0000e+00],
        [-1.6180e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -6.3172e+00, -9.8082e-01,  1.0000e+00, -5.6137e-04,  0.0000e+00],
        [-1.2942e+02, -1.0000e+00,  0.0000e+00,  2.0000e+00,  1.0000e+00,
         -6.3916e+00, -9.5634e-01,  0.0000e+00,  0.0000e+00,  1.6667e-01]])""")
        print("EDGES: " + str(observation.edge_features.indices))
        self.assertEqual(str(observation.edge_features.indices),
        """tensor([[0, 0, 1, 1, 2, 2],
        [0, 1, 0, 1, 0, 1]])""")
        print("EDGE VALS: " + str(observation.edge_features.values))
        self.assertEqual(str(observation.edge_features.values),
        """tensor([[-10.0000,  -0.0998],
        [-15.0000,  -0.1497],
        [-20.0000,  -0.1236],
        [-16.0000,  -0.0989],
        [-17.0000,  -0.1314],
        [-11.0000,  -0.0850]])""")


if __name__ == "__main__":
    unittest.main()
