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
import string
import random
import os
import sys
import graph_datasets.bipartite_graph as bg
import graph_datasets.bipartite_graph_dataset as bgd
import ilp_solver


def advance_to_root_node(model):
    """Utility to advance a model to the root node."""
    dyn = ecole.dynamics.BranchingDynamics()
    model = dyn.reset_dynamics(model)
    return model

def make_obs(obs_func, model):
    """Utility function to extract observation on root node."""
    obs_func.before_reset(model)
    advance_to_root_node(model)
    return obs_func.extract(model, False)


class BipartiteGraphDatasetTest(unittest.TestCase):
    def setUp(self):
        # Create a small ILP problem
        solver = ilp_solver.ILPSolver(engine="scip")
        x1 = solver.create_integer_var("x1")
        x2 = solver.create_integer_var("x2")
        solver.add_constraint(10 * x1 + 15 * x2 >= 100)
        solver.add_constraint(20 * x1 + 16 * x2 >= 160)
        solver.add_constraint(17 * x1 + 11 * x2 >= 130)

        # Minimize the objective
        solver.set_objective_function(80 * x1 + 95 * x2, maximize=False)
        scip_model = solver.as_scip_model()
        self.model = ecole.scip.Model.from_pyscipopt(scip_model)
        self.model.disable_presolve()
        self.model.disable_cuts()

        letters = string.ascii_letters
        self.db = []
        for i in range(6):
            self.db.append('/tmp/' + ''.join(random.choice(letters) for i in range(10)))

    def tearDown(self):
        for db in self.db:
            try:
                os.remove(db)
            except:
                pass
       

    def testBipartiteGraphQueries(self):
        db = bgd.BipartiteGraphDataset(self.db[0], query_opt=False)
        g0 = bg.BipartiteGraph(np.array([0]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0])
        db.add(g0)
        g1 = bg.BipartiteGraph(np.array([1]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0])
        db.add(g1)
        g2 = bg.BipartiteGraph(np.array([2]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0])
        db.add(g2)
        t0 = db.get(0)
        t1 = db.get(1)
        t2 = db.get(2)
        self.assertEqual(t0.constraint_features, g0.constraint_features)
        self.assertEqual(t1.constraint_features, g1.constraint_features)
        self.assertEqual(t2.constraint_features, g2.constraint_features)

    def testBipartiteGraphIterationNoOpt(self):
        db = bgd.BipartiteGraphDataset(self.db[1], query_opt=False)
        db.add(bg.BipartiteGraph(np.array([0]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        db.add(bg.BipartiteGraph(np.array([1]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        self.assertEqual(2, db.len())
        for i in range(5):
            _ = db.get(i % 2)

    def testBipartiteGraphIterationOpt(self):
        db = bgd.BipartiteGraphDataset(self.db[2], query_opt=True)
        db.add(bg.BipartiteGraph(np.array([0]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        db.add(bg.BipartiteGraph(np.array([1]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        self.assertEqual(2, db.len())
        for i in range(5):
            _ = db.get(i % 2)

    def _testDuplicateEntries(self):
        db = bgd.BipartiteGraphDataset(self.db[3], query_opt=True)
        rslt1 = db.add(bg.BipartiteGraph(np.array([0]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        rslt2 = db.add(bg.BipartiteGraph(np.array([0]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        self.assertEqual(True, rslt1)
        self.assertEqual(False, rslt2)
        self.assertEqual(1, db.len())

    def _testMerge(self):
        db1 = bgd.BipartiteGraphDataset(self.db[4], query_opt=True)
        rslt1 = db1.add(bg.BipartiteGraph(np.array([0]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        self.assertEqual(True, rslt1)
        db2 = bgd.BipartiteGraphDataset(self.db[5], query_opt=True)
        rslt2 = db2.add(bg.BipartiteGraph(np.array([0]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        self.assertEqual(True, rslt2)
        rslt2 = db2.add(bg.BipartiteGraph(np.array([1]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        self.assertEqual(True, rslt2)

        db1.merge(db2)
        self.assertEqual(2, db1.len())



class BipartiteGraphDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        letters = string.ascii_letters
        self.db = []
        for i in range(6):
            self.db.append('/tmp/' + ''.join(random.choice(letters) for i in range(10)))

    def tearDown(self):
        for db in self.db:
            try:
                os.remove(db)
            except:
                pass
    

    def _testBipartiteGraphExtraction(self):
        db1 = bgd.BipartiteGraphDataset(self.db[0])
        db1.add(bg.BipartiteGraph(torch.tensor([0]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        db1.add(bg.BipartiteGraph(torch.tensor([1]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        self.assertEqual(db1.len(), 2)

        db2 = bgd.BipartiteGraphDataset(self.db[1])
        db2.add(bg.BipartiteGraph(torch.tensor([2]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        db2.add(bg.BipartiteGraph(torch.tensor([3]), np.array([[0], [0]]), [0], np.array([0]), [0], [0], [0], [0]))
        self.assertEqual(db2.len(), 2)

        db = bgd.BipartiteGraphDatasets([self.db[0], self.db[1]])

        self.assertEqual(db.get(0).constraint_features, torch.tensor([0]))
        self.assertEqual(db.get(1).constraint_features, torch.tensor([1]))
        self.assertEqual(db.get(2).constraint_features, torch.tensor([2]))
        self.assertEqual(db.get(3).constraint_features, torch.tensor([3]))

        for i in range(4):
            t = db.get(i)
            self.assertEqual(t.constraint_features, torch.tensor([i]))


if __name__ == "__main__":
    unittest.main()
