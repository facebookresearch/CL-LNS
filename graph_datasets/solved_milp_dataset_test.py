# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import unittest
import ilp_solver
import random
import string
from graph_datasets.solved_milp_dataset import SolvedMilpDataset


class SolvedMilpDatasetTest(unittest.TestCase):
    def setUp(self):
        # Create a small ILP problem
        solver = ilp_solver.ILPSolver(engine="scip")
        x1 = solver.create_integer_var("x1")
        x2 = solver.create_integer_var("x2")
        solver.add_constraint(10 * x1 + 15 * x2 >= 100.23)
        solver.add_constraint(20 * x1 + 16 * x2 >= 161.8)
        solver.add_constraint(17 * x1 + 11 * x2 >= 129.42)
        solver.set_objective_function(80 * x1 + 95 * x2, maximize=False)
        self.model = solver.as_scip_model()
        self.model.optimize()
        self.solution = self.model.getBestSol()
        self.obj_value = self.model.getObjVal()
        self.gap = self.model.getGap()

        letters = string.ascii_letters
        self.db_name = '/tmp/' + ''.join(random.choice(letters) for i in range(10))
        self.db_name2 = '/tmp/' + ''.join(random.choice(letters) for i in range(10))

    def test_read_write(self):
        dataset = SolvedMilpDataset(self.db_name)
        dataset.add(self.model, self.solution, self.obj_value, self.gap)
        a, b = dataset.get_one(self.model)

        sol = {}
        for v in self.model.getVars():
            val = self.solution[v]
            sol[v.getIndex()] = val
        self.assertEqual(a, sol)
        self.assertEqual(b, self.obj_value)

    def test_missing_entry(self):
        dataset = SolvedMilpDataset(self.db_name)
        try:
            a, b = dataset.get_one(self.model)
            found = True
        except:
            found = False
        self.assertFalse(found)

    def test_overwrite(self):
        dataset = SolvedMilpDataset(self.db_name)
        dataset.add(self.model, self.solution, 10, 23)
        dataset.add(self.model, self.solution, 1.0, 21)
        a, b = dataset.get_one(self.model)

        sol = {}
        for v in self.model.getVars():
            val = self.solution[v]
            sol[v.getIndex()] = val
        self.assertEqual(a, sol)
        self.assertEqual(b, 1.0)
    
        dataset.add(self.model, self.solution, 2.0, 22)
        a, b = dataset.get_one(self.model)
        self.assertEqual(b, 1.0)

    def test_multiple_entries(self):
        dataset = SolvedMilpDataset(self.db_name2, best_solution_only=False)
        dataset.add(self.model, self.solution, 50.0, 23)
        dataset.add(self.model, self.solution, 10.0, 21)
        dataset.add(self.model, self.solution, 2.0, 22)
        expected_obj_value = 50.0
        for a, b in dataset.get_all(self.model):
            sol = {}
            for v in self.model.getVars():
                val = self.solution[v]
                sol[v.getIndex()] = val
            self.assertEqual(a, sol)
            self.assertEqual(b, expected_obj_value)
            expected_obj_value /= 5

    def test_aggregate(self):
        dataset1 = SolvedMilpDataset(self.db_name)
        dataset1.add(self.model, self.solution, 10, 25)
        _, b = dataset1.get_one(self.model)
        self.assertEqual(b, 10)

        dataset2 = SolvedMilpDataset(self.db_name)
        dataset2.add(self.model, self.solution, 1, 11)
        dataset1.merge(dataset2)
        _, b = dataset1.get_one(self.model)
        self.assertEqual(b, 1)

        dataset3 = SolvedMilpDataset(self.db_name)
        dataset3.add(self.model, self.solution, 5, 17)
        dataset1.merge(dataset3)
        _, b = dataset1.get_one(self.model)
        self.assertEqual(b, 1)