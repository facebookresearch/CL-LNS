# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import unittest

import numpy as np
import string
import random
import os
import sys
import graph_datasets.evaluation_data as ed
import ilp_solver


class EvaluationDataMiningTest(unittest.TestCase):
    def setUp(self):
        solver = ilp_solver.ILPSolver(engine="scip")
        x1 = solver.create_integer_var("x1")
        x2 = solver.create_integer_var("x2")
        solver.add_constraint(10 * x1 + 15 * x2 >= 100)
        solver.add_constraint(20 * x1 + 16 * x2 >= 160)
        solver.add_constraint(17 * x1 + 11 * x2 >= 130)

        # Minimize the objective
        solver.set_objective_function(80 * x1 + 95 * x2, maximize=False)
        self.instance1 = solver.as_scip_model()

        solver = ilp_solver.ILPSolver(engine="scip")
        x1 = solver.create_integer_var("x1")
        x2 = solver.create_integer_var("x2")
        x3 = solver.create_integer_var("x3")
        solver.add_constraint(10 * x1 + 15 * x2 >= 100)
        solver.add_constraint(20 * x1 + 16 * x2 >= 160)
        solver.add_constraint(17 * x3 + 11 * x2 >= 130)

        # Minimize the objective
        solver.set_objective_function(80 * x1 + 95 * x2 + 17 * x3, maximize=False)
        self.instance2 = solver.as_scip_model()

        letters = string.ascii_letters
        self.db = []
        for i in range(3):
            self.db.append('/tmp/' + ''.join(random.choice(letters) for i in range(10)))

    def tearDown(self):
        for db in self.db:
            try:
                os.remove(db)
            except:
                pass
       
    def testSingleVersion(self):
        data = ed.EvaluationData(self.db[0])
        # Format is instance, model_version, step_id, primal, dual, nb_nodes, timestamp
        data.add(self.instance1, "v1", 1, 0, 123, 2, 1.0)
        data.add(self.instance1, "v1", 2, 0, 125, 4, 1.5)
        data.commit()

        miner = ed.EvaluationDataMining(self.db[0], ["v1"])
        nb_nodes, integrals_over_nodes, integrals_over_time = miner.compute_metrics()
        
        self.assertEqual(nb_nodes, {"v1": [4]})
        self.assertEqual(integrals_over_nodes, {"v1": [246.0]})
        self.assertEqual(integrals_over_time, {"v1": [61.5]})
        
    def testMultipleVersions(self):
        data = ed.EvaluationData(self.db[1])
        # Format is instance, model_version, step_id, primal, dual, nb_nodes, timestamp
        data.add(self.instance1, "v1", 1, 0, 123, 2, 1.0)
        data.add(self.instance1, "v1", 2, 0, 125, 4, 1.5)
        data.add(self.instance1, "v2", 4, 0, 321, 3, 2.0)
        data.add(self.instance1, "v2", 5, 0, 432, 7, 2.7)
        data.commit()

        miner = ed.EvaluationDataMining(self.db[1], ["v1"])
        nb_nodes, integrals_over_nodes, integrals_over_time = miner.compute_metrics()
        
        self.assertEqual(nb_nodes, {"v1": [4]})
        self.assertEqual(integrals_over_nodes, {"v1": [246.0]})
        self.assertEqual(integrals_over_time, {"v1": [61.5]})

        miner = ed.EvaluationDataMining(self.db[1], ["v2"])
        nb_nodes, integrals_over_nodes, integrals_over_time = miner.compute_metrics()

        self.assertEqual(nb_nodes, {"v2": [7]})
        self.assertEqual(integrals_over_nodes, {"v2": [1284.0]})
        self.assertEqual(integrals_over_time, {"v2": [224.70000000000005]})

        miner = ed.EvaluationDataMining(self.db[1], ["v1", "v2"])
        nb_nodes, integrals_over_nodes, integrals_over_time = miner.compute_metrics()

        self.assertEqual(nb_nodes, {"v1": [4], "v2": [4]})
        self.assertEqual(integrals_over_nodes, {"v1": [246.0], "v2": [321.0]})
        self.assertEqual(integrals_over_time, {"v1": [61.5], "v2": [0.0]})

    def testMultipleVersionsMultipleInstances(self):
        data = ed.EvaluationData(self.db[2])
        # Format is instance, model_version, step_id, primal, dual, nb_nodes, timestamp
        data.add(self.instance1, "v1", 1, 0, 123, 2, 1.0)
        data.add(self.instance1, "v1", 2, 0, 125, 4, 1.5)
        data.add(self.instance1, "v2", 4, 0, 321, 3, 2.0)
        data.add(self.instance1, "v2", 5, 0, 432, 7, 2.7)
        data.add(self.instance2, "v1", 11, 0, 1123, 12, 11.0)
        data.add(self.instance2, "v1", 12, 0, 1125, 14, 11.5)
        data.add(self.instance2, "v2", 14, 0, 1321, 13, 12.0)
        data.add(self.instance2, "v2", 15, 0, 1432, 17, 12.7)
        data.commit()

        miner = ed.EvaluationDataMining(self.db[2], ["v1"])
        nb_nodes, integrals_over_nodes, integrals_over_time = miner.compute_metrics()
        
        self.assertEqual(nb_nodes, {"v1": [4, 14]})
        self.assertEqual(integrals_over_nodes, {"v1": [246.0, 2246.0]})
        self.assertEqual(integrals_over_time, {"v1": [61.5, 561.5]})

        miner = ed.EvaluationDataMining(self.db[2], ["v2"])
        nb_nodes, integrals_over_nodes, integrals_over_time = miner.compute_metrics()
     
        self.assertEqual(nb_nodes, {"v2": [7, 17]})
        self.assertEqual(integrals_over_nodes, {"v2": [1284.0, 5284.0]})
        self.assertEqual(integrals_over_time, {"v2": [224.70000000000005, 924.699999999999]})

        miner = ed.EvaluationDataMining(self.db[2], ["v1", "v2"])
        nb_nodes, integrals_over_nodes, integrals_over_time = miner.compute_metrics()
      
        self.assertEqual(nb_nodes, {"v1": [4, 14], "v2": [4, 14]})
        self.assertEqual(integrals_over_nodes, {"v1": [246.0, 2246.0], "v2": [321.0, 1321.0]})
        self.assertEqual(integrals_over_time, {"v1": [61.5, 561.5], "v2": [0, 0]})

    def _testRealResults(self):
        miner = ed.EvaluationDataMining("/data/home/benoitsteiner/ml4co-dev/ml4co/results.db", ["SCIP", "09/09/2021 17:26:14"])
        nb_nodes, integrals_over_nodes, integrals_over_time = miner.compute_metrics()
        print(str(nb_nodes))
        print(str(integrals_over_nodes))
        print(str(integrals_over_time))

        self.assertEqual(1, 2)