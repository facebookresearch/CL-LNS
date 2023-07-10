# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import pyscipopt as scip


# Wrap is the scip solver under a common API
class ScipSolver:
    def __init__(self, timeout_s=None):
        self.constraints = []
        self.maximize = True
        self.timeout = timeout_s
        self.model = scip.Model()

    def create_integer_var(self, name, lower_bound, upper_bound):
        v = self.model.addVar(name=name, lb=lower_bound, ub=upper_bound, vtype="I")
        return v

    def create_real_var(self, name, lower_bound, upper_bound):
        v = self.model.addVar(name=name, lb=lower_bound, ub=upper_bound, vtype="C")
        return v

    def create_binary_var(self, name):
        v = self.model.addVar(name=name, vtype="B")
        return v

    def set_objective_function(self, equation, maximize=True):
        self.model.setObjective(equation)
        if maximize:
            self.model.setMaximize()
        else:
            self.model.setMinimize()

    def add_constraint(self, cns):
        self.model.addCons(cns)

    def disable_presolver(self):
        self.model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        self.model.setBoolParam("lp/presolving", False)

    def disable_cuts(self):
        self.model.setSeparating(scip.SCIP_PARAMSETTING.OFF)

    def disable_heuristics(self):
        self.model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


    def solve(self):
        # Solve the problem. Return the result as a dictionary of values
        # indexed by the corresponding variables or an empty dictionary if the
        # problem is infeasible.
        if self.timeout:
            self.model.setParam('limits/time', self.timeout)

        self.model.optimize()
           
        sol = None
        if self.model.getNSols() > 0:
            sol = self.model.getBestSol()
        return sol

    def primal_dual_gap(self):
        return (self.model.getObjVal(), self.model.getDualbound())

    def primal_dual_integral(self):
        # TBD
        return None

    def load(self, mps_filename):
        self.model.readProblem(mps_filename)

    def export(self, lp_output_filename):
        assert False

    def as_scip_model(self):
        return self.model
