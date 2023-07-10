# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import sys

import scip_solver as scip
import xpress_solver as xp


# Wrap is the various MILP solvers including SCIP and Xpress under a unified API
# to ensure that we can easily try several solvers to check which one performs best
# on a given instance.
class ILPSolver:
    def __init__(self, timeout_s=None, engine="xpress"):
        if engine == "xpress":
            self.solver = xp.XpressSolver(timeout_s)
        else:
            print(engine)
            assert engine == "scip"
            self.solver = scip.ScipSolver(timeout_s)

    def create_integer_var(self, name, lower_bound=None, upper_bound=None):
        assert name is not None
        if type(name) != str:
            name = str(name)
        lb = -sys.maxsize if lower_bound is None else lower_bound
        ub = sys.maxsize if upper_bound is None else upper_bound
        return self.solver.create_integer_var(name, lb, ub)

    def create_real_var(self, name, lower_bound=None, upper_bound=None):
        assert name is not None
        if type(name) != str:
            name = str(name)
        lb = -float("inf") if lower_bound is None else lower_bound
        ub = float("inf") if upper_bound is None else upper_bound
        return self.solver.create_real_var(name, lb, ub)

    def create_binary_var(self, name):
        assert name is not None
        if type(name) != str:
            name = str(name)
        return self.solver.create_binary_var(name)

    def set_objective_function(self, equation, maximize=True):
        self.solver.set_objective_function(equation, maximize)

    def add_constraint(self, cns):
        self.solver.add_constraint(cns)

    def disable_presolver(self):
        self.solver.disable_presolver()

    def disable_cuts(self):
        self.solver.disable_cuts()

    def disable_heuristics(self):
        self.solver.disable_heuristics()


    def solve(self):
        return self.solver.solve()

    # Returns the primal dual gap as the (upper bound, lower bound) tuple. This
    # should only be called after the problem has been solved.
    def primal_dual_gap(self):
        return self.solver.primal_dual_gap()

    # Returns the integral of the primal-dual gap over time. This
    # should only be called after the problem has been solved.
    def primal_dual_integral(self):
        return self.solver.primal_dual_integral()

    # Import the problem from the specified mps file
    def load(self, mps_filename):
        #self.solver.import(mps_filename)
        return 

    # Export the problem in the specified mps file
    def export(self, lp_output_filename):
        return self.solver.export(lp_output_filename)

    # Access the underlying scip.Model. Only valid if the engine is SCIP
    def as_scip_model(self):
        assert isinstance(self.solver, scip.ScipSolver)
        return self.solver.as_scip_model()

    # Access the underlying xpress.problem. Only valid if the engine is XPress
    def as_xpress_problem(self):
        assert isinstance(self.solver, xp.XpressSolver)
        return self.solver.as_xp_problem()
