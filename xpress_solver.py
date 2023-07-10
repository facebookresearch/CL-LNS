# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import xpress as xp


# Wrap is the xpress solver (https://pypi.org/project/xpress/, doc available at
# https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/GUID-616C323F-05D8-3460-B0D7-80F77DA7D046.html)
class XpressSolver:
    def __init__(self, timeout_s=None):
        self.vars = []
        self.constraints = []
        self.maximize = True
        self.timeout = timeout_s
        self.pd_gap = (None, None)
        self.pd_integral = None

    def create_integer_var(self, name, lower_bound, upper_bound):
        v = xp.var(name=name, lb=lower_bound, ub=upper_bound, vartype=xp.integer)
        self.vars.append(v)
        return v

    def create_real_var(self, name, lower_bound, upper_bound):
        v = xp.var(name=name, lb=lower_bound, ub=upper_bound, vartype=xp.continuous)
        self.vars.append(v)
        return v

    def create_binary_var(self, name):
        v = xp.var(name=name, vartype=xp.binary)
        self.vars.append(v)
        return v

    def set_objective_function(self, equation, maximize):
        self.of = equation
        self.maximize = maximize

    def add_constraint(self, cns):
        self.constraints.append(cns)

    def disable_presolver(self):
        # TBD
        pass

    def disable_cuts(self):
        # TBD
        pass
    def disable_heuristics(self):
        # TBD
        pass

    def solve(self):
        # Solve the problem. Return the result as a dictionary of values
        # indexed by the corresponding variables or an empty dictionary if the
        # problem is infeasible.
        p = self.as_xpress_problem()

        # Make sure the problem is feasible
        if p.iisfirst(0) == 0:
            raise RuntimeError("Problem is not feasible")
        # Solve and return the values for all the variables.
        if self.timeout:
            p.controls.maxtime = self.timeout
        p.solve()
        result = {}
        for v in self.vars:
            result[v] = p.getSolution(v)

        # Record the value of the primal dual gap.
        self.pd_gap = (p.getAttrib("mipbestobjval"), p.getAttrib("bestbound"))
        self.pd_integral = p.getAttrib("primaldualintegral")

        return result

    def primal_dual_gap(self):
        return self.pd_gap

    def primal_dual_integral(self):
        return self.pd_integral

    def load(self, mps_filename):
        # Not supported yet.
        assert False

    def export(self, lp_output_filename):
        p = xp.problem(self.vars, self.of, self.constraints)
        if self.maximize:
            p.chgobjsense(xp.maximize)
        else:
            p.chgobjsense(xp.minimize)
        p.write(lp_output_filename, "lp")

    def as_xpress_problem(self):
        p = xp.problem(self.vars, self.of, self.constraints)
        if self.maximize:
            p.chgobjsense(xp.maximize)
        else:
            p.chgobjsense(xp.minimize)
        return p
