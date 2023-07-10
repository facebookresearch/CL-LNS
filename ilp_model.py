# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from numpy.lib.utils import byte_bounds


class Solution:
    def __init__(self, model, scip_solution, obj_value):
        self.solution = {}
        for v in model.getVars():
            self.solution[v.name] = scip_solution[v]
        self.obj_value = obj_value

    def value(self, var):
        return self.solution[var.name]


class Model:
    def __init__(self, ecole_model, deep_copy=False):
        assert not deep_copy
        self.model = ecole_model.as_pyscipopt()
        self.initial_vals = {}
        for var in self.model.getVars():
            self.initial_vals[var.getIndex()] = (var.getLbGlobal(), var.getUbGlobal())

    def find_initial_solution(self, initial_time_limit=1):
        old_time_limit = self.model.getParam('limits/time')

        found = False
        time_limit = initial_time_limit
        while not found:
            self.model.setParam('limits/time', time_limit)
            self.model.optimize()
            num_solutions_found = self.model.getNSols()
            found = (num_solutions_found > 0)
            time_limit *= 2

        solution = self.model.getBestSol()
        obj_value = self.model.getSolObjVal(solution)

        self.model.setParam('limits/time', old_time_limit)

        return Solution(self.model, solution, obj_value)

    def get_primal_dual_bounds(self):
        # Must have attempted to optimize the model before querying for bounds
        if self.model.getNSols() == 0:
            raise RuntimeError("Must find a solution before calling get_primal_dual_bounds()")
        return (self.model.getPrimalbound(), self.model.getDualbound())

    def improve_solution(self, solution, vars_to_unassign):
        unassign_set = set()
        for v in vars_to_unassign:
            unassign_set.add(v.getIndex())
        preserve_set = {}
        for v in self.model.getVars():
            preserve_set[v.getIndex()] = solution.value(v)

        self.model.freeTransform()
        self.model.freeReoptSolve()
        for var in self.model.getVars():
            if var.getIndex() in unassign_set:
                #print("Unassigning " + str(var.getIndex()) + " with " + str(var.getLbGlobal()) + " / " + str(var.getUbGlobal()))
                lb, ub = self.initial_vals[var.getIndex()]
                self.model.chgVarLb(var, lb)
                self.model.chgVarLbGlobal(var, lb)
                self.model.chgVarUb(var, ub)
                self.model.chgVarUbGlobal(var, ub)
            else:
                val = preserve_set[var.getIndex()]
                self.model.chgVarLb(var, val)
                self.model.chgVarLbGlobal(var, val)
                self.model.chgVarUb(var, val)
                self.model.chgVarUbGlobal(var, val)

        self.model.optimize()
        assert self.model.getNSols() > 0
        solution = self.model.getBestSol()
        obj_value = self.model.getObjVal()
        return Solution(self.model, solution, obj_value)
