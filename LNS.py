# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from graph_datasets.bipartite_graph import *
from graph_datasets.bipartite_graph_dataset import BipartiteGraphDataset
import graph_datasets.bipartite_graph_observations as bgo
from instance_loader import InstanceLoader
from ilp_model import Solution
import argparse
import copy
import random
import pyscipopt
from neural_nets.gnn_policy import GNNPolicy
from pyscipopt import quicksum
import time
import ecole
import networkx as nx
import pickle
import statistics
from graph_datasets.featurization_test import make_obs 
import os
import sys
from IPython import embed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global WIND_SIZE
WIND_SIZE = 3



COLLECT_SOLVE_TIME_LIMIT = 60 * 60 * 1 #2hours
STEP_PER_COLLECT = 1

class MyEvent(pyscipopt.Eventhdlr):
    def eventinit(self):
        print("init event")
        self._start_time = time.monotonic()
        self.scip_log = []
        self.start_time = time.monotonic()
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        print("exit event")
        #self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        print("exec event")
        self.end_time = time.monotonic()
        #obj = self.model.getPrimalbound()
        #print(obj, self.end_time - self._start_time)
        sol = self.model.getBestSol()
        obj = self.model.getSolObjVal(sol)
        Sol = Solution(self.model, sol, obj)
        log_entry = dict()
        log_entry['best_primal_sol'] = Sol
        log_entry['best_primal_scip_sol'] = sol
        log_entry['primal_bound'] = obj
        log_entry['solving_time'] = self.end_time - self.start_time
        log_entry['iteration_time'] = self.end_time - self.start_time
        log_entry['selection_time'] = 0
        var_index_to_value = dict()
        for v in self.model.getVars():
            v_name = v.name
            v_value = Sol.value(v)
            var_index_to_value[v_name] = v_value
        log_entry['var_index_to_value'] = copy.deepcopy(var_index_to_value)
        self.scip_log.append(log_entry)
        self.start_time = self.end_time
        #print(log_entry['primal_bound'], log_entry['solving_time'], self.end_time - self._start_time)




def run_vanilla_scip(model, args):
    model = model.__repr__.__self__
    event = MyEvent()
    model.includeEventhdlr(
        event,
        "",
        ""
    )
    model.setParam("limits/time", args.time_limit)
    if "AGGR" in args.destroy_heuristic:
        print("Enabled aggressive mode for BnB with SCIP")
        model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.AGGRESSIVE)
    model.optimize()
    return event.scip_log


NUM_OF_EXPERT_SAMPLES = 50


def isInteger(x):
    return abs(x - round(x)) <=1e-8
        

def add_scip_config_to_mip_model(scip_config):
    for param, value in scip_config.items():
        model.setRealParam(param, value)
    return model

def scip_solve(model, incumbent_solution = None, scip_config = None, timer = None, get_initial_solution = False, primal_bound = None, prev_LNS_log = None, get_num_solutions = 1, mute = False, isMIPLIB = False):

    start_time = time.monotonic()

    if primal_bound is not None:
        objective_sense = model.getObjectiveSense()
        if objective_sense == "minimize":
            model.addCons(model.getObjective() <= primal_bound + 1e-8)
            #if not mute:
                #print("---added a new constraint using the primal bound for minimization")
        else:
            model.addCons(model.getObjective() >= primal_bound - 1e-8)
            #if not mute:
                #print("---added a new constraint using the primal bound for maximization")

        #print("added a new constraint using the primal bound")

    #model = add_scip_config_to_mip_model(model, scip_config)
    if scip_config is not None:
        for param, value in scip_config.items():
            #print(param, value)
            model.setParam(param, value)
    found = True
    init_time = None
    if get_initial_solution == True:
        found = False
        
        #runtime_left = model.getParam('limits/time')
        runtime_left = 900
        #time_limit = 610 if isMIPLIB else 10
        time_limit = model.getParam('limits/time')
        while not found and time_limit <= runtime_left:
            #if time_limit * 2 >= runtime_left:
            #    time_limit = runtime_left
            #time_limit = min(time_limit, runtime_left)
            
            model.setParam('limits/time', time_limit)
            start_time = time.monotonic()
            #embed()
            model.optimize()
            end_time = time.monotonic()
            init_time = end_time - start_time
            num_solutions_found = model.getNSols()
            found = (num_solutions_found > 0)
            #runtime_left -= time_limit
            if time_limit >= runtime_left-1e3:
                break
            time_limit *= 2
            time_limit = min(time_limit, runtime_left)
    else:
        model.optimize()
        end_time = time.monotonic()
        init_time = end_time - start_time
        if not mute:
            print("finished optimizing sub mip")
    end_time = time.monotonic()
    

    status = model.getGap()#model.getStatus()
    log_entry = None
    if found == True:
        if model.getNSols() == 0: # if no solution in a LNS iteration, then return the same copy of the previous log but change the runtime
            if prev_LNS_log is None:
                return -1, None
            log_entry = dict()
            for k, v in prev_LNS_log.items():
                log_entry[k] = v
            #log_entry = copy.deepcopy(prev_LNS_log)
            log_entry['solving_time'] = init_time
            return status, log_entry

        sol = model.getBestSol()
        obj = model.getSolObjVal(sol)
        Sol = Solution(model, sol, obj)
        log_entry = {}
        log_entry['best_primal_sol'] = Sol
        log_entry['best_primal_scip_sol'] = sol
        log_entry['primal_bound'] = obj
        if not (init_time is None):
            log_entry['solving_time'] = init_time
            log_entry['iteration_time'] = init_time
        else:
            log_entry['solving_time'] = end_time - start_time
            log_entry['iteration_time'] = end_time - start_time
        log_entry['selection_time'] = 0
        var_index_to_value = dict()
        for v in model.getVars():
            v_name = v.name
            v_value = Sol.value(v)
            var_index_to_value[v_name] = v_value
        log_entry['var_index_to_value'] = copy.deepcopy(var_index_to_value)

        if get_num_solutions > 1:
            var_index_to_values = dict()
            for v in model.getVars():
                var_index_to_values[v.name] = []
            #embed()
            sol_list = model.getSols()
            obj_list = []
            
            sol_list.reverse()
            #if len(sol_list) > 30:
            #    sol_list= sol_list[:30]

            for sol in sol_list:
                Sol = Solution(model, sol, obj)
                obj = model.getSolObjVal(sol)
                if primal_bound is not None:
                    objective_sense = model.getObjectiveSense()
                    if objective_sense == "minimize":
                        if obj >= primal_bound - 1e-8: continue
                        #model.addCons(model.getObjective() <= primal_bound + 1e-8)
                    else:
                        if obj <= primal_bound + 1e-8: continue
                        #model.addCons(model.getObjective() >= primal_bound - 1e-8)
                for v in model.getVars():
                    v_name = v.name
                    v_value = Sol.value(v)
                    v_incumbent_value = incumbent_solution.value(v)
                    var_index_to_values[v_name].append(0 if round(v_value) == round(v_incumbent_value) else 1)
                obj_list.append((obj, primal_bound))
            log_entry['var_index_to_values'] = copy.deepcopy(var_index_to_values)
            log_entry['primal_bounds'] = copy.deepcopy(obj_list)
            #embed()
        else:
            log_entry['var_index_to_values'] = None
            log_entry['primal_bounds'] = None


        #log_entry['solving_time_calibrated'] = timer.elapsed_calibrated_time
        #sol_data.write(log_entry, force_save_sol=True)
    #print(sol)
    return status, log_entry



def get_LP_relaxation_solution(model):
    LP_relaxation = pyscipopt.Model(sourceModel = model, origcopy = True)
    for var in LP_relaxation.getVars():
        LP_relaxation.chgVarType(var, 'C')
    scip_solve_LP_relaxation_config = {
        'limits/time' : 300,
    }
    #status, log_entry = scip_solve(LP_relaxation, scip_config = scip_solve_LP_relaxation_config)
    return scip_solve(LP_relaxation, scip_config = scip_solve_LP_relaxation_config)





def random_sample_variable_based(model, G, variables_to_nodes, neighborhood_size, pre_selected_pivot = None, pivot_num = 1):
    all_int_variables = [v.name for v in model.getVars() if v.vtype()  in ["BINARY", "INTEGER"]]

    pivot_node = []
    for i in range(pivot_num):
        sample_var = random.choice(all_int_variables)
        while variables_to_nodes[sample_var] in pivot_node:
            sample_var = random.choice(all_int_variables)
        pivot_node.append(variables_to_nodes[sample_var]) 

    if pre_selected_pivot is not None:
        pivot_node = [variables_to_nodes[var] for var in pre_selected_pivot]

    destroy_nodes = pivot_node
    current_group = pivot_node

    top = [v for v in G.nodes() if G.nodes[v]['bipartite'] == 0]
    pos = nx.bipartite_layout(G, top)


    for u, v in G.edges():
        assert(G.nodes[u]['bipartite'] == 0)
        assert(G.nodes[v]['bipartite'] == 1)

    while len(destroy_nodes) < neighborhood_size:
        new_group = []
        for v in current_group:
            for n in G.neighbors(v):
                new_group.append(n)
                #print(G.in_degree(n))
                assert(G.nodes[n]['bipartite'] == 1)
        new_group = list(set(new_group))
        G_predecessors = []
        for v in new_group:
            for n in G.predecessors(v):
                if not (G.nodes[n]["scip_variable"] in all_int_variables):
                    continue
                G_predecessors.append(n)
                assert(G.nodes[n]['bipartite'] == 0)
        #new_group = [n for v in current_group for n in G.neighbors(v)]
        #G_predecessors = [n for v in new_group for n in G.predecessors(v)]
        G_predecessors = list(set(G_predecessors) - set(destroy_nodes))
        if len(G_predecessors) == 0: break
        for v in G_predecessors:
            assert G.nodes[v]['bipartite'] == 0, str(v)
        if len(G_predecessors) + len(destroy_nodes) <= neighborhood_size:
            destroy_nodes = destroy_nodes + G_predecessors
        else:
            destroy_nodes = destroy_nodes + random.sample(G_predecessors, neighborhood_size - len(destroy_nodes))
        current_group = copy.deepcopy(G_predecessors)
    
    for v in destroy_nodes:
        assert(G.nodes[v]["scip_variable"] in all_int_variables)
    destroy_variables = [G.nodes[v]["scip_variable"] for v in destroy_nodes]
    assert(len(destroy_variables) <= neighborhood_size)
    return destroy_variables


def normalize_score(score, neighborhood_size):
    l = 0
    r = 100
    while r - l > 1e-8:
        m = (l + r) * 0.5
        tp_score = torch.pow(score, m)
        tp_sum = torch.sum(tp_score).item()
        if tp_sum > neighborhood_size:
            l = m
        else:
            r = m
    return torch.pow(score, l)

def normalize_score2(logit, neighborhood_size):
    l = 0
    r = 1
    while r - l > 1e-8:
        m = (l + r) * 0.5
        tp_logit = torch.mul(logit, m)
        tp_score = torch.sigmoid(tp_logit)
        tp_sum = torch.sum(tp_score).item()
        if tp_sum < neighborhood_size:
            r = m
        else:
            l = m
    tp_logit = torch.mul(logit, l)
    tp_score = torch.sigmoid(tp_logit)
    return tp_score



#ML_info = (policy, observation, incumbent_history, LB_relaxation_history)
def create_neighborhood_with_heuristic(model, LNS_log, neighborhood_size = 20, heuristic = "RANDOM", bipartite_graph = None, variables_to_nodes = None, improved = None, num_samples = 30, eps_clip = 0.05, ML_info = None, original_neighborhood_size = None, get_num_solutions = 1):
    if original_neighborhood_size is None:
        original_neighborhood_size = neighborhood_size
    all_variables = model.getVars()
    all_int_variables = [v.name for v in model.getVars() if v.vtype() in ["BINARY", "INTEGER"]] # currently only considering binary variables


    objective_sense = model.getObjectiveSense()
    obj_sense = 1 if objective_sense == "minimize" else -1

    if heuristic == "RANDOM":
        if neighborhood_size >= len(all_int_variables):
            return all_int_variables, None
        else:
            return random.sample(all_int_variables, neighborhood_size), None

    elif heuristic == "VARIABLE":
        assert(bipartite_graph is not None)
        assert(variables_to_nodes is not None)
        return random_sample_variable_based(model, bipartite_graph, variables_to_nodes, neighborhood_size), None
    


    elif "ML" in heuristic:
        #embed()
        ML_inference_start_time = time.monotonic()
        assert ML_info is not None
        local_branching_mip = pyscipopt.Model(sourceModel = model, origcopy = True)
        #print(model)
        #print(local_branching_mip)
        incumbent_solution = LNS_log[-1]['best_primal_sol']
        variables_equal_one = []
        variables_equal_zero = []
        
        all_int_variables = [v.name for v in local_branching_mip.getVars() if v.vtype() in ["BINARY", "INTEGER"]]
        for v in local_branching_mip.getVars():
            if v.name in all_int_variables:
                v_value = incumbent_solution.value(v)
                if round(v_value) == 1:
                    variables_equal_one.append(v)
                else:
                    variables_equal_zero.append(v)
        #need to decide whether to use original neighborhood size or adaptive one
        if "ORINH" in heuristic:
            local_branching_mip.addCons(quicksum(v for v in variables_equal_zero) + quicksum( (1-v)  for v in variables_equal_one) <= original_neighborhood_size) 
            print("constructed mip for local branching with neighorhood size %d" % (original_neighborhood_size))
        else:
            local_branching_mip.addCons(quicksum(v for v in variables_equal_zero) + quicksum( (1-v)  for v in variables_equal_one) <= neighborhood_size) 
            print("constructed mip for local branching with neighorhood size %d" % (neighborhood_size))

        int_var = [v for v in model.getVars() if v.vtype() in ["BINARY", "INTEGER"]]
        LB_relaxation_solution = []
        if "feat1" in args.mode or "feat2" in args.mode:
            #print("No LP solving")
            for var in int_var:
                LB_relaxation_solution.append(0)
            LB_LP_relaxation_solution = LNS_log[-1]['best_primal_sol']
            
        else:
            LB_LP_relaxation_status, LB_LP_relaxation_log_entry = get_LP_relaxation_solution(local_branching_mip)
            LB_LP_relaxation_solution = LB_LP_relaxation_log_entry['best_primal_sol']
            for var in int_var:
                LB_relaxation_solution.append(LB_LP_relaxation_solution.value(var))
        #embed()
        policy, observation, incumbent_history, _LB_relaxation_history = ML_info
        LB_relaxation_history = copy.deepcopy(_LB_relaxation_history)
        LB_relaxation_history.append(LB_relaxation_solution)
        dynamic_features = torch.zeros((observation.column_features.shape[0], WIND_SIZE * 3), dtype = torch.float32)
        number_of_history_added = 0
        assert(len(incumbent_history) == len(LB_relaxation_history))
        for i in reversed(range(len(LB_relaxation_history))):
            dynamic_features[:, number_of_history_added*3]  = torch.FloatTensor([1]*len(int_var))
            dynamic_features[:, number_of_history_added*3+1] = torch.FloatTensor(incumbent_history[i])
            if not ("feat1" in args.mode or "feat2" in args.mode):
                dynamic_features[:, number_of_history_added*3+2] = torch.FloatTensor(LB_relaxation_history[i])
            else:
                dynamic_features[:, number_of_history_added*3+2] = torch.zeros(len(LB_relaxation_history[i]))
                #print("No relaxation features")
            number_of_history_added += 1
            if number_of_history_added == WIND_SIZE:
                break
        observation.column_features[:, -WIND_SIZE * 3:] = dynamic_features
        with torch.no_grad():
            obs = (observation.row_features.to(DEVICE),
                   observation.edge_features.indices.to(DEVICE),
                   observation.edge_features.values.to(DEVICE),
                   observation.column_features.to(DEVICE))
            logits = policy(*obs)
            score = torch.sigmoid(logits)
        info = dict()
        
        #info["LB_gap"] = status
        info["LB_LP_relaxation_solution"] = LB_LP_relaxation_solution
        

        distribution_destroy_variable = []
        all_int_variables = [v.name for v in int_var]
        for i, v in enumerate(model.getVars()):
            if v.name in all_int_variables:
                v_value = score[i].item()
                v_logit = logits[i].item()
                distribution_destroy_variable.append((v.name, v_value, v_logit))
        distribution_destroy_variable.sort(key = lambda x: x[2])
        #from IPython import embed; embed();
        num_cand = len(distribution_destroy_variable)
        info = dict()
        info["LB_LP_relaxation_solution"] = LB_LP_relaxation_solution
        destroy_variables = []

        ML_inference_end_time = time.monotonic()
        print("ML inference time=", ML_inference_end_time-ML_inference_start_time)
        info["ML_time"] = ML_inference_end_time-ML_inference_start_time
        #embed()
        best_primal_bound = None
        
        if "SAMPLE" in heuristic:
            #embed()
            normalized_score = normalize_score(score, neighborhood_size)
            if torch.sum(normalized_score).item() > neighborhood_size * 1.5: #numerical issues
                normalized_score = normalize_score2(logits, neighborhood_size)
            #embed()
            for i, v in enumerate(model.getVars()):
                if v.name in all_int_variables:
                    v_value = normalized_score[i].item() #score[i].item()
                    coin_flip = random.uniform(0, 1)
                    if coin_flip <= v_value:
                        destroy_variables.append(v.name)

            
            return destroy_variables, info

        elif "GREEDY" in heuristic:
            return [v_name for v_name, _, __ in distribution_destroy_variable[-min(neighborhood_size, num_cand):]], info
        else:
            assert False, "Unknown sampling methods for ML"
            

        return destroy_variables, info
        

    elif heuristic.startswith("LOCAL"):
        local_branching_mip = pyscipopt.Model(sourceModel = model, origcopy = True)
        #print(model)
        #print(local_branching_mip)
        incumbent_solution = LNS_log[-1]['best_primal_sol']
        variables_equal_one = []
        variables_equal_zero = []
        #embed()
        
        
        all_int_variables = [v.name for v in local_branching_mip.getVars() if v.vtype() in ["BINARY", "INTEGER"]]
        for v in local_branching_mip.getVars():
            if v.name in all_int_variables:
                v_value = incumbent_solution.value(v)
                if round(v_value) == 1:
                    variables_equal_one.append(v)
                else:
                    variables_equal_zero.append(v)
        original_LP_relaxation_status, original_LP_relaxation_log_entry = None, None # get_LP_relaxation_solution(local_branching_mip)
        local_branching_mip.addCons(quicksum(v for v in variables_equal_zero) + quicksum( (1-v)  for v in variables_equal_one) <= neighborhood_size)
        print("constructed mip for local branching")
        scip_solve_local_branching_config = {
            'limits/time' : 3600 if "LONG" in heuristic else 600, 
        }
        if args.mode == "COLLECT" or args.collect_along_test == 1:
            scip_solve_local_branching_config['limits/time'] = COLLECT_SOLVE_TIME_LIMIT

        destroy_variables = []
        if "RELAXATION" in heuristic:
            LB_LP_relaxation_status, LB_LP_relaxation_log_entry = get_LP_relaxation_solution(local_branching_mip)
            #original_LP_relaxation_solution = original_LP_relaxation_log_entry['best_primal_sol']
            original_LP_relaxation_solution = LB_LP_relaxation_log_entry['best_primal_sol']
            LB_LP_relaxation_solution = LB_LP_relaxation_log_entry['best_primal_sol']
            both_integer = 0
            LB_integer = 0
            original_integer = 0
            for v in all_variables:
                if v.name in all_int_variables:
                    v_orignal_value = original_LP_relaxation_solution.value(v)
                    v_LB_value = LB_LP_relaxation_solution.value(v)
                    if isInteger(v_orignal_value) and isInteger(v_LB_value):
                        both_integer += 1
                    elif isInteger(v_orignal_value):
                        original_integer +=1
                    elif isInteger(v_LB_value):
                        LB_integer += 1
            #print("---LB LP runtime", LB_LP_relaxation_log_entry['solving_time'])#, "original LP runtime", original_LP_relaxation_log_entry['solving_time'])
            #print("---both integer", both_integer, "original integer", original_integer, "LB integer", LB_integer)
            #print("---selecting using LP relaxation")
            same_integer_value_inc_and_LB_LP = 0
            same_integer_value_LB_and_LB_LP = 0
            if "RS" in heuristic:   
                distribution_destroy_variable = []
                for v in all_variables:
                    if v.name in all_int_variables:
                        v_value = incumbent_solution.value(v)
                        #v_LB_value = local_branching_solution.value(v)
                        v_LB_LP_value = LB_LP_relaxation_solution.value(v)
                        distribution_destroy_variable.append((v.name, abs(v_LB_LP_value - v_value), v_value))
                best_destroy_variables = None
                best_primal_bound = None
                
                NUM_OF_EXPERT_SAMPLES = num_samples
                for _ in range(NUM_OF_EXPERT_SAMPLES):
                    tmp_destroy_variables = []
                    for v_name, prob, t in distribution_destroy_variable:
                        coin_flip = random.uniform(0, 1)
                        #if coin_flip <= max(min(1 - eps_clip, prob), eps_clip):
                        if coin_flip <= (1 - 2 * eps_clip) * prob + eps_clip:
                            tmp_destroy_variables.append(v_name)
                    if NUM_OF_EXPERT_SAMPLES == 1:
                        info = dict()
                        info["LB_LP_relaxation_solution"] = LB_LP_relaxation_solution
                        return tmp_destroy_variables, info
                    sub_mip = create_sub_mip(model, tmp_destroy_variables,  LNS_log[-1]['best_primal_sol'])
                    scip_solve_destroy_config = {
                        'limits/time' : 120,
                    }
                
                    status, log_entry = scip_solve(sub_mip, primal_bound = LNS_log[-1]['primal_bound'], 
                        scip_config = scip_solve_destroy_config)
                    print("sample improvement", log_entry['primal_bound'])
                    if best_destroy_variables is None or log_entry['primal_bound'] * obj_sense < best_primal_bound * obj_sense:
                        best_primal_bound = log_entry['primal_bound']
                        best_destroy_variables = copy.deepcopy(tmp_destroy_variables)

                print("best destroy variable chosen with obj =", best_primal_bound)
                info = dict()
                info["LB_LP_relaxation_solution"] = LB_LP_relaxation_solution
                info["num_ori_relax_integer"] = original_integer
                info["num_LB_relax_integer"] = LB_integer
                info["num_both_integer"] = both_integer
                return best_destroy_variables, info

            elif "MI" in heuristic or "LI" in heuristic:
                distribution_destroy_variable = []
                for v in all_variables:
                    if v.name in all_int_variables:
                        v_value = incumbent_solution.value(v)
                        #v_LB_value = local_branching_solution.value(v)
                        v_LB_LP_value = LB_LP_relaxation_solution.value(v)
                        if True or abs(v_LB_LP_value - v_value) > 1e-8:
                            #distribution_destroy_variable.append((v.name, max(abs(v_LB_LP_value - v_value), 1 - abs(v_LB_LP_value - v_value)), v_value))
                            distribution_destroy_variable.append((v.name, abs(v_LB_LP_value - v_value), v_value))
                distribution_destroy_variable.sort(key = lambda x: x[1])
                #from IPython import embed; embed();
                num_cand = len(distribution_destroy_variable)
                info = dict()
                info["LB_LP_relaxation_solution"] = LB_LP_relaxation_solution
                info["num_ori_relax_integer"] = original_integer
                info["num_LB_relax_integer"] = LB_integer
                info["num_both_integer"] = both_integer
                if "MI" in heuristic:
                    return [v_name for v_name, _, __ in distribution_destroy_variable[:min(num_cand, neighborhood_size)]], info
                else:
                    return [v_name for v_name, _, __ in distribution_destroy_variable[-min(neighborhood_size, num_cand):]], info
            #elif "LI" in heuristic:
            #    pass
            else:
                for v in all_variables:
                    if v.name in all_int_variables:
                        v_value = incumbent_solution.value(v)
                        #v_LB_value = local_branching_solution.value(v)
                        v_LB_LP_value = LB_LP_relaxation_solution.value(v)

                        if isInteger(v_LB_LP_value):
                            if round(v_LB_LP_value) == round(v_value):
                                same_integer_value_inc_and_LB_LP += 1

                        #print("---selecting using LP relaxation")
                        if isInteger(v_LB_LP_value):
                            if round(v_LB_LP_value) != round(v_value):
                                #destroy_variables.append(v.getIndex())
                                destroy_variables.append(v.name)
                        else:
                            #destroy_variables.append(v.getIndex())
                            destroy_variables.append(v.name)

        
            #print("---num same integer values LB and LB LP", same_integer_value_LB_and_LB_LP)
            #print("---num same integer values inc and LB LP", same_integer_value_inc_and_LB_LP)
            #print("---num destroy variables", len(destroy_variables))
            if len(destroy_variables) > neighborhood_size:
                destroy_variables = random.sample(destroy_variables, neighborhood_size)
            #print("num of variables selected by LB relaxation", len(destroy_variables), "with LP obj =", LB_LP_relaxation_log_entry['primal_bound'])
            info = dict()
            #"num_LB_relax_integer"
            info["LB_LP_relaxation_solution"] = LB_LP_relaxation_solution
            info["num_ori_relax_integer"] = original_integer
            info["num_LB_relax_integer"] = LB_integer
            info["num_both_integer"] = both_integer
            return destroy_variables, info
        else:
            status, log_entry = scip_solve(local_branching_mip,  
                incumbent_solution = incumbent_solution,
                primal_bound = LNS_log[-1]['primal_bound'],
                prev_LNS_log = LNS_log[-1],
                scip_config = scip_solve_local_branching_config, 
                get_num_solutions = get_num_solutions)
            local_branching_solution = log_entry['best_primal_sol']
            
            
            LB_LP_relaxation_status, LB_LP_relaxation_log_entry = get_LP_relaxation_solution(local_branching_mip)
            #original_LP_relaxation_solution = original_LP_relaxation_log_entry['best_primal_sol']
            if LB_LP_relaxation_log_entry is None:
                original_LP_relaxation_solution = local_branching_solution
                LB_LP_relaxation_solution = local_branching_solution
            else:
                original_LP_relaxation_solution = LB_LP_relaxation_log_entry['best_primal_sol']
                LB_LP_relaxation_solution = LB_LP_relaxation_log_entry['best_primal_sol']
            
            

            tmp_observation = dict()
            tmp_observation["selected_by_LB"] = []
            tmp_observation["selected_by_LB_relax"] = []

            same_integer_value_inc_and_LB_LP = 0
            same_integer_value_LB_and_LB_LP = 0
            all_variables = local_branching_mip.getVars()
            for v in all_variables:
                if v.name in all_int_variables:
                    v_value = incumbent_solution.value(v)
                    v_LB_value = local_branching_solution.value(v)
                    v_LB_LP_value = LB_LP_relaxation_solution.value(v)

                    if isInteger(v_LB_LP_value):
                        if round(v_LB_LP_value) == round(v_LB_value):
                            same_integer_value_LB_and_LB_LP += 1
                        if round(v_LB_LP_value) == round(v_value):
                            same_integer_value_inc_and_LB_LP += 1


          
                    if heuristic.endswith("RELAXATION"):
                        print("---selecting using LP relaxation")
                        if isInteger(v_LB_LP_value):
                            if round(v_LB_LP_value) != round(v_value):
                                #destroy_variables.append(v.getIndex())
                                destroy_variables.append(v.name)
                                
                        else:
                            #destroy_variables.append(v.getIndex())
                            destroy_variables.append(v.name)
                            #tmp_observation.append((v.name, v_value, v_LB_value, v_LB_LP_value))
                    else: 
                        if round(v_LB_value) == round(v_value): continue
                        #destroy_variables.append(v.getIndex())
                        destroy_variables.append(v.name)                            
                        tmp_observation["selected_by_LB"].append((v.name, v_value, v_LB_value, v_LB_LP_value))

                    if isInteger(v_LB_LP_value):
                        if round(v_LB_LP_value) != round(v_value):
                            #destroy_variables.append(v.getIndex())
                            tmp_observation["selected_by_LB_relax"].append((v.name, v_value, v_LB_value, v_LB_LP_value))
                                
                    else:
                        #destroy_variables.append(v.getIndex())
                        #destroy_variables.append(v.name)
                        tmp_observation["selected_by_LB_relax"].append((v.name, v_value, v_LB_value, v_LB_LP_value))

            #print("---num same integer values LB and LB LP", same_integer_value_LB_and_LB_LP)
            #print("---num same integer values inc and LB LP", same_integer_value_inc_and_LB_LP)
            #print("num of variables selected by LB", len(destroy_variables), "with obj =", log_entry['primal_bound'], "runtime =", log_entry['solving_time'])
            #print("selected by LB =", tmp_observation["selected_by_LB"])
            #print("selected by LB relax=", tmp_observation["selected_by_LB_relax"])
            
        assert(heuristic.endswith("RELAXATION") or len(destroy_variables) <= neighborhood_size)
        info = dict()
        info["LB_primal_solution"] = log_entry["primal_bound"]
        info["LB_gap"] = status
        info["LB_LP_relaxation_solution"] = LB_LP_relaxation_solution
        if get_num_solutions > 1:
            info["multiple_solutions"] = copy.deepcopy(log_entry['var_index_to_values'])
            info["multiple_primal_bounds"] = copy.deepcopy(log_entry['primal_bounds'])
        #return random.sample(all_int_variables, neighborhood_size)
        return destroy_variables, info




def create_sub_mip(model, destroy_variables, incumbent_solution, local_branching_distance = None, mute = False):
    sub_mip = pyscipopt.Model(sourceModel = model, origcopy = True)
    num_free_variables = 0
    all_variables = sub_mip.getVars()

    if len(destroy_variables) > 0:
        if type(destroy_variables[0]) == type("string"):
            destroy_variables_name = copy.deepcopy(destroy_variables)
        else:
            destroy_variables_name = [v.name for v in model.getVars() if v.getIndex() in destroy_variables]
    else:
        destroy_variables_name = []
    
    variables_equal_one = []
    variables_equal_zero = []

    for v in all_variables:
        if not (v.name in destroy_variables_name):
            if not (v.vtype() in ["BINARY", "INTEGER"]): 
                continue
            fixed_value = incumbent_solution.value(v)
            
            sub_mip.chgVarLb(v, fixed_value)
            sub_mip.chgVarLbGlobal(v, fixed_value)
            sub_mip.chgVarUb(v, fixed_value)
            sub_mip.chgVarUbGlobal(v, fixed_value)
            #sub_mip.addCons(v >= fixed_value)
        else:
            assert v.vtype() in ["BINARY", "INTEGER"], "destroy variable %s not binary is instead %s"%(v.name, v.vtype())
            v_value = incumbent_solution.value(v)
            if round(v_value) == 1:
                variables_equal_one.append(v)
            else:
                variables_equal_zero.append(v)
            num_free_variables += 1
    if not mute:
        print("num_free_variables =", num_free_variables)
    if not (local_branching_distance is None):        
        if not mute:
            print("added local branching constraint in sub-mip")
        sub_mip.addCons(quicksum(v for v in variables_equal_zero) + quicksum( (1-v)  for v in variables_equal_one) <= local_branching_distance)
    return sub_mip


def get_bipartite_graph_representation(m, model): #m is a ecole mip model
    model = m.as_pyscipopt()
    bg = nx.DiGraph()
    #don't know why ecole.observation.NodeBipartite() won't work properly
    #implementing my own get_bipartite_graph_representation()
    var_name_to_index = dict()
    for var in model.getVars():
        var_name_to_index[var.name] = var.getIndex()
    
    num_var = model.getNVars()
    num_cons = model.getNConss()

    for i in range(num_var):
        bg.add_node(i)
        bg.nodes[i]['bipartite'] = 0
    for i in range(num_cons):
        bg.add_node(i+num_var)
        bg.nodes[i+num_var]['bipartite'] = 1

    all_constraints = model.getConss()
    for i, cons in enumerate(all_constraints):
        var_in_cons = model.getValsLinear(cons)
        for key, value in var_in_cons.items():
            var_index = var_name_to_index[key]
            bg.add_edge(var_index, i + num_var)


    all_variables = list(model.getVars())
    variables_to_nodes = dict()
    for i, feat_dict in bg.nodes(data = True):
        if i < len(all_variables):
            #assert(i ==  all_variables[i].getIndex())
            feat_dict.update({"scip_variable": all_variables[i].name})
            variables_to_nodes.update({all_variables[i].name: i})
        else:
            break
    for u, v in bg.edges():
        assert(bg.nodes[u]['bipartite'] == 0)
        assert(bg.nodes[v]['bipartite'] == 1)
    return bg, variables_to_nodes


def print_log_entry_to_file(save_to_file, LNS_log):
    with open(save_to_file, "wb") as f:
        for log_entry in LNS_log:
            log_entry.pop('best_primal_sol', None)
            log_entry.pop('best_primal_scip_sol', None)
        pickle.dump(LNS_log, f)



def extract_root_features(m, args, id):
    m.disable_presolve()
    observation, khalil = make_obs((bgo.BipartiteGraphObservations(), ecole.observation.Khalil2016(pseudo_candidates = True)), m, branching = False)
    extract_end_time = time.monotonic()
    branching_vars = np.array([i for i in range(observation.column_features.shape[0])])
    observation.add_khalil_features(khalil, branching_vars)
    return observation

 
def load_policy_from_checkpoint(args):

    policy = GNNPolicy(args.gnn_type)
    
    try:
        ckpt = torch.load(args.model, map_location=DEVICE)
        try_again = False
    except Exception as e:
        print("Checkpoint " + args.model + " not found, bailing out: " + str(e))
        sys.exit(1)
    

    policy.load_state_dict(ckpt.state_dict())
    #policy = policy.to(DEVICE)
    #model_version = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print("Loaded checkpoint")
    print(f"Will run evaluation on {DEVICE} device", flush=True)
    return policy


def get_perturbed_samples(args, model, destroy_variables, LNS_log, scip_solve_destroy_config, new_improvement, num_of_samples_to_generate, int_var):

    var_name_to_index = dict()
    fixed_variables = []
    for i, var in enumerate(int_var):
        var_name_to_index[var.name] = i
        if not (var.name in destroy_variables): 
            fixed_variables.append(var.name)
    
    primal_bound = LNS_log[-1]['primal_bound']

    objective_sense = model.getObjectiveSense()
    obj_sense = 1 if objective_sense == "minimize" else -1

    collected_samples = []
    primal_bounds = []
    negative_labels = []
    #embed()
    for num_of_replaced_variables in range(5, len(destroy_variables)-1, 5):
        no_negative_sample = 0
        for t in range(90):
            
            perturbed_destroy_variables = random.sample(destroy_variables, len(destroy_variables) - num_of_replaced_variables) + random.sample(fixed_variables, num_of_replaced_variables)


            sub_mip = create_sub_mip(model, perturbed_destroy_variables,  LNS_log[-1]['best_primal_sol'], mute = True)
            scip_solve_destroy_config = {
                'limits/time' : 240, # 240 for facilities 120 for the others
            }
    
            status, log_entry = scip_solve(sub_mip, incumbent_solution = LNS_log[-1]['best_primal_scip_sol'], 
                primal_bound = LNS_log[-1]['primal_bound'], scip_config = scip_solve_destroy_config, timer = None, prev_LNS_log = LNS_log[-1], mute = True)
            improvement = abs(primal_bound - log_entry["primal_bound"])
            improved = (obj_sense * (primal_bound - log_entry["primal_bound"]) > 1e-5)
            new_primal_bound = log_entry["primal_bound"]

            if (not improved) or (improvement < 0.05 * new_improvement):
                print(f"Found negative samples with {num_of_replaced_variables} replaced, primal bound = {primal_bound}, new primal bound = {new_primal_bound}")
                negative_sample = [0] * len(int_var)
                for var_name in perturbed_destroy_variables:
                    negative_sample[var_name_to_index[var_name]] = 1
                collected_samples.append(negative_sample)
                primal_bounds.append((log_entry["primal_bound"], primal_bound))
                negative_labels.append(improvement)
                no_negative_sample = 0
            else:
                no_negative_sample += 1
                if no_negative_sample >= 10: 
                    print(f"No negative samples for 10 consecutive samples with {num_of_replaced_variables} variables replaced")
                    break
                #print(f"This is not negative samples, primal bound = {primal_bound}, new primal bound = {new_primal_bound}")
            if len(collected_samples) == num_of_samples_to_generate:
                return collected_samples, primal_bounds, negative_labels
    return collected_samples, primal_bounds, negative_labels

    
    

def run_LNS(m, args, id):
    # m: ecole.scip.model, a mip model from ecole
    instance_id = m 
    if type(instance_id) == int:
        loader = InstanceLoader(presolve = args.presolve, competition_settings = False)
        for i, _m in enumerate(loader.load(args.problem_set)):
            if i == instance_id:
                m = _m
                break
    
    observation = None
    if (args.mode in ["COLLECT", "TEST_ML"]) or ("TEST_ML" in args.mode):
        print("Initializing Ecole for feature extraction...This might take a few minutes")
        observation = extract_root_features(m, args, id) 

    if type(instance_id) == int:
        loader = InstanceLoader(presolve = args.presolve, competition_settings = False)
        for i, _m in enumerate(loader.load(args.problem_set)):
            if i == instance_id:
                m = _m
                break


    model = m.as_pyscipopt()

        

    int_var = [v for v in model.getVars() if v.vtype() in ["BINARY", "INTEGER"]]
    num_int_var = len(int_var) # currently only considering binary variables

    args_neighborhood_size = args.neighborhood_size

    if args.neighborhood_size == 0:
        args.neighborhood_size = int(num_int_var * 0.2)

    collection_local_branching_runtime = COLLECT_SOLVE_TIME_LIMIT


    neighborhood_size = args.neighborhood_size
    destroy_heuristic = args.destroy_heuristic
    objective_sense = model.getObjectiveSense()
    obj_sense = 1 if objective_sense == "minimize" else -1


    print("Problem:",args.problem_set, instance_id)
    print("Using destroy heuristics:", destroy_heuristic)
    print("Neighborhood size:", neighborhood_size)
    print("Preprocessing...")

    if "VANILLA" in args.destroy_heuristic:
        scip_log = run_vanilla_scip(model, args)
        if args.save_log == 1:
            print_log_entry_to_file("tmp/log/%s_%s_nhsize%d.txt"%(id, destroy_heuristic, 0), scip_log)
        return


    bg ,variables_to_nodes = get_bipartite_graph_representation(m, model)
    
    if args.mode == "COLLECT" or args.collect_along_test == 1:
        db = BipartiteGraphDataset(args.data_loc + "%s_%d.db"%(args.problem_set, instance_id))
        #LB_relaxation_history = []
        #incumbent_history = []
    
    # find initial solution with SCIP
    scip_solve_init_config = {
        'limits/solutions' :10000,
        'limits/time' : 610 if "MIPLIB" in args.problem_set else args.init_time_limit,
    }
    #    scip_solve_init_config['limits/time'] = 300


    timer = None
    status, log_entry = scip_solve(model, scip_config = scip_solve_init_config, timer = timer,
        get_initial_solution = True, isMIPLIB = "MIPLIB" in args.problem_set)

    if log_entry is None:
        print('Did not find incumbent solution for MIP: skipping this instance; try a longer runtime')
        return
    else:
        print("initial solution obj =", log_entry['primal_bound'], "found in time", log_entry['solving_time'])
    log_entry['limits/time'] = scip_solve_init_config['limits/time']
    LNS_log = [log_entry]
    
    improved = True

    runtime_used = log_entry['solving_time']
    count_no_improve = 0
    print("solving steps limit =", args.num_solve_steps)


    # initialize incumbent_history with the initial solution
    if args.mode == "COLLECT" or "TEST_ML" in args.mode:
        incumbent_solution = []
        incumbent_history = []
        improvement_history = []
        LB_relaxation_history = []
        for var in int_var:        
            incumbent_solution.append(log_entry["var_index_to_value"][var.name])
        incumbent_history.append(incumbent_solution)

    if "TEST_ML" in args.mode:
        policy = load_policy_from_checkpoint(args)
        policy = policy.to(DEVICE)
        if "feat1" in args.mode:
            observation.column_features[:, 23:] = torch.zeros(observation.column_features.shape[0], observation.column_features.shape[1]-23)

        observation.column_features = torch.hstack((observation.column_features, torch.zeros(observation.column_features.shape[0], args.wind_size*3)))

    #embed()

    not_switched = True

    if args.ml_neighborhood_size == 0:
        args.ml_neighborhood_size = args.neighborhood_size

    for s in range(args.num_solve_steps):

        iteration_start_time = time.monotonic()
        


        incumbent_solution = LNS_log[-1]['best_primal_scip_sol']
        primal_bound = LNS_log[-1]['primal_bound']


        ML_info = None
        if "TEST_ML" in args.mode:
            ML_info = (policy, observation, incumbent_history, LB_relaxation_history)

        destroy_variables, info_destroy_heuristic = create_neighborhood_with_heuristic(model, LNS_log, 
            neighborhood_size = neighborhood_size, bipartite_graph = bg, variables_to_nodes = variables_to_nodes,
            heuristic = destroy_heuristic, num_samples = args.num_samples, eps_clip = args.eps_clip,
            ML_info = ML_info, original_neighborhood_size = args.ml_neighborhood_size, ## alert!!!
            get_num_solutions = 20 if args.mode == "COLLECT" else 1)
        #print("destroy variables =", destroy_variables)
        
        if "CONSTRAINTED_REPAIR" in args.destroy_heuristic:
            sub_mip = create_sub_mip(model, destroy_variables,  LNS_log[-1]['best_primal_sol'], local_branching_distance = args.neighborhood_size)
        else:
            sub_mip = create_sub_mip(model, destroy_variables,  LNS_log[-1]['best_primal_sol'])
        #print("sub mip created =", sub_mip)


        scip_solve_destroy_config = {
            'limits/time' : 120,
        }

        
        if args.mode == "COLLECT":
            scip_solve_destroy_config['limits/time'] = collection_local_branching_runtime
        status, log_entry = scip_solve(sub_mip, incumbent_solution = incumbent_solution, 
            primal_bound = LNS_log[-1]['primal_bound'], scip_config = scip_solve_destroy_config, timer = timer, prev_LNS_log = LNS_log[-1])        

        iteration_end_time = time.monotonic()
        log_entry["iteration_time"] = iteration_end_time - iteration_start_time
        log_entry["selection_time"] = log_entry["iteration_time"] - log_entry["solving_time"]
        if "ML" in args.mode and "ML" in destroy_heuristic:
            log_entry["ML_time"] = info_destroy_heuristic["ML_time"]
        else:
            log_entry["ML_time"] = 0
        log_entry["destroy_variables"] = destroy_variables
        log_entry["destroy_heuristic"] = destroy_heuristic

        log_entry["neighborhood_size"] = neighborhood_size
        if info_destroy_heuristic and "num_LB_relax_integer" in info_destroy_heuristic:
            log_entry["num_LB_relax_integer"] = info_destroy_heuristic["num_LB_relax_integer"]
        if info_destroy_heuristic and "num_ori_relax_integer" in info_destroy_heuristic:
            log_entry["num_ori_relax_integer"] = info_destroy_heuristic["num_ori_relax_integer"]
        if info_destroy_heuristic and "num_both_integer" in info_destroy_heuristic:
            log_entry["num_both_integer"] = info_destroy_heuristic["num_both_integer"]


        improvement = abs(primal_bound - log_entry["primal_bound"])
        improved = (obj_sense * (primal_bound - log_entry["primal_bound"]) > 1e-5)
        if improved == False:
            if round(neighborhood_size * args.adaptive) < round(num_int_var * 0.5):
                neighborhood_size = round(neighborhood_size * args.adaptive)
                count_no_improve = 0
            else:
                neighborhood_size = round(num_int_var * 0.5)
                count_no_improve += 1
                if "GREEDY" in destroy_heuristic:
                    destroy_heuristic = destroy_heuristic.replace("GREEDY", "SAMPLE")
        else:
            count_no_improve = 0
       

        LNS_log.append(log_entry)

        if "TEST_ML" in args.mode and improved == True:
            LB_relaxation_solution = []
            incumbent_solution = []
            relaxation_value = info_destroy_heuristic["LB_LP_relaxation_solution"]
            for var in int_var:
                LB_relaxation_solution.append(relaxation_value.value(var))
                incumbent_solution.append(log_entry["var_index_to_value"][var.name])
            LB_relaxation_history.append(LB_relaxation_solution)
            incumbent_history.append(incumbent_solution)
            improvement_history.append(improvement)
            
        if (args.mode == "COLLECT" and improved == True) or (args.collect_along_test == 1 and s % STEP_PER_COLLECT == 0):

            if args.collect_along_test == 1:
                destroy_variables, info_destroy_heuristic = create_neighborhood_with_heuristic(model, LNS_log[:-1], 
                    neighborhood_size = args.neighborhood_size if args.ml_neighborhood_size == 0 else args.ml_neighborhood_size, 
                    bipartite_graph = bg, variables_to_nodes = variables_to_nodes,
                    heuristic = "LOCAL_BRANCHING", num_samples = args.num_samples, eps_clip = args.eps_clip,
                    ML_info = ML_info, original_neighborhood_size = args.neighborhood_size if args.ml_neighborhood_size == 0 else args.ml_neighborhood_size, 
                    get_num_solutions =  20)
                print("destroy variables =", destroy_variables)

            assert info_destroy_heuristic is not None
            relaxation_value = info_destroy_heuristic["LB_LP_relaxation_solution"]
            assert relaxation_value is not None
            #candidate_scores = [0] * num_int_var
            candidate_scores = []
            LB_relaxation_solution = []
            incumbent_solution = []
            for var in int_var:
                if var.name in destroy_variables: 
                    candidate_scores.append(1)
                else:
                    candidate_scores.append(0)
                LB_relaxation_solution.append(relaxation_value.value(var))
                incumbent_solution.append(log_entry["var_index_to_value"][var.name])
            new_improvement = abs(primal_bound - info_destroy_heuristic["LB_primal_solution"])
            new_improved = (obj_sense * (primal_bound - info_destroy_heuristic["LB_primal_solution"]) > 1e-5)
            if args.mode == "COLLECT" or (args.collect_along_test == 1 and improved == False and new_improved == True):
                LB_relaxation_history.append(LB_relaxation_solution)
                incumbent_history.append(incumbent_solution)
                improvement_history.append(improvement)
                
            negative_samples, negative_info, negative_labels = get_perturbed_samples(args, model, destroy_variables, LNS_log[:-1], scip_solve_destroy_config, new_improvement, 90, int_var)
            
            candidates = [str(var.name) for var in int_var]
            candidate_choice = None
            info = dict()

            positive_samples = []
            positive_labels = []
            for i in range(len(info_destroy_heuristic["multiple_primal_bounds"])):
                positive_sample = [0] * len(int_var)
                
                for j, var in enumerate(int_var):
                    positive_sample[j] = info_destroy_heuristic["multiple_solutions"][var.name][i]
                positive_samples.append(positive_sample)
                obj_info = info_destroy_heuristic["multiple_primal_bounds"][i]
                positive_labels.append( abs(obj_info[0] - obj_info[1]))

            info["num_positive_samples"] = len(positive_samples)
            info["positive_samples"] = positive_samples
            info["positive_labels"] = positive_labels

            info["num_negative_samples"] = len(negative_samples)
            info["negative_samples"] = negative_samples
            info["negative_labels"] = negative_labels

            info["#iteration"] = s
            info["instance_id"] = id
            info["incumbent_history"] = incumbent_history
            info["LB_relaxation_history"] = LB_relaxation_history
            info["neighborhood_size"] = args.neighborhood_size
            info["LB_gap"] = info_destroy_heuristic["LB_gap"]
            info["primal_bound"] = log_entry["primal_bound"] if args.mode == "COLLECT" else info_destroy_heuristic["LB_primal_solution"]
            info["LB_runtime"] = log_entry["iteration_time"]
            candidate_scores = torch.LongTensor(np.array(candidate_scores, dtype = np.int32))


            graph = BipartiteGraph(observation.row_features, observation.edge_features.indices,
                               observation.edge_features.values, observation.column_features[:,:95],
                               candidates, candidate_choice, candidate_scores, info, 
                               iteration = i, instance_id = id, incumbent_history = incumbent_history, LB_relaxation_history = LB_relaxation_history, improvement_history = improvement_history,
                               neighborhood_size = neighborhood_size)
            
            if args.mode == "COLLECT" or (args.collect_along_test == 1 and new_improved == True):
                assert len(LB_relaxation_history) + 1 == len(incumbent_history)
                assert len(LB_relaxation_history) > 0
                rslt = db.add(graph)
                if not rslt:
                    print("Skipping duplicate datapoint")
                else:
                    print("Saving to database")

            if (improved == False and args.collect_along_test == 1 and new_improved == True):
                LB_relaxation_history.pop()
                incumbent_history.pop()
                improvement_history.pop()

        runtime_used += log_entry['iteration_time']       
        print("Finished LNS iteration #%d: obj_val = %.2f with time %.2f (total time used %.2f)" % (s, log_entry['primal_bound'], log_entry['iteration_time'], runtime_used))# -log_entry["ML_time"]))
        if runtime_used >= args.time_limit: break
        


    if args.save_log == 1:
        print_log_entry_to_file("tmp/log/%s_%s_nhsize%d.txt"%(id, args.destroy_heuristic, args_neighborhood_size), LNS_log)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type = int,
                        help="random seed")
    parser.add_argument("--problem-set", default="INDSET_test",
                        help="Problem set")
    parser.add_argument("--adaptive", default=1, type = float, 
                        help = "adaptive neighborhood size")
    parser.add_argument("--num-solve-steps", default=100000, type=int,
                        help="Number of LNS iterations")
    parser.add_argument("--neighborhood-size", default=100, type=int,
                        help="Upper bound on the neighborhood size")
    parser.add_argument("--ml-neighborhood-size", default=0, type=int,
                        help="ML neighborhood size")
    parser.add_argument("--eps-clip", default=0.05, type=float,
                        help="Clipping on LB_relax::RS probablity, will affect actual neighborhood size")
    parser.add_argument("--time-limit", default=3600, type=int,
                        help="Time limit per instance")
    parser.add_argument("--init-time-limit", default=10, type=int,
                        help="Initial solution time limit")
    parser.add_argument("--destroy-heuristic", default="RANDOM", type=str,
                        help="Destroy heuristics: RANDOM, LOCAL_BRANCHING, LOCAL_BRANCHING::RELAXATION, VARIABLE")
    parser.add_argument("--mode", default="TEST", type=str,
                        help="Solving mode: COLLECT, TEST, TEST_ML")
    parser.add_argument("--gnn-type", default="gat", type=str,
                        help="GNN type: gasse or gat")
    parser.add_argument("--model", default=None, type=str,
                        help="Path to the ML model")
    parser.add_argument("--num-samples", default=30, type=int,
                        help="Number of samples with sample-and-select-best heuristics")
    parser.add_argument("--save-log", default=0, type = int,
                        help="save log (1) or not (0)")
    parser.add_argument("--collect-along-test", default=0, type=int,
                        help="collect data along the trajectory generated by this one")
    parser.add_argument("--wind-size", default=3, type = int,
                        help="window size = the number of past incumbent features in features")
    parser.add_argument("--presolve", default=False, type = bool,
                        help="presolve or not")

    args = parser.parse_args()

    
    WIND_SIZE = args.wind_size

    if args.mode == "COLLECT" or args.collect_along_test == 1:
        if args.mode == "COLLECT":
            args.destroy_heuristic = "LOCAL_BRANCHING"
        
        try:
            os.mkdir("training_data")
        except OSError as error:
            print(error)    
        try:
            os.mkdir("training_data/" + args.problem_set)
        except OSError as error:
            print(error)    
        args.data_loc = "training_data/" + args.problem_set + "/" 


    print(args)

    random.seed(args.seed)
    

    loader = InstanceLoader(presolve = args.presolve, competition_settings = False) # default False if change presolve here, also 
    if args.destroy_heuristic == "VANILLA":
        args.adaptive = 1
    

    for i, m in enumerate(loader.load(args.problem_set)):
        model = m.as_pyscipopt()
        #all_int_variables = [v.getIndex() for v in model.getVars() if v.vtype()  in ["BINARY", "INTEGER"]]
        name = args.problem_set + str(i)
        if args.adaptive > 1:
            name = args.problem_set + str(round(args.adaptive*100)) + "_" + str(i)
        if args.mode == "COLLECT" or args.collect_along_test == 1:
            name += "COLLECT"
        run_LNS(i, args, id = name)
    print("Finish LNS for MIP solving")

