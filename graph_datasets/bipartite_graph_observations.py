# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import ecole
import torch
import numpy as np
import math
import time

def augment_variable_features_with_dynamic_ones(batch, args, initial_solution = {}):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #DEVICE = 'cpu'

    nh_size_threshold = dict() # filter out training data below certain neighborhood size threshold

    window_size = args.window_size 
    # to add features of the last $window_size improving solutions in LNS
    # each window contains 1. whether we have the solution 2. incumbent values 3. LB relax values
    dynamic_feature_size = window_size * 3
    static_feature_size = batch.variable_features.shape[-1]
    dynamic_features = torch.zeros((batch.variable_features.shape[0], window_size * 3), dtype = torch.float32)

    if "feat1" in args.experiment: #feat1: no Khalil's feature and no LB relax feature
        batch.variable_features[:,23:] = torch.zeros(batch.variable_features.shape[0], batch.variable_features.shape[1] - 23)


    assert len(batch.incumbent_history) == len(batch.LB_relaxation_history)

    tot_variables = 0
    batch_weight = []
    batch_n_candidates = []
    #embed()
    for i in range(len(batch.LB_relaxation_history)):

        
        #pop the incumbent solution
        batch.incumbent_history[i].pop()

    
        assert len(batch.incumbent_history[i]) == len(batch.LB_relaxation_history[i])
        number_of_history_added = 0
        number_of_variables = len(batch.LB_relaxation_history[i][0])

        total_candidates = torch.sum(batch.candidate_scores[tot_variables:tot_variables+number_of_variables])
        batch_n_candidates.append(total_candidates)
        #print(total_candidates)
        if args.problem in nh_size_threshold and  total_candidates<nh_size_threshold[args.problem]:
            batch_weight.append(0)
            #print("============No positive labels=============")
        else:
            batch_weight.append(1)

        for j in reversed(range(len(batch.LB_relaxation_history[i]))):
            
            assert number_of_variables == len(batch.incumbent_history[i][j])
            assert number_of_variables == len(batch.LB_relaxation_history[i][j])
            dynamic_features[tot_variables:tot_variables+number_of_variables, number_of_history_added*3]  = torch.FloatTensor([1]*number_of_variables)
            dynamic_features[tot_variables:tot_variables+number_of_variables, number_of_history_added*3+1] = torch.FloatTensor(batch.incumbent_history[i][j])
            if "feat1" in args.experiment or "feat2" in args.experiment:
                dynamic_features[tot_variables:tot_variables+number_of_variables, number_of_history_added*3+2] = torch.zeros(len(batch.LB_relaxation_history[i][j]))
            else:   
                dynamic_features[tot_variables:tot_variables+number_of_variables, number_of_history_added*3+2] = torch.FloatTensor(batch.LB_relaxation_history[i][j])

            number_of_history_added += 1
            if number_of_history_added == window_size:
                break
        #print(number_of_history_added)
        tot_variables += number_of_variables
        #embed()
    
    assert tot_variables == batch.variable_features.shape[0]
    dynamic_features = dynamic_features.to(DEVICE)

    #this implementation is bad, again due to a bug during data collection
    if batch.variable_features.shape[-1] == 104:
        batch.variable_features[:,-9:] = dynamic_features
    else:
        all_features = torch.hstack((batch.variable_features, dynamic_features))
        batch.variable_features = all_features
    #print("batch valid sample %d / %d"% (sum(batch_weight), len(batch_weight)))
    batch_weight = torch.tensor(batch_weight)
    #embed()
    batch.batch_weight = batch_weight.to(DEVICE)
    return batch

        

class MilpEdgeFeatures():
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values

class MilpProblemObservation():
    def __init__(self, column_features, row_features, edge_features):
        self.column_features = column_features
        self.row_features = row_features
        self.edge_features = edge_features

    def add_LB_relaxation_value(self, LB_relaxation_value):
        pass


    def add_khalil_features(self, khalil, action_set):
        # Validate and cleanup the Khalil features
        assert khalil.features.shape[0] == len(action_set)
        khalil_features = np.nan_to_num(khalil.features.astype(np.float32),
                                        posinf=1e6,
                                        neginf=-1e6)

        # Concatenate the khalil features with the existing features
        column_feature_size = self.column_features.shape[-1]
        khalil_feature_size = khalil_features.shape[-1]
        total_feature_size = column_feature_size + khalil_feature_size
        col_features = torch.zeros(
            (self.column_features.shape[0], total_feature_size),
            dtype=torch.float32)
        col_features[:, :column_feature_size] = self.column_features
        col_features[action_set.astype(np.int32),
                     column_feature_size:] = torch.from_numpy(khalil_features)
        self.column_features = col_features

    def check_features(self):
        assert not torch.any(torch.isinf(self.row_features))
        assert not torch.any(torch.isinf(self.column_features))
        assert not torch.any(torch.isinf(self.edge_features.indices))
        assert not torch.any(torch.isinf(self.edge_features.values))
        assert not torch.any(torch.isnan(self.row_features))
        assert not torch.any(torch.isnan(self.column_features))
        assert not torch.any(torch.isnan(self.edge_features.indices))
        assert not torch.any(torch.isnan(self.edge_features.values))


# Completement the basic Gasse features with some of our own:
# Lower and upper bound for each variable
# Coefficients associated with each variable in the objective function
# Lower and upper bound for each constraint
class BipartiteGraphObservations(ecole.observation.NodeBipartite):
    def __init__(self, check_for_nans=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_for_nans = check_for_nans
        self.num_calls = 0
        self.feature_extraction_time = 0
        self.feature_cleanup_time = 0
        self.extra_col_feature_extraction_time = 0
        self.extra_row_feature_extraction_time = 0
        self.feature_normalization_time = 0
        self.feature_merge_time = 0
        self.total_time = 0

    def before_reset(self, model):
        super().before_reset(model)
        #model.write_problem("/tmp/pb.lp")

    def extract(self, model, done):
        if done:
            return None

        start = time.monotonic()

        # Extract the Gasse features
        base_obs = super().extract(model, done)

        stop = time.monotonic()
        self.feature_extraction_time += stop - start

        scip_model = model.as_pyscipopt()

        #sense = scip_model.getObjectiveSense()
        #assert(sense == "minimize")

        # Delete the incubent column features. They are always NaN when the scip heuristics are turned off.
        print(base_obs.variable_features.shape)
        base_obs.variable_features = np.delete(base_obs.variable_features, 14, axis=1)
        base_obs.variable_features = np.delete(base_obs.variable_features, 13, axis=1)
        stop = time.monotonic()
        self.feature_cleanup_time += stop - start

        assert not np.isinf(base_obs.variable_features.astype(np.float32)).any()

        #total_col_features = 3 + base_obs.column_features.shape[-1]
        extra_col_features = np.empty((base_obs.variable_features.shape[0], 6), dtype=np.float32)
        cols = scip_model.getLPColsData()
        assert(len(cols) == base_obs.variable_features.shape[0])
        vars = scip_model.getVars(transformed=True)
        assert(len(vars) == base_obs.variable_features.shape[0])
        for i in range(base_obs.variable_features.shape[0]):
            col = cols[i]
            assert i == col.getLPPos()
            #print("BASIS = " + str(col.getBasisStatus()))
            #print("POS = " + str(col.getLPPos()))
            #print("POVArS = " + str(col.getVar()))
            #print(str(base_obs.column_features[i]), flush=True)
            #print(str(base_obs.column_features[i][6]))

            #print("LB = " + str(col.getLb()))
            #print("UB = " + str(col.getUb()))
            extra_col_features[i, 0] = col.getLb()
            extra_col_features[i, 1] = col.getUb()

            var = vars[i]
            assert i == var.getCol().getLPPos()
            assert var.ptr() == col.getVar().ptr()
            extra_col_features[i, 2] = var.getLbGlobal()
            extra_col_features[i, 3] = var.getUbGlobal()
            extra_col_features[i, 4] = var.getObj()
            assert var.getLPSol() == col.getPrimsol()
            extra_col_features[i, 5] = var.getLPSol()

            #print("OBJ = " + str(var.getObj()))
            #print("LP SOL = " + str(var.getLPSol()))

            assert col.getLb() == var.getLbLocal()
            assert col.getUb() == var.getUbLocal()

            #var_map[var.getIndex()] = var

        stop = time.monotonic()
        self.extra_col_feature_extraction_time += stop - start

        assert not np.isinf(extra_col_features).any()


        #extra_col_features[:, 3:] = base_obs.column_features
        #base_obs.column_features = torch.from_numpy(extra_col_features)

        #total_row_features = 3 + base_obs.row_features.shape[-1]
        extra_row_features = np.empty((base_obs.row_features.shape[0], 5), dtype=np.float32)
        rows = scip_model.getLPRowsData()
        assert len(rows) <= base_obs.row_features.shape[0]
        ecole_cns_id = 0
        for i in range(len(rows)):
            row = rows[i]
            assert i == row.getLPPos()

            # If a constraint has both a lhs and a rhs, ecole will create 2 constraints under the hood
            lhs_set = not scip_model.isInfinity(abs(row.getLhs()))
            rhs_set = not scip_model.isInfinity(abs(row.getRhs()))
            assert lhs_set or rhs_set
            if lhs_set:
                cns = -row.getLhs()
                extra_row_features[ecole_cns_id, 0] = cns
                extra_row_features[ecole_cns_id, 1] = math.copysign(1, cns)
                extra_row_features[ecole_cns_id, 2] = row.getConstant()
                extra_row_features[ecole_cns_id, 3] = row.getOrigintype()
                extra_row_features[ecole_cns_id, 4] = row.isIntegral()
                ecole_cns_id += 1
            if rhs_set:
                cns = row.getRhs()
                extra_row_features[ecole_cns_id, 0] = cns
                extra_row_features[ecole_cns_id, 1] = math.copysign(1, cns)
                extra_row_features[ecole_cns_id, 2] = row.getConstant()
                extra_row_features[ecole_cns_id, 3] = row.getOrigintype()
                extra_row_features[ecole_cns_id, 4] = row.isIntegral()
                ecole_cns_id += 1
            #extra_row_features[i, 0] = -row.getLhs()
            #extra_row_features[i, 1] = row.getRhs()
            #extra_row_features[i, 1] = row.getConstant()

            #lhs = row.getLhs()
            #print("- LHS = " + str(lhs))
            #rhs = row.getRhs()
            #print("- RHS = " + str(rhs))
            #cons = row.getConstant()
            #print("- CONS = " + str(cons))

            #print("- POS: " + str(pos))
            #val = row.getVals()
            #print("- VALS = " + str(val))
            #for col in row.getCols():
            #    print("- COLS: " + str(cols))
            #row = scip_model.getTransformedCons(row)
            #lhs = row.getLhs()
            #print("- LHS = " + str(lhs))
            #rhs = row.getRhs()
            #print("- RHS = " + str(rhs))
            #cons = row.getConstant()
            #print("- CONS = " + str(cons))
            #pos = row.getLPPos()
            #print("- POS: " + str(pos))
            #val = row.getVals()
            #print("- VALS = " + str(val))
            #node_id += 1

        assert ecole_cns_id == base_obs.row_features.shape[0]

        stop = time.monotonic()
        self.extra_row_feature_extraction_time += stop - start


        #extra_row_features[:, 3:] = base_obs.row_features
        #base_obs.row_features = torch.from_numpy(extra_row_features)

        #vars = scip_model.getVars(transformed=False)
        #for var in vars:
        #    print("VAR = " + str(var) + ": " + str(var.getCol()) + " " + str(var.getObj()))

        #vars = scip_model.getVars(transformed=True)
        #i = 0
        #for var in vars:
        #    print("TRANSFORMED VAR = " + str(var) + ": " + str(var.getCol()) + " " + str(var.getObj()))
        #    assert i == var.getCol().getLPPos()
        #    i += 1
        #    #print("LB = " + str(var.getLbOriginal()) + "/" + str(var.getLbLocal()) + "/" + str(var.getLbGlobal()))
        #    #print("UB = " + str(var.getUbOriginal()) + "/" + str(var.getUbLocal()) + "/" + str(var.getUbGlobal()))

        #conss = scip_model.getConss()
        #assert(len(conss) == base_obs.row_features.shape[0])
        #for cons in conss:
        #    print(str(cons))

        #obj = scip_model.getObjective()
        #print("OBJ = " + str(obj))

        #params = model.get_params()
        #print("PARAMS: " + str(params))

        #lp_columns = model.lp_columns()
        #print("LP_COLUMNS " + str(lp_columns))

        #lp_rows = model.lp_rows()
        #print("LP_ROWS " + str(lp_rows))

        #constraints = scip_model.getConss()
        #print("CNS: " + str(constraints))

        #constraints = scip_model.getNConss()
        #print("NCNS: " + str(len(cols)) + " vs " + str(base_obs.column_features.shape[0]), flush=True)
        #print("NROWS: " + str(len(rows)) + " vs " + str(base_obs.row_features.shape[0]), flush=True)




        #print("CNS: " + str(base_obs.row_features))
        #print("EDGES: " + str(base_obs.edge_features.indices))
        #print("EDG VALS: " + str(base_obs.edge_features.values))
        #print("VARS: " + str(base_obs.column_features), flush=True)

        #print("WHOLE FEATURIZATION" + str(base_obs))


        ##############
        # MORE STUFF
        #scip_model.getRowLPActivity()


        # Normalize the objective features
        factor = 1.0 / np.max(np.absolute(extra_col_features[:, 4]))
        extra_col_features[:, 4] *= factor

        # Store both normalized and unormalized constraints
        new_edge_values = np.tile(base_obs.edge_features.values.astype(np.float32).reshape(-1, 1), (1, 2))
        #assert not np.any(np.isnan(new_edge_values))

        cns_id = base_obs.edge_features.indices[0, :]
        cns = extra_row_features[cns_id, 0]
        div = np.maximum(1e-6, np.abs(cns))
        new_edge_values[:, 1] /= div
        #assert not np.any(np.isnan(new_edge_values))

        stop = time.monotonic()
        self.feature_normalization_time += stop - start

        column_features = torch.from_numpy(np.concatenate([extra_col_features, base_obs.variable_features.astype(np.float32)], axis=1))
        row_features = torch.from_numpy(np.concatenate([extra_row_features, base_obs.row_features.astype(np.float32)], axis=1))
        edge_features = MilpEdgeFeatures(torch.from_numpy(base_obs.edge_features.indices.astype(np.int64)), torch.from_numpy(new_edge_values))
        obs = MilpProblemObservation(column_features, row_features, edge_features)

        stop = time.monotonic()
        self.feature_merge_time += stop - start

        if self.check_for_nans:
            assert not torch.any(torch.isnan(obs.row_features))
            assert not torch.any(torch.isnan(obs.column_features))
            assert not torch.any(torch.isnan(obs.edge_features.indices))
            assert not torch.any(torch.isnan(obs.edge_features.values))
            assert not torch.any(torch.isinf(obs.row_features))
            assert not torch.any(torch.isinf(obs.column_features))
            assert not torch.any(torch.isinf(obs.edge_features.indices))
            assert not torch.any(torch.isinf(obs.edge_features.values))

        stop = time.monotonic()
        self.total_time += stop - start
        self.num_calls += 1
        
        '''
        print("feature_extraction_time", self.feature_extraction_time)
        print("feature_cleanup_time", self.feature_cleanup_time)
        print("extra_col_feature_extraction_time", self.extra_col_feature_extraction_time)
        print("extra_row_feature_extraction_time", self.extra_row_feature_extraction_time)
        print("feature_normalization_time", self.feature_normalization_time)
        print("feature_merge_time", self.feature_merge_time)
        print("total_time", self.total_time)
        '''
        return obs

    def timings(self):
        if self.num_calls == 0:
            return ""

        timing = f"observation time = {self.feature_extraction_time/self.num_calls: >.4f} {self.feature_cleanup_time/self.num_calls: >.4f} {self.extra_col_feature_extraction_time/self.num_calls: >.4f} {self.extra_row_feature_extraction_time/self.num_calls: >.4f} {self.feature_normalization_time/self.num_calls: >.4f} {self.feature_merge_time/self.num_calls: >.4f} {self.total_time/self.num_calls: >.4f}"
        return timing
