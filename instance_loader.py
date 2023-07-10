# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import os.path
import tarfile
import zipfile
import ecole
import geco
import geco.generator
import glob
import re
import json
import pyscipopt
import hashlib
import string
import random
import pyscipopt


class InstanceLoader:

    LOCAL_INSTANCE = {
        "INDSET_test": "instances/INDSET_ER_6000/instance_ER4_*.cip", 
        "INDSET_train": "instances/INDSET_ER_6000/train/train_instance_ER4_*.cip", 
    }

    ECOLE = {
        # The settings are taken from the Gasse paper
        # (https://papers.nips.cc/paper/2019/file/d14c2267d848abeb81fd590f371d39bd-Paper.pdf)
        "SET_COVER_EASY": ecole.instance.SetCoverGenerator(n_rows=500, n_cols=500, density=0.05),
        "SET_COVER_MEDIUM": ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
        "SET_COVER_HARD": ecole.instance.SetCoverGenerator(n_rows=500, n_cols=2000, density=0.05),
        "INDEPENDENT_SET_EASY": ecole.instance.IndependentSetGenerator(n_nodes=500),
        "INDEPENDENT_SET_MEDIUM": ecole.instance.IndependentSetGenerator(n_nodes=1000),
        "INDEPENDENT_SET_HARD": ecole.instance.IndependentSetGenerator(n_nodes=1500),
        "AUCTION_EASY": ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500),
        "AUCTION_MEDIUM": ecole.instance.CombinatorialAuctionGenerator(n_items=200, n_bids=1000),
        "AUCTION_HARD": ecole.instance.CombinatorialAuctionGenerator(n_items=300, n_bids=1500),
        "FACILITY_EASY": ecole.instance.CapacitatedFacilityLocationGenerator(n_facilities=100, n_customers=100),
        "FACILITY_MEDIUM": ecole.instance.CapacitatedFacilityLocationGenerator(n_facilities=100, n_customers=200),
        "FACILITY_HARD": ecole.instance.CapacitatedFacilityLocationGenerator(n_facilities=100, n_customers=400),
    }

    GECO = {
        # Instances from the GeCO generator
        "KNAPSACK_UC": lambda seed: geco.mips.knapsack.pisinger.uncorrelated(n=1974, c=2864, seed=seed),
        "KNAPSACK_WC": lambda seed: geco.mips.knapsack.pisinger.weakly_correlated(n=1974, c=2864, seed=seed),
        "KNAPSACK_SC": lambda seed: geco.mips.knapsack.pisinger.strongly_correlated(n=1974, c=2864, seed=seed),
        "KNAPSACK_ISC": lambda seed: geco.mips.knapsack.pisinger.inverse_strongly_correlated(n=1974, c=2864, seed=seed),
        "KNAPSACK_ASC": lambda seed: geco.mips.knapsack.pisinger.almost_strongly_correlated(n=1974, c=2864, seed=seed),
        "KNAPSACK_SUBSET_SUM": lambda seed: geco.mips.knapsack.pisinger.subset_sum(n=1974, c=2864, seed=seed),
        "KNAPSACK_UWSW": lambda seed: geco.mips.knapsack.pisinger.uncorrelated_with_similar_weights(n=1974, c=2864, seed=seed),
        #"KNAPSACK_SPANNER": lambda seed: geco.mips.knapsack.pisinger.spanner(v=345, m=2, n=995, distribution=**uncorrelated_distribution(), capacity=1720, seed=seed),
        "KNAPSACK_PROFIT_CEILING": lambda seed: geco.mips.knapsack.pisinger.profit_ceiling(n=2974, c=1864, d=1.5, seed=seed),
        "KNAPSACK_CIRCLE": lambda seed: geco.mips.knapsack.pisinger.circle(n=1974, c=2864, seed=seed),
        "KNAPSACK_MSC": lambda seed: geco.mips.knapsack.pisinger.multiple_strongly_correlated(n=1974, c=2864, k1=1, k2=2, d=3, seed=seed),
        "KNAPSACK_YANG": lambda seed: geco.mips.knapsack.yang.yang_instance(n=2368, seed=seed),
        "SCHEDULING_HEINZ": lambda seed: geco.mips.scheduling.heinz.heinz_instance(number_of_facilities=43, number_of_tasks=114, seed=seed),
        "SCHEDULING_HOOKER": lambda seed: geco.mips.scheduling.hooker.hooker_instance(number_of_facilities=23, number_of_tasks=73, time_steps=25, seed=seed),
        "SET_PACKING": lambda seed: geco.mips.set_packing.yang.yang_instance(m=734, seed=seed),
        "SET_COVER_SUN": lambda seed: geco.mips.set_cover.sun.sun_instance(n=1927, m=1467, seed=seed),
        "SET_COVER_YANG": lambda seed: geco.mips.set_cover.yang.yang_instance(m=1513, seed=seed),
        #"PRODUCTION_PLANNING": lambda seed: geco.mips.production_planning.tang.tang_instance(T=5, seed=seed),
        "MAX_INDEPENDENT_SET": lambda seed: geco.mips.independent_set.barabasi_albert.barabasi_albert_instance(m=10, n=100, seed=seed),
        "MAX_CUT": lambda seed: geco.mips.max_cut.tang.tang_instance(n=593, m=684, seed=seed),
        "PACKING": lambda seed: geco.mips.packing.tang.tang_instance(n=321, m=428, seed=seed),
        #"GRAPH_COLORING": lambda seed: geco.mips.graph_coloring.generic.assigment(seed=seed),
        #"FACILITY_CORNUEJOLS": lambda seed: geco.mips.facility_location.cornuejols.cornuejols_instance(n_customers=385, n_facilities=683, ratio=.95, seed=seed),
    }

    GECO_MIPLIB = {
        "MIPLIB_BENCHMARK": geco.mips.loading.miplib.benchmark_instances(),
        "MIPLIB_EASY": geco.mips.loading.miplib.easy_instances(),
        "MIPLIB_HARD": geco.mips.loading.miplib.hard_instances(),
        "MIPLIB_OPEN": geco.mips.loading.miplib.open_instances(),
        #"ORLIB": geco.mips.loading.orlib_load_instance(),
    }

    

    DATASETS = {
        "BCOL": "mip_BCOL-CLS.tar.gz",
        "CORLAT": "mip_COR-LAT.tar.gz",
        "MIPLIB": "collection.zip",
        "MIPLIB_FILTERED": "collection.zip",
        "RCW2": "mip_RCW2.tar.gz",
        "Regions200": "mip_Regions200.tar.gz",
    }

    COMPETITION = {
        "ANONYMOUS": "anonymous.tar.gz",
        "ITEM_PLACEMENT": "item_placement.tar.gz",
        "LOAD_BALANCING": "load_balancing.tar.gz",
    }

    #DFLT_TMP_FILE_LOC = "/tmp/" + str(os.geteuid()) + "/"
    #DFLT_TMP_FILE_LOC = "/tmp/" + str(2575) + "/"
    DFLT_TMP_FILE_LOC = ""

    def __init__(self, dataset_loc = "", tmp_file_loc = DFLT_TMP_FILE_LOC, mode="*", repeat=False, presolve=True, competition_settings=True, load_metadata=False, shard=0, shard_count=0, pprocess = False):
        dataset_loc = os.path.expanduser(dataset_loc)
        #try:
        #    os.mkdir(tmp_file_loc)
        #except FileExistsError:
        #    pass
        self.dataset_loc = dataset_loc
        self.tmp_file_loc = tmp_file_loc
        self.mode = mode
        self.repeat = repeat
        self.presolve = presolve
        self.competition_settings=competition_settings
        self.load_metadata = load_metadata
        self.shard = shard
        self.shard_count = shard_count
        self.filtered_instances = []
        self.post_process = pprocess
        assert shard >= 0
        assert (shard < shard_count or shard_count == 0)
        
    @staticmethod
    def hash_model(model):
        letters = string.ascii_letters
        tmp_file = '/tmp/' + ''.join(random.choice(letters) for i in range(10)) + '.lp'
        model.as_pyscipopt().writeProblem(tmp_file)
        with open(tmp_file, 'r') as f:
            problem = f.read()
            problem = problem.encode()
        key = hashlib.blake2s(problem, digest_size=4).hexdigest()
        return key

    def load(self, dataset_name):
        if not self.repeat:
            for m in self.load_datasets(dataset_name):
                yield m
        else:
            while True:
                for m in self.load_datasets(dataset_name):
                    yield m

    def load_datasets(self, dataset_name):
        datasets = dataset_name.split('+')
        for d in datasets:
            for m in self.load_once(d):
                yield m
        print(self.filtered_instances)

    def load_once(self, dataset_name):
        if dataset_name in self.ECOLE:
            return self.load_ecole(dataset_name)
        elif dataset_name in self.GECO:
            return self.load_geco(dataset_name)
        elif dataset_name in self.GECO_MIPLIB:
            return self.load_geco_miplib(dataset_name)
        elif dataset_name in self.LOCAL_INSTANCE:
            return self.load_local_instance(dataset_name)
        elif dataset_name in self.COMPETITION:
            return self.load_competition(dataset_name)

        filename = self.DATASETS[dataset_name]
        local_version = os.path.join(self.dataset_loc, filename)
        if zipfile.is_zipfile(local_version):
            return self.load_zip(local_version)
        elif tarfile.is_tarfile(local_version):
            filter = re.compile(".+mps|.+lp")
            return self.load_tar(local_version, filter=filter)
        else:
            assert False

    def setup(self, ecole_model):
        if self.competition_settings:
            #print("disabling")
            # disable SCIP heuristics and restarts
            scip_model = ecole_model.as_pyscipopt()
            scip_model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
            ecole_model.set_params({
                'estimation/restarts/restartpolicy': 'n',
            })

    def preprocess(self, ecole_model):
        self.setup(ecole_model)
        #print(self.presolve)
        if self.presolve:
            return ## NEVER presolve 
            print("presolving mip")
            ecole_model.presolve()

    def load_zip(self, local_version):
        
        with zipfile.ZipFile(local_version) as z:
            if self.shard_count:
                files = z.namelist()
                shard = files[slice(self.shard, None, self.shard_count)]
            else:
                shard = z.namelist()

            for member in shard:
                f = z.extract(member, path=self.tmp_file_loc)
                instance = os.path.join(self.tmp_file_loc, member)
                #yield instance #bad coding :( this is just for loading MIPLIB instance
                continue
                #ecole_model = ecole.scip.Model.from_file(instance)
                temp_model = pyscipopt.Model()
                print(instance)
                temp_model.readProblem(instance)
                if temp_model.getNVars() != temp_model.getNBinVars():
                    continue
                #self.filtered_instances.append(member)
                #print(self.filtered_instances)
                ecole_model = ecole.scip.Model.from_pyscipopt(temp_model)
                self.preprocess(ecole_model)
                if not ecole_model.is_solved:
                    yield ecole_model

    def load_tar(self, local_version, filter=None, presolved=False):
        with tarfile.open(local_version) as t:
            members = t.getmembers()
            if self.shard:
                members = members[slice(self.shard, None, self.shard_count)]

            for member in members:
                if not member.isfile():
                    continue
                if filter and not filter.match(member.name):
                    continue
                f = t.extract(member, path=self.tmp_file_loc)
                instance = os.path.join(self.tmp_file_loc, member.name)
                
                #ecole_model = ecole.scip.Model.from_file(instance)
                temp_model = pyscipopt.Model()
                temp_model.readProblem(instance)
                ecole_model = ecole.scip.Model.from_pyscipopt(temp_model)
                self.setup(ecole_model)
                if self.presolve and not presolved:
                    ecole_model.presolve()
                if ecole_model.is_solved:
                    continue

                if not self.load_metadata:
                    yield ecole_model
                else:
                    metadata_loc = member.name.replace('mps', 'json')
                    f = t.extract(metadata_loc, path=self.tmp_file_loc)
                    raw_metadata = os.path.join(self.tmp_file_loc, metadata_loc)
                    with open(raw_metadata) as f:
                        metadata = json.load(f)
                    yield (ecole_model, metadata)

    def load_ecole(self, instance_type):
        instances = self.ECOLE[instance_type]
        instances.seed(self.shard)
        for ecole_model in instances:
            self.preprocess(ecole_model)
            if not ecole_model.is_solved:
                yield ecole_model

    def load_geco(self, instance_type):
        generator = self.GECO[instance_type]
        for m in geco.generator.generate(generator, seed=self.shard):
            ecole_model = ecole.scip.Model.from_pyscipopt(m)
            self.preprocess(ecole_model)
            if not ecole_model.is_solved:
                yield ecole_model

    def load_geco_miplib(self, instance_type):
        # Sharding not supported yet
        assert self.shard_count == 0
        instances = self.GECO_MIPLIB[instance_type]
        for m in instances:
            ecole_model = ecole.scip.Model.from_pyscipopt(m)
            self.preprocess(ecole_model)
            if not ecole_model.is_solved:
                yield ecole_model

    def load_local_instance(self, instance_type):
        # Sharding not supported yet
        assert self.shard_count == 0
        dir = self.LOCAL_INSTANCE[instance_type]
        for instance in glob.glob(dir):
            print(instance)
            
            temp_model = pyscipopt.Model()
            if self.post_process:
                yield temp_model
                continue
            temp_model.readProblem(instance)
            ecole_model = ecole.scip.Model.from_pyscipopt(temp_model)
            
            #ecole_model = ecole.scip.Model.from_file(instance)
            self.preprocess(ecole_model)
            #self.setup(ecole_model)
            yield ecole_model

    def load_competition(self, instance_type):
        filename = self.COMPETITION[instance_type]
        local_version = os.path.join(self.dataset_loc, filename)
        filter = re.compile(".+mps")
        return self.load_tar(local_version, filter=filter, presolved=True)


if __name__ == '__main__':
    loader = InstanceLoader()
    for m in loader.load("KNAPSACK_YANG"):                                                                                                                                                                                                                 
        print(str(m))
        break
