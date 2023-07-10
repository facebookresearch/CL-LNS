# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import sqlite3
from pathlib import Path
import hashlib
import string
import random
import functools
from collections import defaultdict


class EvaluationData():
    def __init__(self, db, read_only=False):
        p = Path(db)
        if not p.parent.exists():
            p.parent.mkdir(exist_ok=True)

        already_created = p.exists()
        assert already_created or not read_only

        uri = "file:" + db
        if read_only:
            uri += "?mode=ro"
        self.db = sqlite3.connect(uri, uri=True)
        self.cur = self.db.cursor()

        # Create table if needed
        if not already_created:
            self.cur.execute('''CREATE TABLE eval_data (instance_id string not null, model_version string not null, step_id integer not null, nb_nodes integer not null, timestamp float, primal float, dual float)''')
            self.cur.execute('''CREATE INDEX per_instance_id ON eval_data(instance_id)''') 
            self.cur.execute('''CREATE INDEX per_model_version ON eval_data(model_version)''') 
            self.db.commit()

        self.added_rows = 0

    def __del__(self):
        self.db.commit()
        self.db.close()

    @functools.lru_cache(maxsize=16)
    def _instance_to_key(self, model):
        letters = string.ascii_letters
        tmp_file = '/tmp/' + ''.join(random.choice(letters) for i in range(10)) + '.lp'
        model.writeProblem(tmp_file)
        with open(tmp_file, 'r') as f:
            problem = f.read()
            problem = problem.encode()
        key = hashlib.sha256(problem).hexdigest()
        return key

    def add(self, instance, model_version, step_id, primal, dual, nb_nodes, timestamp):
        instance_id = self._instance_to_key(instance)
        self.cur.execute(f"INSERT INTO eval_data VALUES (\'{instance_id}\', \'{model_version}\', {step_id}, {nb_nodes}, {timestamp}, {primal}, {dual})")
        self.added_rows += 1
        if self.added_rows % 1000 == 0:
            self.db.commit()

    def commit(self):
        self.db.commit()


class EvaluationDataMining():
    def __init__(self, db, models):
        self.db = EvaluationData(db, read_only=True)
        self.models = models

    def compute_metrics(self):
        model_filter = f"model_version == '{self.models[0]}' "
        for m in self.models[1:]:
            model_filter += f"OR model_version == '{m}' "
        query = f"SELECT DISTINCT instance_id FROM eval_data WHERE {model_filter}"
        #print(query)
        self.db.cur.execute(query)
        instances = self.db.cur.fetchall()
        #print(str(instances))

        integrals_over_time = defaultdict(lambda: [])
        integrals_over_nodes = defaultdict(lambda: [])
        nb_nodes = defaultdict(lambda: [])

        for instance in instances:
            instance_id = instance[0]
            max_nb_nodes = 1e100
            for version in self.models:
                query = f"SELECT MAX(nb_nodes) FROM eval_data WHERE instance_id == '{instance_id}' AND model_version == '{version}'" 
                #print(query)
                self.db.cur.execute(query)
                num_nodes = self.db.cur.fetchone()
                #print(str(num_nodes))
                max_nb_nodes = min(max_nb_nodes, int(num_nodes[0]))

            for version in self.models:
                #print(version)
                nb_nodes[version].append(max_nb_nodes)

                integral_over_time = 0
                integral_over_nodes = 0
                query = f"SELECT nb_nodes, dual, timestamp FROM eval_data WHERE instance_id == '{instance_id}' AND model_version == '{version}' AND nb_nodes <= {max_nb_nodes} ORDER BY nb_nodes ASC" 
                #print(query)
                first = True
                last_dual = 0
                last_nb_nodes = 0
                last_timestamp = 0
                for rslt in self.db.cur.execute(query):
                    #print("ORDERED RSLT:" + str(rslt))
                    if not first:
                        integral_over_time += last_dual * (float(rslt[2]) - last_timestamp)
                        integral_over_nodes += last_dual * (int(rslt[0]) - last_nb_nodes)
                    first = False
                    last_dual = float(rslt[1])
                    last_nb_nodes = int(rslt[0])
                    last_timestamp = float(rslt[2])
                if last_nb_nodes < max_nb_nodes:
                    integral_over_nodes += last_dual * (max_nb_nodes - last_nb_nodes)

                integrals_over_time[version].append(integral_over_time)
                integrals_over_nodes[version].append(integral_over_nodes)
                
        return (nb_nodes, integrals_over_nodes, integrals_over_time)

    def draw_in_tensorboard(self):
        pass
