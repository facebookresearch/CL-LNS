# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import sqlite3
import pickle
from pathlib import Path
import hashlib
import string
import random
import base64
import functools


class SolvedMilpDataset():
    """
    This class stores the best solution found for a collection of milp instances.
    """
    def __init__(self, sample_db, read_only=False, best_solution_only=True):
        self.best_solution_only = best_solution_only
        if best_solution_only:
            self.sql_insert = "REPLACE"
        else:
            self.sql_insert = "INSERT"

        p = Path(sample_db)
        if not p.parent.exists():
            p.parent.mkdir(exist_ok=True)

        already_created = p.exists()
        assert already_created or not read_only

        uri = "file:" + sample_db
        if read_only:
            uri += "?mode=ro"
        self.db = sqlite3.connect(uri, uri=True)
        self.cur = self.db.cursor()

        # Create table if needed
        if not already_created:
            if best_solution_only:
                self.cur.execute('''CREATE TABLE milp (id text primary key, problem text, solution text, objective_sense text, objective_value float, gap float)''')
            else:
                self.cur.execute('''CREATE TABLE milp (id text key, problem text, solution text, objective_sense text, objective_value float, gap float)''')
                self.cur.execute('''CREATE INDEX id_index ON milp (id)''')

    def __del__(self):
        self.db.close()

    @functools.lru_cache(maxsize=16)
    def _model_to_key_pb(self, model):
        letters = string.ascii_letters
        tmp_file = '/tmp/' + ''.join(random.choice(letters) for i in range(10)) + '.lp'
        model.writeProblem(tmp_file)
        with open(tmp_file, 'r') as f:
            problem = f.read()
            problem = problem.encode()
        key = hashlib.sha256(problem).hexdigest()
        return key, problem

    def _better_solution_exists(self, key, obj_sense, obj_value):
        try:
            query = f"SELECT objective_value FROM milp WHERE id = \'{key}\'"
            self.cur.execute(query)
            rslt = self.cur.fetchone()
            old_value = rslt[0]
            found = True
        except:
            found = False

        if found and ((obj_sense == "minimize" and old_value < obj_value) or (obj_sense == "maximize" and old_value > obj_value)):
            return True
        else:
            return False

    def get_one(self, model):
        """
        Load the solution(s) and variable assignment(s) for the specified model.
        Encodes the solutions as the ({key, value}, obj_value) tuple, where key is the
        index of a variable in the array returned by model.getVars(transformed=True),
        value is the value of this variable in the solution, and obj_value is the
        objective value of the solution.
        """
        key, _ = self._model_to_key_pb(model)
        query = f"SELECT solution, objective_value FROM milp WHERE id = \'{key}\'"
        self.cur.execute(query)
        rslt = self.cur.fetchone()
        solution = base64.b64decode(rslt[0].encode())
        solution = pickle.loads(solution)
        obj_value = rslt[1]
        return (solution, obj_value)
     
    def get_all(self, model):
        """
        Load the solution(s) and variable assignment(s) for the specified model.
        Encodes the solutions as the ({key, value}, obj_value) tuple, where key is the
        index of a variable in the array returned by model.getVars(transformed=True),
        value is the value of this variable in the solution, and obj_value is the
        objective value of the solution.
        """
        key, _ = self._model_to_key_pb(model)
        query = f"SELECT solution, objective_value FROM milp WHERE id = \'{key}\'"
        self.cur.execute(query)
        rslts = self.cur.fetchmany()
        while len(rslts) > 0:
            for rslt in rslts:
                solution = base64.b64decode(rslt[0].encode())
                solution = pickle.loads(solution)
                obj_value = rslt[1]
                yield (solution, obj_value)
            rslts = self.cur.fetchmany()

    def add(self, model, solution, objective_value, gap):
        """
        Stores the solution and variable assignment for the specified model.
        """
        # Insert a row of data or replace it if a better solution is found
        key, problem = self._model_to_key_pb(model)
        obj_sense = model.getObjectiveSense()
        if self.best_solution_only and self._better_solution_exists(key, obj_sense, objective_value):
            return

        sol = {}
        vars = model.getVars(transformed=True)
        for i in range(len(vars)):
            v = vars[i]
            val = solution[v]
            sol[i] = val
        sol = pickle.dumps(sol)
        problem = base64.b64encode(problem).decode()
        sol = base64.b64encode(sol).decode()
        query = f"{self.sql_insert} INTO milp VALUES (\'{key}\', \'{problem}\', \'{sol}\', \'{obj_sense}\', {objective_value}, {gap})"
        self.cur.execute(query)
        self.db.commit()


    def merge(self, dataset):
        """
        Add another dataset to the current one
        """
        query = "SELECT id, problem, solution, objective_sense, objective_value, gap FROM milp"
        sample_cnt = 0
        for milp in dataset.cur.execute(query):
            key = milp[0]
            obj_sense = milp[3]
            obj_value = milp[4]
            if self.best_solution_only and self._better_solution_exists(key, obj_sense, obj_value):
                continue

            insert = f"{self.sql_insert} INTO milp VALUES (\'{milp[0]}\', \'{milp[1]}\', \'{milp[2]}\', \'{milp[3]}\', {milp[4]}, {milp[5]})"
            self.cur.execute(insert)
            sample_cnt += 1
            if sample_cnt % 1000 == 0:
                self.db.commit()

        self.db.commit()

