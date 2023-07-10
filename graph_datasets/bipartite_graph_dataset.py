# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch_geometric
import sqlite3
import pickle
import base64
import random
from pathlib import Path
from graph_datasets.bipartite_graph import BipartiteGraph
import intervaltree
import zlib
import torch


class BipartiteGraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_db, query_opt=False, read_only=False, first_k=None):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.query_opt = query_opt

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
            self.cur.execute('''CREATE TABLE samples (id integer primary key asc, features text not null unique)''')
            #self.cur.execute('''CREATE UNIQUE INDEX per_id ON samples(id)''') 
            self.cur.execute('''INSERT INTO samples VALUES (-1, \'0\')''')
            self.db.commit()
            self.sample_cnt = 0
        else:
            self.cur.execute("SELECT features FROM samples WHERE id = -1")
            rslt = self.cur.fetchone()
            self.sample_cnt = int(rslt[0])

        if first_k is not None:
            self.sample_cnt = min(self.sample_cnt, first_k)
            print(f"Use first_k = {first_k}. Dataset size  = {self.sample_cnt}")

    def __del__(self):
        self.db.close()
    

    def len(self):
        return self.sample_cnt

    def get(self, index):
        """
        Load a bipartite graph observation as saved on the disk during data collection.
        """
        #print("here: get")
        #assert False
        #from IPython import embed; embed()
        if self.query_opt:
            # Ignore the requested index, so we can stream data
            rslt = self.cur.fetchone()
            if rslt is None:
                query = "SELECT features FROM samples WHERE id >= 0"
                self.cur.execute(query)
                rslt = self.cur.fetchone()
                assert rslt is not None
        else:
            # Fetch the data at the requested index. This is much slower
            query = f"SELECT features FROM samples WHERE id = {index}"
            self.cur.execute(query)
            rslt = self.cur.fetchone()

        entry = base64.b64decode(rslt[0].encode())
        try:
            raw = zlib.decompress(entry)
        except:
            # Old uncompressed dataset
            raw = entry

        graph = pickle.loads(raw)
        #from IPython import embed; embed()
        #if torch.sum(graph.candidate_scores).item() < 25:
        #    return None
        #if index % 2 ==0 :
        #    return None
        return graph

    def add(self, graph):
        """
        Add a bipartite graph observation to the dataset. Only adds the observation if it wasn't
        already present in the dataset
        """
        # Insert a row of data
        raw = pickle.dumps(graph)
        compressed = zlib.compress(raw, level=9)
        sample = base64.b64encode(compressed).decode()
        query = f"INSERT INTO samples VALUES ({self.sample_cnt}, \'{sample}\')"
        try:
            self.cur.execute(query)
            self.sample_cnt += 1
            self.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def merge(self, dataset):
        """
        Add another dataset to the current one
        """
        query = "SELECT features FROM samples WHERE id >= 0"
        for sample in dataset.cur.execute(query):
            insert = f"INSERT INTO samples VALUES ({self.sample_cnt}, \'{sample[0]}\')"
            try:
                self.cur.execute(insert)
                self.sample_cnt += 1
            except sqlite3.IntegrityError:
                continue
            if self.sample_cnt % 1000 == 0:
                self.commit()

        self.commit()

    def merge_multiple(self, datasets):
        """
        Add several other datasets to the current one
        """
        query = "SELECT features FROM samples WHERE id >= 0"
        for dataset in datasets:
            dataset.cur.execute(query)

        done = False
        while not done:
            idx = random.randint(0, len(datasets)-1)
            dataset = datasets[idx]
            sample = dataset.cur.fetchone()
            if sample is None:
                datasets.pop(idx)
                if len(datasets) == 0:
                    done = True
            else:
                insert = f"INSERT INTO samples VALUES ({self.sample_cnt}, \'{sample[0]}\')"
                try:
                    self.cur.execute(insert)
                    self.sample_cnt += 1
                except sqlite3.IntegrityError:
                    continue
                if self.sample_cnt % 1000 == 0:
                    self.commit()

        self.commit()

    def commit(self):
        query = f"INSERT OR REPLACE INTO samples VALUES (-1, \'{self.sample_cnt}\')"
        self.cur.execute(query)
        self.db.commit()


class BipartiteGraphDatasets(torch_geometric.data.Dataset):
    """
    Allows training on the data from multiple datasets.
    """
    def __init__(self, databases, query_opt=False, first_k=None):
        super().__init__(root=None, transform=None, pre_transform=None)

        if first_k:
            first_k = max(1,first_k // len(databases))
        self.dbs = intervaltree.IntervalTree()
        self.sample_cnt = 0
        for db in databases:
            p = Path(db)
            assert p.exists()
            dataset = BipartiteGraphDataset(db, query_opt, read_only=True, first_k=first_k)
            new_samples = dataset.len()
            self.dbs[self.sample_cnt:self.sample_cnt+new_samples] = dataset
            self.sample_cnt += new_samples

    def len(self):
        return self.sample_cnt

    def get(self, index):
        """
        Load a bipartite graph observation as saved on the disk during data collection.
        """
        rslt = None
        #while rslt is None:
        d = self.dbs[index].pop()
        db = d.data
        index -= d.begin
        rslt = db.get(index)
        return rslt
