# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import graph_datasets.bipartite_graph_dataset as bgd
import torch_geometric
#import dgl
import random
import torch



class BipartiteGraphLoader:
    def __init__(self, db, shuffle=True, first_k=None):
        self.shuffle = shuffle
        dbs = db.split('+')
        if len(dbs) == 1:
            self.data = bgd.BipartiteGraphDataset(db, query_opt=not shuffle, read_only=True, first_k=first_k)
        else:
            self.data = bgd.BipartiteGraphDatasets(dbs, query_opt=not shuffle, first_k=first_k)

    def num_examples(self):
        return self.data.sample_cnt
            
    def load(self, batch_size=32, format="pt_geom"):
        #from IPython import embed;embed()
        if format == "pt_geom":
            #print("here")
            def my_collate(batch):
                #embed()
                #print(len(batch))
                #batch = list(filter(lambda x: torch.sum(x.candidate_scores) > 0.5 * x.info["neighborhood_size"], batch))
                #return None
                #from IPython import embed; embed()
                batch = list(filter(lambda x: x is not None), batch)
                return torch.utils.data.dataloader.default_collate(batch)

            loader = torch_geometric.loader.DataLoader(self.data, batch_size, shuffle=self.shuffle)#, collate_fn=my_collate)
            for ptg in loader:
                #from IPython import embed;embed()
                yield ptg
            return

        elif format == 'dgl':
            k = self.data.len()
            permutation = random.sample(range(k), k)
            graphs = []
            for loc in permutation:
                ptg = self.data.get(loc)
                ntx = ptg.to_networkx()
                #print("here")
                #from IPython import embed;embed()
        
                dgl_graph = dgl.bipartite_from_networkx(ntx, utype='variables', etype='edges', vtype='constraints',
                                u_attrs=['variable_features'], e_attrs=['edge_attr'], v_attrs=['constraint_features'])

                # Annotate the variables with other information
                num_variables = dgl_graph.nodes("variables").size(0)
                fsb_scores = torch.full((num_variables,), -1.0e10)  #, dype=torch.float)
                candidate_scores = ntx.graph["candidate_scores"]
                branching_candidates = ntx.graph["candidates"]
                num_candidates = branching_candidates.size(0)
                for i in range(num_candidates):
                    candidate_id = branching_candidates[i]
                    candidate_score = candidate_scores[i]
                    assert candidate_score >= 0
                    fsb_scores[candidate_id] = candidate_score

                dgl_graph.nodes['variables'].data['fsb_scores'] = fsb_scores

                graphs.append(dgl_graph)
                if len(graphs) == batch_size:
                    yield dgl.batch(graphs)
                    graphs = []
            return

        assert format == 'ntx'
        k = self.data.len()
        permutation = random.sample(range(k), k)
        batch = []
        for loc in permutation:
            ptg = self.data.get(loc)
            ntx = ptg.to_networkx()
            batch.append(ntx)
            if len(batch) == batch_size:
                yield batch
                batch = []

