# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch_geometric
import torch
import numpy as np
import networkx as nx


class BipartiteGraph(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores, info, 
                 iteration = None, instance_id = None, incumbent_history = None, LB_relaxation_history = None, improvement_history = None, neighborhood_size = None):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        #print("Variable features shape", variable_features.shape)
        self.candidates = candidates
        self.nb_candidates = len(candidates) if candidates is not None else 0
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores
        self.info = info
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        self.num_nodes = constraint_features.shape[0] if constraint_features is not None else 0
        self.num_nodes += variable_features.shape[0] if variable_features is not None else 0
        self.iteration = iteration 
        self.instance_id = instance_id
        self.incumbent_history = incumbent_history
        self.LB_relaxation_history = LB_relaxation_history
        self.improvement_history = improvement_history
        self.neighborhood_size = neighborhood_size


    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.shape[0]], [self.variable_features.shape[0]]])
        elif key == 'candidates':
            return self.variable_features.shape[0]
        else:
            return super().__inc__(key, value)

    def to_networkx(self):
        G = nx.DiGraph(candidates=self.candidates, candidate_scores=self.candidate_scores,
                       nb_candidates=self.nb_candidates, candidate_choice=self.candidate_choices,
                       info=self.info)

        G.add_nodes_from(range(self.num_nodes))
       
        num_vars = self.variable_features.shape[0]
        #print(num_vars)
        for i, (v, u) in enumerate(self.edge_index.T.tolist()):
            G.add_edge(u, v+num_vars)
            #print(u, v)
            assert 0 <= u and u < num_vars, str(u)
            assert v >= 0, str(v)
            G[u][v+num_vars]["edge_attr"] = self.edge_attr[i]

        for i, feat_dict in G.nodes(data=True):
            if i < num_vars:
                feat_dict.update({"variable_features": self.variable_features[i].squeeze()})
                feat_dict.update({"bipartite": 0})
            else:
                feat_dict.update({"constraint_features": self.constraint_features[i-num_vars].squeeze()})
                feat_dict.update({"bipartite": 1})
        for u, v in G.edges():
            #print(u, v, G.nodes[u]['bipartite'], G.nodes[v]['bipartite'], num_vars)
            assert(G.nodes[u]['bipartite'] == 0)
            assert(G.nodes[v]['bipartite'] == 1)
        return G
