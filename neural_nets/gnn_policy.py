# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn.init as init

from neural_nets import gat_convolution
from neural_nets import gin_convolution
from neural_nets import gasse_convolution
from neural_nets import prenorm 

# Implements the branching policy described in
# https://papers.nips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
class GNNPolicy(torch.nn.Module):
    def __init__(self, gnn_type="gasse"):
        super().__init__()
        emb_size = 64
        cons_nfeats = 10
        edge_nfeats = 2
        var_nfeats = 104 # hard-coded no good

        # Constraint embedding
        self.cons_norm = prenorm.Prenorm(cons_nfeats)
        self.cons_embedding = torch.nn.Sequential(
            self.cons_norm,
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # Edge embedding
        self.edge_norm = prenorm.Prenorm(edge_nfeats)
        self.edge_embedding = torch.nn.Sequential(
            self.edge_norm,
            torch.nn.Linear(edge_nfeats, emb_size),
        )
        #self.edge_embedding = torch.nn.Linear(edge_nfeats, emb_size)

        # Variable embedding
        self.var_norm = prenorm.Prenorm(var_nfeats, preserve_features=[2])
        self.var_embedding = torch.nn.Sequential(
            self.var_norm,
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        if gnn_type == "gasse":
            self.conv_v_to_c = gasse_convolution.GasseConvolution()
            self.conv_c_to_v = gasse_convolution.GasseConvolution()
        elif gnn_type == "gin":
            self.conv_v_to_c = gin_convolution.GINConv()
            self.conv_c_to_v = gin_convolution.GINConv()
        else:
            self.conv_v_to_c = gat_convolution.GATConvolution()
            self.conv_c_to_v = gat_convolution.GATConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )
        
        self.reset_parameters()


    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                init.orthogonal_(t)
            else:
                init.normal_(t)

    def freeze_normalization(self):
        if not self.cons_norm.frozen:
            self.cons_norm.freeze_normalization()
            self.edge_norm.freeze_normalization()
            self.var_norm.freeze_normalization()
            self.conv_v_to_c.reset_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_v_to_c.frozen:
            self.conv_v_to_c.freeze_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_c_to_v.frozen:
            self.conv_c_to_v.freeze_normalization()
            return False
        return True


    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output
