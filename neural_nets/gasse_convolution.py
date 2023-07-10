# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch_geometric
import torch
import torch.nn.init as init
from neural_nets import prenorm 


# Implements the graph convolution described in
# https://papers.nips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
class GasseConvolution(torch_geometric.nn.MessagePassing):
    """
    Graph convolution layer. THis is the heart of our GNNPolicy
    """
    def __init__(self):
        super().__init__('add')
        emb_size = 64

        self.feature_module_left = torch.nn.Linear(emb_size, emb_size)
        self.feature_module_edge = torch.nn.Linear(emb_size, emb_size, bias=False)
        self.feature_module_right = torch.nn.Linear(emb_size, emb_size, bias=False)

        self.final_norm = prenorm.Prenorm(emb_size, shift=False)
        self.feature_module_final = torch.nn.Sequential(
            self.final_norm,
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )
        self.post_conv_module = prenorm.Prenorm(emb_size, shift=False)

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        
        self.reset_parameters()


    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                init.orthogonal_(t)
            else:
                init.normal_(t)

    def reset_normalization(self):
        self.final_norm.reset_normalization()
        self.post_conv_module.reset_normalization()

    @property
    def frozen(self):
        return self.final_norm.frozen and self.post_conv_module.frozen
        
    def freeze_normalization(self):
        if not self.final_norm.frozen:
            self.final_norm.freeze_normalization()
            self.post_conv_module.reset_normalization()
            return False
        if not self.post_conv_module.frozen:
            self.post_conv_module.freeze_normalization()
            return False
        return True
        
    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output
