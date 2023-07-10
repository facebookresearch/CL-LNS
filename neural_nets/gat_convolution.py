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


# GATConvolution network derived https://arxiv.org/abs/2105.14491
# Added edge embedding as well 
class GATConvolution(torch_geometric.nn.MessagePassing):
    """
    Graph convolution layer. THis is the heart of our GNNPolicy
    """
    def __init__(self,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 **kwargs):
        super().__init__('add')
        emb_size = 64

        self.heads = 8
        self.in_channels = emb_size
        self.out_channels = emb_size // self.heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = torch.nn.Linear(self.in_channels, self.heads * self.out_channels, bias=True)
        self.lin_r = torch.nn.Linear(self.in_channels, self.heads * self.out_channels, bias=True)

        self.att = torch.nn.Parameter(torch.Tensor(1, self.heads, self.out_channels * 3))

        # output_layer
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.lin_l.weight)
        init.orthogonal_(self.lin_r.weight)
        init.orthogonal_(self.att)

    def freeze_normalization(self):
        pass

    def reset_normalization(self):
        pass

    @property
    def frozen(self):
        return False


    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        H, C = self.heads, self.out_channels

        x_l = self.lin_l(left_features)
        x_r = self.lin_r(right_features)

        out = self.propagate(edge_indices, x=(x_l, x_r), size=(left_features.shape[0], right_features.shape[0]), edge_features=edge_features)
        return self.output_module(torch.cat([out, right_features], dim=-1))

    def message(self, x_j, x_i,
                index,
                edge_features):
        x = torch.cat([x_i, x_j, edge_features], dim=-1)
        x = torch.nn.functional.leaky_relu(x, self.negative_slope)
        x = x.view(-1, self.heads, self.out_channels * 3)
        alpha = (x * self.att).sum(dim=-1)
        alpha = torch_geometric.utils.softmax(alpha, index)
        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)
        x = x_j.view(-1, self.heads, self.out_channels) * alpha.unsqueeze(-1)
        return x.view(-1, self.heads * self.out_channels)

