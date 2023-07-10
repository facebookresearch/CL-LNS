# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch_geometric
import torch
import torch.nn.init as init
#from neural_nets import prenorm 

# GINConv network derived from https://arxiv.org/abs/1810.00826
# Added the ability to embed edge information as well.
class GINConv(torch_geometric.nn.MessagePassing):
    def __init__(self, eps: float = 0.5, train_eps: bool = True):
        #kwargs.setdefault('aggr', 'add')
        #super(GINEConv, self).__init__(**kwargs)
        super().__init__('add')
        emb_size = 64

        #self.final_norm = prenorm.Prenorm(emb_size, shift=False)
        #self.feature_module_final = torch.nn.Sequential(
        #    self.final_norm,
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(emb_size, emb_size)
        #)
        #self.feature_module_final = torch.nn.ReLU()
        #self.post_conv_module = prenorm.Prenorm(emb_size, shift=False)

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size * 2, emb_size * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size * 4, emb_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size * 2, emb_size),
        )

        #self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                init.orthogonal_(t)
            else:
                init.normal_(t)
        self.eps.data.fill_(self.initial_eps)

    # def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
    #             edge_attr: OptTensor = None, size: Size = None) -> Tensor:
    #     """"""
    #     if isinstance(x, Tensor):
    #         x: OptPairTensor = (x, x)

    #     # Node and edge feature dimensionalites need to match.
    #     if isinstance(edge_index, Tensor):
    #         assert edge_attr is not None
    #         assert x[0].size(-1) == edge_attr.size(-1)
    #     elif isinstance(edge_index, SparseTensor):
    #         assert x[0].size(-1) == edge_index.size(-1)

    #     # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
    #     out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    #     x_r = x[1]
    #     if x_r is not None:
    #         out += (1 + self.eps) * x_r

    #     return self.nn(out)


    #def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        #return F.relu(x_j + edge_attr)

    def freeze_normalization(self):
        pass
        #self.final_norm.freeze_normalization()
        #self.post_conv_module.freeze_normalization()

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)

        output += (1 + self.eps) * right_features
        return self.output_module(output)

    def message(self, node_features_j, edge_features):
        output = torch.nn.functional.relu(node_features_j + edge_features)
        return output


