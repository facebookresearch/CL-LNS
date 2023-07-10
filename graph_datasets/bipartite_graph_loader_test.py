# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import unittest
import torch
import random
import string
import os

import graph_datasets.bipartite_graph as bg
import graph_datasets.bipartite_graph_dataset as bgd
import graph_datasets.bipartite_graph_loader as bgl


class BipartiteGraphLoaderTest(unittest.TestCase):
    def build_db(self, seed=None):
        random.seed(seed)
        letters = string.ascii_letters
        name = '/tmp/' + ''.join(random.choice(letters) for i in range(10))
        if os.path.exists(name):
            os.remove(name)

        db = bgd.BipartiteGraphDataset(name)
        # create a graph with 2 variables and 1 constraint
        db.add(bg.BipartiteGraph(torch.FloatTensor([123]), torch.IntTensor([[0, 1], [0, 0]]), torch.FloatTensor([32, 21]), torch.FloatTensor([78, 910]), torch.LongTensor([0]), [0], torch.FloatTensor([0.65]), [0]))
        # create a graph with 3 variables and 2 constraints
        db.add(bg.BipartiteGraph(torch.FloatTensor([456, 567]), torch.IntTensor([[0, 1, 1, 2], [0, 0, 1, 1]]), torch.FloatTensor([654, 765, 876, 987]), torch.FloatTensor([987, 109, 111]), torch.LongTensor([1, 2]), [0], torch.FloatTensor([0.56, 0.12]), [0]))
        db.commit()
        return name

    def testLoadAsPTGeom(self):
        name = self.build_db(seed="pt_geom")

        loader = bgl.BipartiteGraphLoader(name, shuffle=False)
        gen = loader.load(batch_size=1, format="pt_geom")
        g1 = next(gen)
        self.assertEqual(g1.constraint_features, torch.FloatTensor([123]))
        self.assertTrue(torch.equal(g1.variable_features, torch.FloatTensor([78, 910])))
        self.assertTrue(torch.equal(g1.edge_attr, torch.FloatTensor([32, 21])))
        g2 = next(gen)
        self.assertTrue(torch.equal(g2.constraint_features, torch.FloatTensor([456, 567])))
        self.assertTrue(torch.equal(g2.variable_features, torch.FloatTensor([987, 109, 111])))
        self.assertTrue(torch.equal(g2.edge_attr, torch.FloatTensor([654, 765, 876, 987])))

    def testLoadAsDGL(self): 
        name = self.build_db(seed="dgl")

        loader = bgl.BipartiteGraphLoader(name, shuffle=False)
        gen = loader.load(batch_size=1, format="dgl")

        g1 = next(gen)
        self.assertTrue(torch.equal(g1.nodes['variables'].data['variable_features'], torch.FloatTensor([78, 910])))
        self.assertTrue(torch.equal(g1.nodes['variables'].data['fsb_scores'], torch.FloatTensor([0.65, -1.0e10])))
        self.assertEqual(g1.nodes['constraints'].data['constraint_features'], torch.FloatTensor([123]))
        self.assertTrue(torch.equal(g1.edges['edges'].data['edge_attr'], torch.FloatTensor([32, 21])))
        self.assertTrue(g1.has_edges_between(0, 0, ("variables", "edges", "constraints")))
        self.assertTrue(g1.has_edges_between(1, 0, ("variables", "edges", "constraints")))

        g2 = next(gen)
        self.assertTrue(torch.equal(g2.nodes['variables'].data['variable_features'], torch.FloatTensor([987, 109, 111])))
        self.assertTrue(torch.equal(g2.nodes['variables'].data['fsb_scores'], torch.FloatTensor([-1.0e10, 0.56, 0.12])))
        self.assertTrue(torch.equal(g2.nodes['constraints'].data['constraint_features'], torch.FloatTensor([456, 567])))
        self.assertTrue(torch.equal(g2.edges['edges'].data['edge_attr'], torch.FloatTensor([654, 765, 876, 987])))
        self.assertTrue(g2.has_edges_between(0, 0, ("variables", "edges", "constraints")))
        self.assertTrue(g2.has_edges_between(1, 0, ("variables", "edges", "constraints")))
        self.assertTrue(g2.has_edges_between(1, 1, ("variables", "edges", "constraints")))
        self.assertTrue(g2.has_edges_between(2, 1, ("variables", "edges", "constraints")))

    def testLoadAsNTX(self): 
        name = self.build_db(seed="ntx")

        loader = bgl.BipartiteGraphLoader(name, shuffle=False)
        gen = loader.load(batch_size=1, format="ntx")
        
        g1 = next(gen)
        # TODO: figure out how to check the graph
        #nx.write_gpickle(g1, "/tmp/g1.pickle")
        #with open('/tmp/g1.txt', mode='w') as f:
        #    print(str(g1), file=f)
        
        g2 = next(gen)
        #nx.write_gpickle(g2, "/tmp/g2.pickle")
        #with open('/tmp/g2.txt', mode='w') as f:
        #    print(str(g2), file=f)

        reached_end = False
        try:
            _ = next(gen)
        except:
            reached_end = True
        self.assertTrue(reached_end)
