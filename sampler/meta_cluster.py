import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import IterableDataset
from torch.distributions import Normal
from torch_geometric.data import Data
from torch.multiprocessing import Queue, Process

import numpy as np
import pandas as pd
import math
import copy
import time


class MetaClusterSampler(object):
    def __init__(self, data, num_parts=40, shuffle=True, policy=None, node_emb=None):
        self.data = data
        self.num_nodes = data.num_nodes
        self.edge_index = data.edge_index

        self.policy = policy
        self.node_emb = node_emb
        self.num_parts = num_parts
        self.shuffle = shuffle

    def reset(self):
        self.subgraphs = []

    def __produce_subgraph_by_nodes__(self, n_id):
        row, col = self.edge_index
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        edge_mask = torch.zeros(row.size(0), dtype=torch.bool)

        # get edge
        node_mask[n_id] = True
        edge_mask = node_mask[row] & node_mask[col]
        e_id = np.where(edge_mask)[0]

        # edge_index and cent_n_id reindex by n_id
        tmp = torch.empty(self.num_nodes, dtype=torch.long)
        tmp[n_id] = torch.arange(len(n_id))
        edge_index = tmp[self.edge_index[:, e_id]]

        # return data
        data = copy.copy(self.data)
        data.edge_index = edge_index

        N, E = data.num_nodes, data.num_edges
        for key, item in data:
            if item.size(0) == N:
                data[key] = item[n_id]
            elif item.size(0) == E:
                data[key] = item[e_id]
        data.n_id = n_id
        data.e_id = e_id

        return data
    
    def get_state(self):
        return self.node_emb.mean(dim=0)

    def __cluster__(self, action=None):
        state = self.get_state()
        # use self.policy
        if action is None:
            action = self.policy(state)

        centers_emb = action.reshape(self.num_parts, self.node_emb.shape[1])

        # rescale centers_emb to [0, )
        # centers_emb[centers_emb < 0] = 0

        # assign each node to a cluster
        dist = torch.mm(self.node_emb, centers_emb.T)
        prob = F.softmax(dist, dim=1)
        cluster_id = prob.argmax(dim=1)

        # induce subgraph for each cluster
        for i in range(self.num_parts):
            n_id = cluster_id[cluster_id == i]
            # print("subgraph num nodes:", len(n_id))
            self.subgraphs.append(self.__produce_subgraph_by_nodes__(n_id))
    
    def __call__(self):
        self.reset()
        self.__cluster__()
    
    def __len__(self):
        return self.num_parts