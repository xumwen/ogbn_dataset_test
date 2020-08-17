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
    def __init__(self, data, num_parts=40, policy=None, node_emb=None, min_cluster_node_num=100):
        self.data = data
        self.edge_index = data.edge_index
        self.num_nodes = data.num_nodes
        self.num_edges = data.edge_index.shape[1]

        self.policy = policy
        self.node_emb = node_emb
        self.num_parts = num_parts
        self.min_cluster_node_num = min_cluster_node_num


    def reset(self):
        self.subgraphs = []

    def __produce_subgraph_by_nodes__(self, n_id):
        row, col = self.edge_index
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        edge_mask = torch.zeros(self.num_edges, dtype=torch.bool)

        # get edge
        node_mask[n_id] = True
        edge_mask = node_mask[row] & node_mask[col]
        e_id = torch.arange(self.num_edges)[edge_mask]

        # edge_index and cent_n_id reindex by n_id
        tmp = torch.empty(self.num_nodes, dtype=torch.long)
        tmp[n_id] = torch.arange(len(n_id))
        edge_index = tmp[self.edge_index[:, e_id]]

        # return data
        data = copy.copy(self.data)
        data.edge_index = edge_index

        N, E = self.num_nodes, self.num_edges
        print('n_id:', len(n_id), 'e_id:', len(e_id))
        for key, item in data:
            if item.size(0) == N:
                data[key] = item[n_id]
            elif item.size(0) == E:
                data[key] = item[e_id]
            else:
                data[key] = item

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

        # assign each node to a cluster
        dist = torch.mm(self.node_emb, centers_emb.T)
        prob = F.softmax(dist, dim=1)
        cluster_id = prob.argmax(dim=1)

        # induce subgraph for each cluster nodes
        for i in range(self.num_parts):
            n_id = torch.arange(self.num_nodes)[cluster_id == i]
            if len(n_id) >= self.min_cluster_node_num:
                self.subgraphs.append(self.__produce_subgraph_by_nodes__(n_id))
    
    def __call__(self):
        self.reset()
        self.__cluster__()
    
    def __len__(self):
        return self.num_parts