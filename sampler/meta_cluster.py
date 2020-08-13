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

        self.subgraphs = []
        self.__meta_cluster__()

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
        res_n_id = tmp[cent_n_id]

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

    def __meta_cluster__(self):
        # policy outputs cluster center embedding
        state = self.get_state()
        action = self.policy(state)
        centers = action.reshape(self.num_parts, self.node_emb.shape[1])

        #TODO: assign each node to a cluster
        dist = torch.matmul(self.node_emb, subgraph_emb)
        prob = F.softmax(dist, dim=1)
        cluster_id = prob.argmax(prob, dim=1)

        for i in range(self.num_parts):
            n_id = cluster_id[cluster_id == i]
            self.subgraphs.append(__produce_subgraph_by_nodes__(n_id))
        
        return
    
    def __len__(self):
        return self.num_parts