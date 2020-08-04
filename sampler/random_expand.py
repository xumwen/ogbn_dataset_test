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


class RandomExpandSampler(object):
    def __init__(self, data, train_idx, batch_size, subgraph_nodes, 
                sample_step=3, sample_weight=1, shuffle=True, 
                random_expand=True, policy=None, node_emb=None):
        self.policy = policy
        self.data = data
        self.train_idx = train_idx
        self.batch_size = batch_size
        self.num_nodes = data.num_nodes
        self.edge_index = data.edge_index
        self.node_emb = node_emb
        self.subgraph_nodes = subgraph_nodes
        self.sample_step = sample_step
        self.sample_weight = sample_weight
        self.shuffle = shuffle
        # If True，expand with random neighbor sample
        # Or not expand with ppo policy
        self.random_expand = random_expand
    
    def reset(self):
        if self.train_idx is None:
            self.node_visit = torch.zeros(self.num_nodes, dtype=torch.bool)
        else:
            self.node_visit = torch.ones(self.num_nodes, dtype=torch.bool)
            self.node_visit[self.train_idx] = False
    
    def get_init_nodes(self):
        #TODO:split train_idx into several batchs and expand each with multi-process
        left_nodes = np.where(self.node_visit == 0)[0]
        if self.shuffle:
            np.random.shuffle(left_nodes)
        if len(left_nodes) <= self.batch_size:
            n_id = left_nodes
        else:
            n_id = left_nodes[:self.batch_size]
        
        # visited nodes update
        self.node_visit[n_id] = True

        return n_id

    def get_neighbor(self, n_id):
        row, col = self.edge_index
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        # get 1-hop neighbor
        node_mask[n_id] = True
        edge_mask = node_mask[row]
        neighbor_id = col[edge_mask].numpy()
        
        # remove cent nodes
        tmp_node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        tmp_node_mask[neighbor_id] = True
        neighbor_id = np.where(tmp_node_mask & (node_mask==False))[0]

        return neighbor_id

    def random_neighbor_sample(self, n_id, neighbor_id):
        """ 
        Random sample to warm start without node_emb
        and to avoid exceeding subgraph_nodes
        """
        np.random.shuffle(neighbor_id)
        
        weight = torch.ones(len(neighbor_id)) * self.sample_weight
        sample_n_id = neighbor_id[torch.bernoulli(weight) == 1]

        num_left = self.subgraph_nodes - len(n_id)
        if len(sample_n_id) > num_left:
            sample_n_id = sample_n_id[:num_left]
        
        # print(f'Cent: {n_id.shape[0]:02d}, Neighbor: {neighbor_id.shape[0]:02d}, '
        #       f'Sample: {sample_n_id.shape[0]:02d}')
        
        return sample_n_id

    def neighbor_sample(self, n_id, neighbor_id, action, min_sample_num=100):
        if len(neighbor_id) <= min_sample_num:
            # sample all neighbors
            return neighbor_id
        
        subgraph_emb = self.node_emb[n_id].mean(dim=0)
        dist = torch.matmul(self.node_emb[neighbor_id], subgraph_emb)
        dist = self.rescale_dist(dist)

        mu = self.rescale_action(action)
        sigma = 0.5
        weight = self.norm_prob(dist, mu, sigma)
        
        # prob density maybe over limit
        weight[weight > 1] = 1
        sample_n_id = neighbor_id[torch.bernoulli(weight) == 1]

        # print(f'Action: {mu.item():.2f}, Cent: {n_id.shape[0]:02d}, Neighbor: {neighbor_id.shape[0]:02d}, '
        #       f'Sample: {sample_n_id.shape[0]:02d}, Ratio: {(sample_n_id.shape[0] / neighbor_id.shape[0]):.2f}')

        return sample_n_id
    
    def rescale_action(self, action):
        # rescale mu to (0, 1)
        action = np.tanh(action)
        return action * 0.5 + 0.5
    
    def rescale_dist(self, dist):
        # rescale dist to [0, 1]
        dist = dist -  dist.min()
        dist = dist / dist.max()
        return dist
    
    def norm_prob(self, x, mu, sigma):
        s1 = 1.0 / sigma / np.sqrt(2.0 * np.pi)
        s2 = -((x - mu) ** 2) / 2 / (sigma ** 2)
        return s1 * torch.exp(s2)

    def get_state(self, n_id, neighbor_id):
        cent_emb = self.node_emb[n_id].mean(dim=0)
        neighbor_emb = self.node_emb[neighbor_id].mean(dim=0)

        state = torch.cat([cent_emb, neighbor_emb], dim=0)
        return state.detach()

    def __produce_subgraph_by_nodes__(self, cent_n_id, n_id):
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
        data.cent_n_id = cent_n_id
        data.res_n_id = res_n_id
        
        # print(f'Subgraph nodes: {len(data.n_id):02d}, Edges: {len(data.e_id):02d}')

        return data

    def __produce_subgraph__(self):
        # sample several steps
        n_id = self.get_init_nodes()
        cent_n_id = n_id

        for i in range(self.sample_step):
            neighbor_id = self.get_neighbor(n_id)
            if self.random_expand:
                sample_n_id = self.random_neighbor_sample(n_id, neighbor_id)
            else:
                s = self.get_state(n_id, neighbor_id)
                action, logp = self.policy.action(s)
                sample_n_id = self.neighbor_sample(n_id, neighbor_id, action)
                sample_n_id = self.random_neighbor_sample(n_id, sample_n_id)
            
            n_id = np.union1d(n_id, sample_n_id)

            if len(n_id) >= self.subgraph_nodes or len(sample_n_id) == 0:
                break

        return self.__produce_subgraph_by_nodes__(cent_n_id, n_id)

    def __call__(self):
        self.reset()
        while self.node_visit.sum() != self.num_nodes:
            yield self.__produce_subgraph__()
    
    def __len__(self):
        if self.train_idx is not None:
            return (len(self.train_idx) - 1) // self.batch_size + 1
        else:
            return (len(self.num_nodes) - 1) // self.batch_size + 1