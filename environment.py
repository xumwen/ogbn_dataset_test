import math
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sampler.meta_cluster import MetaClusterSampler
from ogb.nodeproppred import Evaluator


class ProteinEnv():
    def __init__(self, data, model, node_emb, device):
        self.model = model
        self.data = data
        self.num_nodes = data.num_nodes

        self.node_emb = node_emb
        self.device = device

        self.meta_cluster = MetaClusterSampler(data, num_parts=40, policy=None, node_emb=node_emb)
    
    def get_state(self):
        return self.meta_cluster.get_state()
    
    def step(self, action, eid):
        self.meta_cluster.reset()
        self.meta_cluster.__cluster__(action)

        loader = self.meta_cluster.subgraphs
        
        return self.test(loader, eid)

    @torch.no_grad()
    def test(self, loader, eid):
        self.model.eval()

        y_true = {'train': [], 'valid': [], 'test': []}
        y_pred = {'train': [], 'valid': [], 'test': []}

        pbar = tqdm(total=len(loader))
        pbar.set_description(f'PPO episode: {eid:04d}')

        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.edge_attr)

            for split in y_true.keys():
                mask = data[f'{split}_mask']
                y_true[split].append(data.y[mask].cpu())
                y_pred[split].append(out[mask].cpu())

            pbar.update(1)

        pbar.close()

        evaluator = Evaluator('ogbn-proteins')

        train_rocauc = evaluator.eval({
            'y_true': torch.cat(y_true['train'], dim=0),
            'y_pred': torch.cat(y_pred['train'], dim=0),
        })['rocauc']

        valid_rocauc = evaluator.eval({
            'y_true': torch.cat(y_true['valid'], dim=0),
            'y_pred': torch.cat(y_pred['valid'], dim=0),
        })['rocauc']

        test_rocauc = evaluator.eval({
            'y_true': torch.cat(y_true['test'], dim=0),
            'y_pred': torch.cat(y_pred['test'], dim=0),
        })['rocauc']

        return valid_rocauc