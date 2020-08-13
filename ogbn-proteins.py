import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.data import RandomNodeSampler
from sampler.meta_cluster import MetaClusterSampler
from policy import PPO
from environment import ProteinEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_parts = 40          # meta_cluster num parts
hidden_size = 32        # node_emb size
num_layers = 28         # deepgcn layers
meta_start_epoch = 1    # use meta_cluster sampler after several epochs for warm start
ppo_start_epoch = 0     # ppo training start epoch for warm start

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(DeeperGCN, self).__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        node_emb = x

        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x), node_emb


def load_proteins_dataset():
    dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.node_species = None
    data.y = data.y.to(torch.float)
    data.n_id = torch.arange(data.num_nodes)

    # Initialize features of nodes by aggregating edge features.
    row, col = data.edge_index
    data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')

    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask
    
    return data


def update_node_embedding(epoch_node_emb, node_emb, moving_avg=0.9):
    # normalize current epoch embedding
    mean = epoch_node_emb.mean()
    std = epoch_node_emb.std()
    epoch_node_emb = (epoch_node_emb - mean) / (std + 1e-5)

    # update node embedding by moving average
    node_emb = moving_avg * node_emb + \
        (1 - moving_avg) * epoch_node_emb
    
    return node_emb


def train(epoch, model, loader, node_emb):
    model.train()

    pbar = tqdm(total=len(loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    epoch_node_emb = torch.zeros_like(node_emb)
    for data in loader:
        # print("node:", data.x.shape)
        # print("edge:", data.edge_attr.shape)
        optimizer.zero_grad()
        data = data.to(device)
        out, batch_emb = model(data.x, data.edge_index, data.edge_attr)
        epoch_node_emb[data.n_id] = batch_emb.to('cpu')

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    node_emb = update_node_embedding(epoch_node_emb, node_emb)

    return total_loss / total_examples, node_emb


@torch.no_grad()
def test(model, loader):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in loader:
        data = data.to(device)
        out, _ = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

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

    return train_rocauc, valid_rocauc, test_rocauc


if __name__ == '__main__':
    # protein task required
    data = load_proteins_dataset()
    model = DeeperGCN(hidden_channels=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    evaluator = Evaluator('ogbn-proteins')

    # meta cluster sampler required
    ppo = PPO(state_size=hidden_size, out_channels=num_parts * hidden_size, device=device)
    node_emb = torch.zeros(data.num_nodes, hidden_size).to('cpu')
    epoch_node_emb = torch.zeros(data.num_nodes, hidden_size).to('cpu')

    # train and evaluation
    for epoch in range(1, 1001):
        if epoch <= meta_start_epoch:
            train_loader = RandomNodeSampler(data, num_parts=num_parts, shuffle=True,
                                        num_workers=5)
        else:
            train_loader = MetaClusterSampler(data, num_parts=40, policy=ppo.policy, node_emb=node_emb)
        
        test_loader = RandomNodeSampler(data, num_parts=5, num_workers=5)

        loss, node_emb = train(epoch, model, train_loader, node_emb)
        train_rocauc, valid_rocauc, test_rocauc = test(model, test_loader)
        print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
            f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')

        if epoch > ppo_start_epoch:
            env = ProteinEnv(data, model, node_emb, device)
            ppo.train_step(env)
        