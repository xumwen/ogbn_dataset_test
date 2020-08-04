import time
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from tqdm import tqdm
from models.sage import SAGE
from parse_args import parse_args
from sampler.random_expand import RandomExpandSampler


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    
    for subgraph in train_loader():
        edge_index = subgraph.edge_index.to(device)

        optimizer.zero_grad()
        out = model(x[subgraph.n_id], edge_index)
        loss = F.nll_loss(out[subgraph.res_n_id], y[subgraph.cent_n_id])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        batch_correct = int(out[subgraph.res_n_id].argmax(dim=-1).eq(y[subgraph.cent_n_id]).sum())
        total_correct += batch_correct

        batch_size = subgraph.cent_n_id.shape[0]
        pbar.update(batch_size)
        # print(f'Batch acc: {(batch_correct / batch_size):.4f}')
        
    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x, test_loader)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    root = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')
    data = dataset[0]

    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split_idx[split]] = True
        data[f'{split}_mask'] = mask

    train_idx = split_idx['train']

    # load train sampler
    train_loader = RandomExpandSampler(data, train_idx=train_idx, 
                               batch_size=args.re_batch_size,
                               subgraph_nodes=args.re_subgraph_nodes, 
                               sample_step=3, 
                               sample_weight=args.re_sample_weight)

    # test sampler use ns with [-1], which means full eval with all neighbor
    test_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                    batch_size=4096, shuffle=False,
                                    num_workers=12)
    
    # define model and input
    model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=args.num_layers, device=device)
    model = model.to(device)

    x = data.x.to(device)
    y = data.y.squeeze().to(device)

    # train and test
    test_accs = []
    for run in range(1, args.num_run+1):
        print('')
        print(f'Run {run:02d}:')
        print('')

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        best_val_acc = final_test_acc = 0
        for epoch in range(1, args.epochs+1):
            loss, acc = train(epoch)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

            if epoch > 0:
                train_acc, val_acc, test_acc = test()
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                    f'Test: {test_acc:.4f}')

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_test_acc = test_acc
        test_accs.append(final_test_acc)

    test_acc = torch.tensor(test_accs)
    print('============================')
    print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')