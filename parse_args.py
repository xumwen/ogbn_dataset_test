import argparse
from easydict import EasyDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='product')
    parser.add_argument('--num_run', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--sampler', type=str, default='re',
                        choices=['rw', 'ns', 'cluster', 're'])
    parser.add_argument('--num_layers', type=int, default=3)
    # for cluster gcn
    parser.add_argument('--cluster_part_num', type=int, default=40000)
    parser.add_argument('--cluster_batch_size', type=int, default=20)
    # for graph saint
    parser.add_argument('--saint_batch_size', type=int, default=2000)
    parser.add_argument('--saint_walk_length', type=int, default=3)
    parser.add_argument('--saint_sample_coverage', type=int, default=50)
    parser.add_argument('--use_saint_norm', type=int, default=1)
    # for random expand
    parser.add_argument('--re_batch_size', type=int, default=2000)
    parser.add_argument('--re_sample_weight', type=float, default=1.0)
    parser.add_argument('--re_subgraph_nodes', type=int, default=50000)

    args = parser.parse_args()
    args = vars(args)

    config = EasyDict()
    for k, v in args.items():
        if v is not None:
            config[k] = v

    return config