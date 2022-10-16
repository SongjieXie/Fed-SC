from model.SCs import SSC, TSC 
from model.fed_sc import fed_sc
from data_setup.distribute_data import distribute_real, distribute_syn, distribute_real_cattering
from utils import compute_metrics

import numpy as np 
import time 


def main(args): 
    client_alg = SSC(args.localK, is_noisy=args.noise==0, gamma=args.gamma)
    if args.method == 'FedSC-SSC':
        server_alg = SSC(args.num_subspaces, is_noisy=args.noise==0, gamma=args.gamma)
    elif args.method == 'FedSC-TSC':
        server_alg = TSC(args.num_subspaces, q= args.q)
    else:
        raise NotImplementedError
    
    if args.d == 'syn':
        L = distribute_syn(args.num_clients, args.ambient_dim, args.subspace_dim, 
                           args.num_subspaces, args.num_points_per_subspace, args.range_localK, noise=0.0)
    elif args.d == 'EMNIST':
        L = distribute_real_cattering('EMNIST', args.num_clients, args.num_points_per_subspace, 
                                      args.range_localK, args.device)
    elif args.d == 'COIL100':
        L = distribute_real('COIL100', args.num_clients, args.num_points_per_subspace, 
                            args.range_localK, args.device)
    else:
        assert 1 == 0, 'Currently do not support such datasets'
        
        
        
    t_start = time.time()
    pred_labs, taps = fed_sc(client_alg, server_alg, L, fixed_dim=args.fixed_dim, m=1)
    t = time.time()-t_start
    print('time: ', t)
    true_labs = np.hstack([cl['label'] for cl in L])
    compute_metrics(pred_labs, true_labs, server_alg.W, taps)
    
    
    

if __name__ == "__main__":
    import argparse
    import os
    import multiprocessing as mp


    parser = argparse.ArgumentParser(description='Demo for Fed-SC')
    
    parser.add_argument('-d', type=str, default='syn', help='Synthetic data (syn) or real-world data (EMNIST or COIL100).')
    
    
    
    parser.add_argument('--method', type=str, default='FedSC-SSC', help='FedSC-SSC or FedSC-TSC')
    parser.add_argument('--gamma', type=int, default=500, help='Parameter for lasso version SSC')
    parser.add_argument('--q', type=int, default=8, help='Parameter for TSC')
    
    parser.add_argument('--localK', type=int, default=None, help='Local number')
    parser.add_argument('--fixed_dim', type=int, default=None, help='FedSC-SSC or FedSC-TSC')
    
    parser.add_argument('--noise', type=float, default=0.0, help='Variance of Gaussian noise added to data')
    parser.add_argument('--num_clients', type=int, default=400, help='The number of clients')
    parser.add_argument('--num_subspaces', type=int, default=20, help='The number of subspaces')
    parser.add_argument('--ambient_dim', type=int, default=20, help='The dimension of the ambient space')
    parser.add_argument('--range_localK', type=str, default='4,6', help='The range of the number of local subspaces')
    parser.add_argument('--num_points_per_subspace', type=int, default=50, help='The number of points in each local subspace')
    parser.add_argument('--subspace_dim', type=str, default='7,8', help='The range of dimension of each subspace.')
    parser.add_argument('--device', type=str, default='cpu')
    
    
    args = parser.parse_args()
    
    
    args.range_localK = [int(s) for s in args.range_localK.split(',')]
    args.subspace_dim = [int(s) for s in args.subspace_dim.split(',')]
    
    main(args)