from model.SCs import SSC, TSC 
from model.fed_sc import fed_sc
from data_setup.distribute_data import distribute_syn
from utils import compute_metrics

import numpy as np 
import time 


def syn(): 
    range_localK = 4
    ambient_dim = 20 # 30
    # subspace_dim = [7,9]
    subspace_dim = [7,8]
    num_subspaces = 20
    num_points_per_subspace = 50

    
    """ SCs """
    local_device_0 = SSC(n_cluster=None, is_noisy=False)
    local_device_1 = SSC(n_cluster=None, is_noisy=False)
    
    central_server_ssc = SSC(n_cluster=num_subspaces, is_noisy=False)
    central_server_tsc = TSC(n_cluster=num_subspaces, q= 6)
    

    num_clients = 400
    
    
        
    L, L_num_clusters = distribute_syn(num_clients, ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, range_localK, noise=0.0)
    true_labs = np.hstack([cl['label'] for cl in L])
        
    t_start = time.time()
    pred_labs, taps = fed_sc(local_device_0, central_server_ssc, L, m=1, fixed_dim=None)
    t = time.time()-t_start
    re = compute_metrics(pred_labs, true_labs, central_server_ssc.W, taps)

if __name__ == "__main__":
    import argparse
    import os
    import multiprocessing as mp


    parser = argparse.ArgumentParser(description='Demo for synthetic data')
    
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('-r', '--root', type=str, default='./trainded_models', help='The root of trained models')
    syn()