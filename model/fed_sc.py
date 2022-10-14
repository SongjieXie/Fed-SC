from utils import resample, test_partition, get_taps

import numpy as np
from sklearn.decomposition import TruncatedSVD
import progressbar




def fed_sc(local_alg, server_alg, L, fixed_dim=None, maximal_dim=15, comm_noise=0.0, m=1):
    """ The python implementation of Fed-SC. 
    It is the sequential running version of Fed-SC where each device runs in sequence and 
    no parallel across the device. The total running time t is the summation of each device t_z
    and the central server t_c, i.e., t = \sum_{z=1}^Z t_z + t_c.  
    
    ------------
    Parameters:
    ------------
        local_alg (class of SC):
            The SC algorithm conduted on each local device. The BaseSC classes with 
            a method .fit() and attributes .label_ and .W are expected.
        server_alg (class of SC):
            The SC algorithm conduted on each local device. The BaseSC classes with 
            a method .fit() and attributes .label_ and .W are expected.
        L (list of dict):
            L = [{'data': [], 'label':[]}, ...]. It is the list of data residing local 
            devices. 
        fixed_dim (int or None. defualt: None):
            If fixed_dim is None, the local sc will estimate dimensionality of the subspace 
            spaned by the data points in each cluster. If fixed_dim is int, the basis of fixed_dim
            singular left vectors is built. 
        maximal_dim (int. default: 15):
            The number of componets in truncated svd for subspace basis construction (subspace encoding). 
        comm_noise (float. default:0.0):
            The variance of Gaussian noise for communication that is added to generated samples. 
        m (int. default: 1):
            The number of samples generated from each local cluster. Fed-SC with multiple samples is still
            under development. 
            
    ------------
    Returns:
    ------------
        pred_labels (numpy.Array):
            The aggregated updated data partitions of all the local devices (\hat{T}). 
        partition_taps (numpy.Array):
            The cluster assignments of all the generated samples.
    """
    
    samples_data = []
    samples_label = []
    clients_label = []
    partition_taps = []
    ccount = 0
    
    """ Initialize the TruncatedSVD for subspace encoding """
    if fixed_dim is None:
        Tsvd = TruncatedSVD(n_components=maximal_dim)
    else:
        Tsvd = TruncatedSVD(n_components=fixed_dim)
        
    for (i_client, client) in enumerate(progressbar.progressbar(L)):
        """ The local clustering """
        local_alg.fit(client['data'])
        client['predict'] = local_alg.labels_
        
        
        """ Re sampling """
        re_samples = resample(local_alg.labels_, client['data'], Tsvd, 
                              is_fixed_dim=fixed_dim is not None, m=1)
        
        ccount += test_partition(client['label'], client['predict'])
        partition_taps.append(
            get_taps(re_samples[1], client['label'], client['predict']))
        samples_data.append(re_samples[0]) 
        samples_label.append(re_samples[1])
        clients_label.append(
            i_client*np.ones_like(re_samples[1])
        )
    
        
    """ Merge generated samples and labels """
    samples_data = np.vstack(samples_data)
    samples_data += np.random.normal(size=samples_data.shape)* comm_noise
    samples_label = np.hstack(samples_label)
    clients_label = np.hstack(clients_label) 
    partition_taps = np.hstack(partition_taps)
    
    
    assert len(partition_taps) == len(samples_data), 'the partition error'
    
    ccount = ccount/samples_data.shape[0]
    
    """ Central clustering """
    server_alg.fit(samples_data)
    global_result = server_alg.labels_
    
    """ Local Update """
    if m == 1:
        for (i_client, client) in enumerate(L):
            turned_label = global_result[clients_label==i_client]
            primer_label = samples_label[clients_label==i_client]
            assert len(primer_label) == len(set(client['predict']))
            final = np.zeros_like(client['predict'])
            for (i, lab) in enumerate(primer_label):
                final[client['predict']== lab] = turned_label[i]
            client['predict'] = final
    
    elif m > 1:
        assert 1 == 0, 'Not implement !!!'
    print('The number of errors in local partition: {0}'.format(ccount))
    
    pred_labels = np.hstack([cl['predict'] for cl in L])
    
    return pred_labels, partition_taps