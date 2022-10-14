import numpy as np 
from sklearn.preprocessing import normalize
from scipy import sparse
from metrics import clustering_accuracy, connectivity
from sklearn.metrics import normalized_mutual_info_score



def compute_metrics(pred_labs, true_labs, graph, taps=None):
    acc = clustering_accuracy(true_labs, pred_labs)
    nmi = normalized_mutual_info_score(true_labs, pred_labs, average_method='geometric')
    if taps is None:
        conn = connectivity(true_labs, graph)
    else:
        conn = connectivity(taps, graph)
    print('acc:', acc, '| nmi:', nmi, '| conn:', conn)
    return acc, nmi, conn

def subspace_encoding(A, svd, is_fixed_dim=False):
    """Estimate the basis of spanned subspace. 

    Args:
        A (numpy.Array): The data points in the cluster.
        svd (TruncatedSVD): SVD for basis estimation. 
        is_fixed_dim (bool, optional): If True, just return the left singular vectors. 
            If False, first estimate the number of bases. Defaults to False.

    Returns:
        orthonormal basis. 
    """
    svd.fit(A)
    if is_fixed_dim:
        return svd.components_
    else:
        d = np.diff(svd.singular_values_)
        r = np.argmin(d)+1
        return svd.components_[:r]
    


def sample_from_subspace(basis, m=1):
    """Generate sample from the subspaces

    Args:
        basis (numpy.Array): The set of orthonormal bases.
        m (int, optional): _description_. Defaults to 1.

    Returns:
        data (numpy.Array): The generated sample.
    """
    amb_d, sub_d = basis.shape[0], basis.shape[1]
    coeff = np.random.normal(size=(sub_d, m))
    coeff = normalize(coeff, norm='l2', axis=0, copy=False)
    data = np.matmul(basis, coeff).T
    return data
    

def resample(labels, X, svd, is_fixed_dim=False, m=1):
    """ Generate sample according to the obatained clusters.

    Args:
        labels (numpy.Array): The predicted labels of local data
        X (numpy.Array): The local data points.
        svd (TruncatedSVD): The TruncatedSVD for basis construction.
        is_fixed_dim (bool, optional): whether estimate the number of bases. Defaults to False.
        m (int, optional): _description_. Defaults to 1.

    Returns:
        set_pts (numpy.Array): Generated samples. 
        re_labels (list): the corresponging labels. 
    """
    set_labels = list(set(labels))
    re_labels = np.array(
        [i for i in set_labels for j in range(m)]
         )
    set_pts = []
    for lab in set_labels:
        data = X[labels==lab]
        basis = subspace_encoding(data, svd, is_fixed_dim=is_fixed_dim).T
            
        data = sample_from_subspace(basis, m=m)
        set_pts.append(data)
    set_pts = np.vstack(set_pts)
    return (set_pts, re_labels)

def get_taps(samples, labels, predict):
    """mark the samples generated from each cluster. If the cluster only contain data points from 
    one class, the tap is the label of this class. If multiple class, just randomly select one label.

    Args:
        samples (numpy.Array): The local data points.
        labels (list): The true labels.
        predict (numpy): The predicted labels.

    Returns:
        numpy.Array: taps for the generated samples.
    """
    re = []
    for sa in samples:
        ll = list(set(labels[predict == sa]))
        if len(ll) == 1:
            re.append(ll[0])
        elif len(ll) > 1:
            re.append(
                np.random.choice(ll)
            )
    return np.array(re)
    

        
def test_partition(labs_trues, labs):
    """Test if the local cluster only contain the data points from only one subspace.

    Args:
        labs_trues (_type_): _description_
        labs (_type_): _description_

    Returns:
        float: the ratio of clusters contain points from clusters.
    """
    set_labs = list(set(labs))
    count = 0
    for lab in set_labs:
        if len(set(labs_trues[labs == lab])) >1:
            # print('PANIC!! PANIC!! PANIC!!')
            count += 1
    return count

def final_merge(L):
    """merge the predicted labels of all devices.

    Args:
        L (list): list of all devices.

    Returns:
        numpy.Array: The aggregated results.
    """
    return np.hstack([client['predict'] for client in L])


def form_graph(labels):
    N = len(labels)
    Graph = np.zeros((N, N))
    rows = []
    cols = []
    vals = []
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                rows.append(i)
                cols.append(j)
                vals.append(1.)
    
    return sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    
def test_count_clusters(d):
    total = sum(d)
    tmp = 0
    for i in range(len(d)):
        tmp += d[i]
        if tmp > 0.2*total:
            return i
    return len(d)-1
        