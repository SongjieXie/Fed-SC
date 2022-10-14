from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import _supervised
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eig
from scipy.sparse.linalg import eigsh
import numpy as np

def clustering_accuracy(labels_true, labels_pred):
    """Clustering Accuracy between two clusterings.
    Clustering Accuracy is a measure of the similarity between two labels of
    the same data. Assume that both labels_true and labels_pred contain n 
    distinct labels. Clustering Accuracy is the maximum accuracy over all
    possible permutations of the labels, i.e.
    \max_{\sigma} \sum_i labels_true[i] == \sigma(labels_pred[i])
    where \sigma is a mapping from the set of unique labels of labels_pred to
    the set of unique labels of labels_true. Clustering accuracy is one if 
    and only if there is a permutation of the labels such that there is an
    exact match
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.
    
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    labels_pred : array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    
    Returns
    -------
    accuracy : float
       return clustering accuracy in the range of [0, 1]
    """
    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)


def connectivity(labels_true, representation_matrix, compute='ave'):
    """[summary]

    Args:
        labels_true ([type]): [description]
        representation_matrix ([type]): [description]
        compute (str, optional): [description]. Defaults to 'min'. 'ave'
    """
    set_labels = set(labels_true)
    connes = []
    for label in set_labels:
        sub_rep_matrix = representation_matrix[labels_true == label][:,labels_true == label]
        
        graph_lap = laplacian(
            np.abs(sub_rep_matrix)+np.abs(sub_rep_matrix.T), 
            normed=True
        )
        s_val, _ = eigsh(graph_lap, which='LM', k=2, sigma=1e-4)
        connes.append(np.sort(s_val)[1])
    if compute == 'min':
        return min(connes)
    elif compute == 'ave':
        return sum(connes)/len(connes)               
        
