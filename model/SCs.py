from matplotlib import test
import numpy as np
import progressbar
import spams

from scipy import sparse
from sklearn import cluster
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class BaseSC:
    """ The basic class for spectral-based subspace clustering methods 
    with a sequence of attributes including building affinity graph, 
    constructing graph laplacian, applying spectral clustering and 
    resetting to clear up for reusage.
    
    ------------
    Parameters:
    -----------
    n_clusters (integer or None. default: None):
        Number of clusters to segment the input dataset
        < n_clusters is int > then the applied spectral clustering with the 
        given number of clusters.
        < n_clussters is None > then the number of inherent clusters need to 
        be estimated before applying spectral clustering
    n_init (integer. default: 20):
        ???
    kmeans_alg ('kmeans++' ???. default: 'kmeans++')
        ???
    
    ------------
    Attributes:
    ------------
    """
    def __init__(self, n_cluster=None, maximal_comps=60, n_init=20, kmeans_alg='k-means++'):
        self.n_cluster = n_cluster
        self.n_init = n_init
        self.kmeans_alg = kmeans_alg
        self._reset()
        
        if n_cluster is None:
            self.svd = TruncatedSVD(maximal_comps)
        
        
        
    def _reset(self):
        self.W = None
        self.labels_ = None
        self.laplacian = None
        
    def reset(self):
        """reset: to clear up the computed affinity, labels_, laplacian for the reusage
        of this class.
        """
        self._reset()
        
    def build_affinity_matrix(self, C):
        """_summary_build_affinity_matrix: Build up the affinity graph W according to the 
        self-expressive matrix C by |C| + |C.T|.

        Args:
            C (sparse matrix): the obtained self-expressiveness matrix 
        """
        C = normalize(C, 'l2')
        self.W = 0.5 * (np.absolute(C) + np.absolute(C.T))
        
    def build_laplacian(self):
        """build_laplacian: construct normalized graph laplacian for the further
        cluster estimation and spectral clustering.
        """
        
        assert self.W is not None, "Build up the affinity first, please!"
        
        laplacian = sparse.csgraph.laplacian(self.W, normed=True)
        self.laplacian = sparse.identity(laplacian.shape[0]) - laplacian
    
    def find_component(self):
        """find_component: estimate the number of clusters by finding the number of 
        connected componets in affinity graph.

        Returns:
        --------
            int: the estimated number of clusters (the  number of connected components
                in the obtained affinity graph)
        """
        assert self.laplacian is not None, "Construct the graph laplacian, please!"
            
        self.svd.fit(self.W)
        d = np.diff(self.svd.singular_values_)
        print()
        print(len(self.svd.singular_values_))
        r = np.argmin(d)+1
        return r
    
    def spectral_clustering(self):
        """spectral_clustering: apply spectral clustering to obtain the labels stored in
            self.labels_
        """
        
        if self.n_cluster is None:
            r = self.find_component()
        else:
            r = self.n_cluster
        
        _, vec = sparse.linalg.eigsh(self.laplacian, 
                                        k=r, sigma=None, which='LA')
        embedding= normalize(vec)
        
        _, self.labels_, _ = cluster.k_means(embedding, r, 
                                             init=self.kmeans_alg, n_init=self.n_init)
        
    
    def fit(self, X):
        """The full function of subspace clustering methods, which must be
        overwrote in the subclass.

        Args:
            X (numpy.Array): input dataset

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
        
class SSC(BaseSC):
    """The sparse subspace clustering method that solve the following optimization
    problem for noiseless data:
        min_{c} ||c||+1   s.t. x_j= X_{-j}c
    and the lasso version for noisy data:
        min_{c} lam/2 ||y - c X_{-j} ||_2^2 + ||c||_1
    The optimizations including noiseless and noisy version are 
    -------
    Parameters:
    ---------
        is_noisy (bool. default: True):
            If True, the lasso optimization is used. If False, the original version is 
            adopted for the l_1 optimization. 
        gamma (float. default: 500):
            lam = max_{j\not i}||x_j^Tx_i||_2/gamma.
    """
    def __init__(self, n_cluster=None, maximal_comps=60, n_init=20, kmeans_alg='k-means++', is_noisy=True, gamma= 500):
        super().__init__(n_cluster, maximal_comps, n_init, kmeans_alg)
        self.is_noisy = is_noisy
        self.gamma = gamma
        
    def SSC_lasso_(self, X, n_nonzero=50):
        n_samples = X.shape[0]
        rows = np.zeros(n_samples * n_nonzero)
        cols = np.zeros(n_samples * n_nonzero)
        vals = np.zeros(n_samples * n_nonzero)
        curr_pos = 0
        for i in range(n_samples):
            y = X[i, :].copy().reshape(1, -1)
            X[i, :] = 0
            
            coh = np.delete(np.absolute(np.dot(X, y.T)), i)
            lam = np.amax(coh) / self.gamma
            
            if self.is_noisy:
                c = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(X.T), 
                                lambda1=lam, lambda2=0)
            else:
                c = spams.lasso(
                    np.asfortranarray(y.T), D=np.asfortranarray(X.T), mode=1,
                    lambda1=1e-3
                )
            c = np.asarray(c.todense()).T[0]
            
            index = np.flatnonzero(c)
            if index.size > n_nonzero:
                #  warnings.warn("The number of nonzero entries in sparse subspace clustering exceeds n_nonzero")
                index = index[np.argsort(-np.absolute(c[index]))[0:n_nonzero]]
            rows[curr_pos:curr_pos + len(index)] = i
            cols[curr_pos:curr_pos + len(index)] = index
            vals[curr_pos:curr_pos + len(index)] = c[index]
            curr_pos += len(index)
            
            X[i, :] = y
            
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
            
        
    def fit(self, X):
        """_summary_

        Args:
            X (numpy.Array): The input data matrix
        """
        self.build_affinity_matrix(
                self.SSC_lasso_(X)
            )
    
        self.build_laplacian()
        self.spectral_clustering()
        
        
      
        
class TSC(BaseSC):
    """_summary_

    Args:
        BaseSC (_type_): _description_
    """
    def __init__(self, n_cluster=None, maximal_comps=60, n_init=20, kmeans_alg='k-means++', q=10):
        super().__init__(n_cluster, maximal_comps, n_init, kmeans_alg)
        self.q = q
        
    def _TSC(self, X):
        """The implementation of TSC algorithm. 

        Args:
            X (numpy.Array): The input data

        Returns:
             sparse.csr_matrix: The self-expressiveness matrix generated by TSC
        """
        X = X.T
        X = normalize(X, axis=0)
        m, N = X.shape
        rows = np.zeros(N*self.q)
        cols = np.zeros(N*self.q)
        vals = np.zeros(N*self.q)
        curr_pos = 0
        for i in progressbar.progressbar(range(N)):
            corvex = np.abs(np.matmul(X.T, X[:, i])) 
            corvex[corvex>=1] = 1
            corvex[i] = 0
            el = np.sort(corvex)
            order = np.argsort(corvex)
            index = order[-self.q:]
            vals[curr_pos:curr_pos + len(index)] = np.exp(-2*np.arccos(el[-self.q:]))
            rows[curr_pos:curr_pos + len(index)] = i
            cols[curr_pos:curr_pos + len(index)] = index
            curr_pos += len(index)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))
        
    def fit(self, X):
        self.build_affinity_matrix(
            self._TSC(X)
        )
        
        self.build_laplacian()
        self.spectral_clustering()
        
