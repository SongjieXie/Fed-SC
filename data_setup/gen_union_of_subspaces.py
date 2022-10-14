import numpy as np
from scipy.linalg import orth
from sklearn.preprocessing import normalize

def gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, noise_level=0.0):
    """This funtion generates a union of subspaces under random model, i.e., 
    subspaces are independently and uniformly distributed in the ambient space,
    data points are independently and uniformly distributed on the unit sphere of each subspace

    Parameters
    -----------
    ambient_dim : int
        Dimention of the ambient space
    subspace_dim : int
        Dimension of each subspace (all subspaces have the same dimension)
    num_subspaces : int
        Number of subspaces to be generated
    num_points_per_subspace : int
        Number of data points from each of the subspaces
    noise_level : float
        Amount of Gaussian noise on data
		
    Returns
    -------
    data : shape (num_subspaces * num_points_per_subspace) by ambient_dim
        Data matrix containing points drawn from a union of subspaces as its rows
    label : shape (num_subspaces * num_points_per_subspace)
        Membership of each data point to the subspace it lies in
    """

    data = np.empty((num_points_per_subspace* num_subspaces, ambient_dim))
    label = np.empty(num_points_per_subspace * num_subspaces, dtype=int)
  
    for i in range(num_subspaces):
        basis = np.random.normal(size=(ambient_dim, subspace_dim))
        basis = orth(basis)
        coeff = np.random.normal(size=(subspace_dim, num_points_per_subspace))
        coeff = normalize(coeff, norm='l2', axis=0, copy=False)
        data_per_subspace = np.matmul(basis, coeff).T

        base_index = i*num_points_per_subspace
        
        data[(0+base_index):(num_points_per_subspace+base_index), :] = data_per_subspace
        label[0+base_index:num_points_per_subspace+base_index,] = i

    data += np.random.normal(size=(num_points_per_subspace * num_subspaces, ambient_dim)) * noise_level
  
    return data, label

def generate_subspaces(ambient_dim, subspace_dim, num_subspaces):
    """Generate subspaces of given dimensions randomly and independently in an ambient space.

    Args:
        ambient_dim (int): Dimenison of ambient space, i.e., R^n.
        subspace_dim (int or list): Dimensions of generated subspaces.
            If this parameter is int, all subspace have same dimensionality. If it is a list,
            the dimension of each subspace is in the range of list.
        num_subspaces (int): Number of generated subspaces.

    Returns:
        list of numpy.Array: The list of orthonormal basis of generated subspaces. 
    """
    if isinstance(subspace_dim, int):
        subspace_dims = subspace_dim*np.ones(num_subspaces)
    elif isinstance(subspace_dim, list):
        subspace_dims = np.random.randint(*subspace_dim, size=num_subspaces)
    
    basis_list = []
    for dim in subspace_dims:
        dim = int(dim)
        basis = np.random.normal(size=(ambient_dim, dim))
        basis = orth(basis)
        basis_list.append(basis)
        
    return basis_list

def generate_pts(basis_list, labs, num_pts_sub, ambient_dim, noise=0.0):
    """Generate data points uniformly in the given subspaces with unit norm.

    Args:
        basis_list (list of numpy.Array): List of basis of subspaces.
        labs (list): Corresponding labels of the subspaces.
        num_pts_sub (int): Number of data points generated from each subspace.
        ambient_dim (int): Dimension of ambient space.
        noise (float, optional): Gaussian noise added to data points. Defaults to 0.0.

    Returns:
        data: The generated data points.
        labels: The corresponding labels. 
    """
    assert len(labs) == len(basis_list)
    data = np.empty((num_pts_sub* len(basis_list), ambient_dim))
    label = np.empty(num_pts_sub* len(labs), dtype=int)
    for i in range(len(labs)):
        basis, lab = basis_list[i], labs[i]
        sub_dim = basis.shape[1]
        coeff = np.random.normal(size=(sub_dim, num_pts_sub))
        coeff = normalize(coeff, norm='l2', axis=0, copy=False)
        data_sub = np.matmul(basis, coeff).T
        
        indexes = i*num_pts_sub
        data[indexes:(num_pts_sub+indexes), :] = data_sub
        label[indexes:(num_pts_sub+indexes)] = lab
    
    data += np.random.normal(size=data.shape)* noise
    return data, label
        
        
 