import numpy as np
import torch
from .gen_union_of_subspaces import generate_subspaces, generate_pts
from .mydatasets import getdata
from torch.utils.data.sampler import SubsetRandomSampler
from kymatio import Scattering2D
from sklearn.decomposition import PCA
from collections.abc import Iterable


def pca_D_reduction(X, dim):
    pca = PCA(n_components=dim)
    return pca.fit_transform(X)

def preprocess(X, img_size, device):
    scattering = Scattering2D(J=3, shape=img_size).to(device)
    X = X.to(device)
    X = scattering(X)
    X = X.cpu().numpy().reshape(X.shape[0], X.shape[2], -1)
    img_norm = np.linalg.norm(X, ord=np.inf, axis=2, keepdims=True)
    X = (X/img_norm).reshape(X.shape[0], -1)
    
    return X

def con_process(X):
    return X.flatten(start_dim=2, end_dim=3).reshape(X.shape[0],-1)
    

def distribute_syn(NUM_client, AMB_D, SUB_D, NUM_sub, NUM_pts_sub, range_LocalK, noise=0.0):
    l_basis = generate_subspaces(AMB_D, SUB_D, NUM_sub)
    labels = np.array(range(NUM_sub))
    L=[]
    # L_num_clusters = []
    for client_i in range(NUM_client):
        device_d = {}
        num_sub = np.random.randint(*range_LocalK)
        labs = np.random.choice(labels, size=num_sub, replace=False)
        client_data, client_lab = generate_pts(
            [l_basis[i] for i in labs],
            labels[labs], NUM_pts_sub, AMB_D, noise=noise
        )
        device_d['data'] = client_data
        device_d['label'] = client_lab
        L.append(device_d)
        # L_num_clusters.append(num_sub)
    return L
        
def get_loaders(datas, batchs, set_labs):
    L = []
    for lab in set_labs:
        torch_target = torch.tensor(datas.targets) if isinstance(datas.targets, list) else datas.targets
        index = torch.where(torch_target == lab)[0] 
        s = SubsetRandomSampler(index)
        batchsize = int(batchs[lab]) if isinstance(batchs, Iterable) else batchs
        L.append(
            torch.utils.data.DataLoader(datas, batch_size=batchsize, sampler = s)
        )
    return L 


def get_imgs(L_loaders, set_labs, img_size, device):
    L_labs = []
    L_data = []
    for loader in L_loaders:
        data_imgs, labels = next(iter(loader))
        L_data.append(data_imgs)
        L_labs.append(labels)
    data = torch.cat(L_data, 0)
    labels = torch.cat(L_labs, 0).numpy()
    precossed_data = preprocess(data, img_size, device)
    L = []
    for lab in set_labs:
        indices = labels == lab
        L.append(
            precossed_data[indices]
        )
    return L

    
def distribute_real(name, NUM_client, NUM_pts_sub, range_LocalK, device, img_size=(32,32), is_con=True):
    datas = getdata(name, resize=img_size, train=True)
    set_labs = list(set(datas.targets)) if isinstance(datas.targets, list) else list(set(datas.targets.tolist()))
    L_loaders = get_loaders(datas, NUM_pts_sub, set_labs)
    L_iterators = [iter(l) for l in L_loaders]
    L = []
    for client_i in range(NUM_client):
        device_d = {'data':[], 'label': []}
        num_sub = np.random.randint(*range_LocalK)
        labs = np.random.choice(set_labs, size=num_sub, replace=True)
        # print('The {0}th client select in set with length {1}'.format(client_i, len(set_labs)))
        for lab in labs:
            flag = 1
            while flag:
                try:
                    data, labels = next(L_iterators[lab])
                    if data.shape[0] == NUM_pts_sub:
                        flag = 0
                    else:
                        L_iterators[lab] = iter(L_loaders[lab])
                except StopIteration:
                    # set_labs = np.delete(set_labs, np.where(set_labs==lab))
                    # print('The {0}th client reset the loader {1}'.format(client_i, labs))
                    L_iterators[lab] = iter(L_loaders[lab])
                    lab = np.random.choice(set_labs)
                    
                    
            device_d['data'].append(data)
            device_d['label'].append(labels)
        if is_con:
            device_d['data'] = con_process(torch.cat(device_d['data'], dim=0))
        else:
            device_d['data'] = preprocess(torch.cat(device_d['data'], dim=0),img_size, device)
        # device_d['data'] = torch.cat(device_d['data'], dim=0)
        device_d['label'] = torch.cat(device_d['label'], dim=0).numpy()
        L.append(device_d)
    return L

def distribute_real_cattering(name, NUM_client, NUM_pts_sub, range_LocalK, device, img_size=(32, 32)):
    datas = getdata(name, resize=img_size, train=True)
    set_labs = list(set(datas.targets)) if isinstance(datas.targets, list) else list(set(datas.targets.tolist()))
    L_num_sub = [np.random.randint(*range_LocalK) for i in range(NUM_client)]
    L_labs = [np.random.choice(set_labs, size=num_sub, replace=False) for num_sub in L_num_sub]
    batch_sizes = np.zeros(len(set_labs))
    for labs_client in L_labs:
        for l in labs_client:
            batch_sizes[l] += 1
    batch_sizes = NUM_pts_sub*batch_sizes
            
    L_loaders = get_loaders(datas, batch_sizes, set_labs)
    L_imgs = get_imgs(L_loaders, set_labs, img_size, device=device)
    L_pos = [0 for i in range(len(set_labs))]
    L = []
    for client_i in range(NUM_client):
        device_d = {'data':[], 'label': []}
        assert len(L_labs) == NUM_client
        labs = L_labs[client_i]
        for lab in labs:
            data = L_imgs[lab][L_pos[lab]:L_pos[lab]+NUM_pts_sub]
            labels = np.array([lab for i in range(data.shape[0])])
            L_pos[lab] += data.shape[0]
            device_d['data'].append(data)
            device_d['label'].append(labels)
        device_d['data'] = np.vstack(device_d['data'])
        device_d['label'] = np.hstack(device_d['label'])
        L.append(device_d)
    return L
    

def merge_all(L):
    data = []
    labels = []
    for l in L:
        data.append(l['data'])
        labels.append(l['label'])
    return (np.vstack(data), np.hstack(labels))


if __name__ == "__main__":
    for kk in [10]:
        device = torch.device('cpu')
        data_name = 'EMNIST'
        num_clients = 400
        n_pts = 50
        range_localK = [kk, kk+1]
        L = distribute_real_cattering(data_name, num_clients, n_pts, range_localK, device)
        
        # L = distribute_real(data_name, num_clients, n_pts, range_localK, device)
        flag = True
        for l in L:
            if l['data'].shape[0] ==0 or l['data'].shape[0]%n_pts != 0:
                print("!!!!", l['data'].shape)
                # print(l['label'])
        if flag:
            print('Save the cattering..')   
            with open('origin_{0}_{1}_{2}.npy'.format(data_name, num_clients, kk), 'wb') as f:
                np.save(f, L)
            print('Done !')
        print(L[-1]['data'].shape)
    

    
    """ ============================== """
    # L = distribute_real('COIL100', num_clients, 200, range_localK, device)
    
    # L = distribute_real('MNIST', NUM_client=20, NUM_pts_sub=20, range_LocalK=[3,4], device=device)
    # print(len(L))
    # print(L[0]['data'].shape)
    # print(L[0]['label'])
    
    # print('===========================================')
    
    # L_syn = distribute_syn(100, 50, [2,10], 13, 20, range_LocalK=[2,4])
    # print(len(L_syn))
    # print(L_syn[0]['data'].shape)
    # print(L_syn[0]['label'])
    
    # d, l = merge_all(L)
    # print(d.shape)
    # print(l)
    
    
    # print(L_syn[1])
    
    # X = torch.randn(size=[7000, 1, 32, 32])
    # X_new = preprocess(X, [32,32], device)
    # print(X_new.shape)
    
    # import matplotlib.pyplot as plt
    # import torchvision.utils as vutils
    # def play_show(X_imgs, N=1, t=None):
    #     plt.figure(figsize=(6,6))
    #     plt.axis("off")
    #     plt.title(t)
    #     plt.imshow(np.transpose(vutils.make_grid(
    #         X_imgs, nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
    # device = torch.device('cpu')
    # L = distribute_real('EYaleB', NUM_client=2, NUM_pts_sub=10, range_LocalK=[3,4], device=device)
    # x, l = merge_all(L)
    # print('labs:', l.tolist())
    # print(x.shape)
    # play_show(x)
    # plt.show()
    
    
    

    