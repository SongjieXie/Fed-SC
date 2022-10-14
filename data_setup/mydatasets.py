import numpy as np
import torch 
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import Sampler


def get_transform(resize, name=None):
    if name == 'EYaleB':
        trans = transforms.Compose(
            [transforms.Grayscale(),
             transforms.Resize(resize),
             transforms.RandomAffine(0,shear=30 ),
             transforms.ToTensor()]
        )
    elif name == 'COIL100':
        trans = transforms.Compose(
            [
             transforms.Grayscale(),
             transforms.Resize(resize),
             transforms.ColorJitter(brightness=0.5, contrast=0.5),
             transforms.ToTensor()]
        )
    else:
        trans = transforms.Compose(
            [transforms.Resize(resize), 
             transforms.ToTensor()]
        )
    return trans

class SubsetSampler(Sampler):
    def __init__(self, data):
        self.data = data
        
    def __iter__(self):
        pass
    


root = r'/Users/songjiexie/Desktop/Projects/fed-SSC/Codes/reference-codes/subspace-clustering-e59c9c2cb911ae753a88c3548a1851f5058749ef/data/'
# def getdata(name, batchsize, resize=(32, 32), shuffle=False, n_worker=0, train=True, splits='balanced'):
def getdata(name, resize=(32, 32), train=True, splits='byclass'):
    if name == 'MNIST': # 10
        trans = get_transform(resize)
        dataset = datasets.MNIST(root, train=train, transform=trans, target_transform=None, download=True)
    elif name == 'EMNIST': ## byclass : 814,255 characters. 62 unbalanced classes. # EMNIST Balanced:  131,600 characters. 47 balanced classes. 
        trans = get_transform(resize)
        dataset = datasets.EMNIST(root,split=splits, train=train, transform=trans, target_transform=None, download=True) 
    elif name == 'FMNIST': # 10, 60,000 
        trans = get_transform(resize)
        dataset = datasets.FashionMNIST(root, train=train, transform=trans, target_transform=None, download=True)
    elif name == 'EYaleB': # 39, 66*39=2574
        trans = get_transform((48, 42), name=name)
        dataset = datasets.ImageFolder(root+'CroppedYale', transform=trans)
    elif name == 'COIL100': # 100, 72*100= 7200
        trans = get_transform((32,32), name=name)
        dataset = datasets.ImageFolder(root+'coil-100', transform=trans)
    else:
        raise ValueError('No such dataset ! !')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=n_worker)
    return dataset



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils
    def play_show(X_imgs, N=1, t=None):
        plt.figure(figsize=(6,6))
        plt.axis("off")
        plt.title(t)
        plt.imshow(np.transpose(vutils.make_grid(
            X_imgs, nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
    dataset = getdata('COIL100')
    datal = torch.utils.data.DataLoader(dataset, batch_size=360, shuffle=False, num_workers=1)
    x, l = next(iter(datal))
    print('labs:', l)
    print(x.shape)
    x_dis = torch.vstack(
        (
            x[2:2+10],
            x[2+72:2+72+10],
            x[2+72*2:2+72*2+10],
            x[2+72*3:2+72*3+10],
            x[2+72*4:2+72*4+10],
        )
    )
    
    play_show(x_dis)
    plt.show()
    
    # datas = getdata('COIL100', resize=(32, 32))
    # torch_target = torch.tensor(datas.targets)
    # index = torch.where(torch_target == 1)[0]
    # print(
    #     len(index)
    # )
    # print(datas.targets[int(index[3])])
    # print(set(datas.targets))
    
    
    

