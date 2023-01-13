import numpy as np 
from sklearn.preprocessing import normalize


def convert_labs(l, d):
    for i in range(len(l)):
        l[i] = d[l[i]]
    return l

def read_raw_data(path_anno, path_data, p=0.9):
    with open(path_anno, "r") as f:
        anno = f.readlines()
        anno = anno[1:]
        d = dict()
        for row in anno:
            d[row.split()[0]] = row.split()[2]+ row.split()[3]
        l = set(d.values())
        # print(len(l))
        m = dict(zip(l, range(len(l))))
        for i in d:
            d[i] = m[d[i]]

    with open(path_data, "r") as f:
        data = f.readlines()
        countt = 0
        # index = data[0].split()
        # index = index[1:]
        labs = np.array(convert_labs(data[0].split()[1:], d))
        data = data[1:]
        s_data = []
        for i in range(len(data)):
            line_data = data[i].split()[1:]
            if line_data.count('0')/len(line_data) < p:
                countt +=1
                s_data.append([int(line_data[i]) for i in range(len(line_data))])
                
        s_data = np.array(s_data).T 
        s_data = normalize(s_data)
        
    L_d = {'label': labs, 'data':s_data}
    # print(countt)
    # print(L_d['label'])
    # print(L_d['data'].shape)
    return L_d
            
            
def distribute_with_labs(s_data, labs, labels, NUM_points_per_sub):
    ambient_dim = s_data.shape[1]
    client_data = np.empty((NUM_points_per_sub* len(labels), ambient_dim))
    client_label = np.empty(NUM_points_per_sub* len(labels), dtype=int)
    for i in range(len(labels)):
        label = labels[i]
        index = np.arange(len(labs))
        while len(index[labs==label]) < NUM_points_per_sub:
            label = np.random.choice(labs)
    
        idx = np.random.choice(index[labs==label], size=NUM_points_per_sub, replace=False)
        data_sub = s_data[idx]
        
        indexes = i*NUM_points_per_sub
        client_data[indexes:(NUM_points_per_sub+indexes), :] = data_sub
        client_label[indexes:(NUM_points_per_sub+indexes)] = labels[i]
    return client_data, client_label
        
    
def distribute_seq(s_data, labs, NUM_client, range_LocalK, NUM_points_per_sub):
    L=[]
    # L_num_clusters = []
    for client_i in range(NUM_client):
        device_d = {}
        num_sub = np.random.randint(*range_LocalK)
        labels = np.random.choice(labs, size=num_sub, replace=False)
        client_data, client_lab = distribute_with_labs(
            s_data, labs, labels, NUM_points_per_sub
        )
        device_d['data'] = client_data
        device_d['label'] = client_lab
        L.append(device_d)
    return L


if __name__ == "__main__":
    path_anno = 'E-MTAB-6678/meta_ss2.txt'
    path_data = 'E-MTAB-6678/raw_data_ss2.txt'
    p = 0.9
    range_LocalK = [3,4]
    num_clients= 400
    num_per = 50
    
    L_d  = read_raw_data(path_anno, path_data, p)
    L = distribute_seq(L_d['data'], L_d['label'], num_clients, range_LocalK, num_per)
    with open('gene_{0}_{1}_{2}_3.npy'.format(num_clients, num_per, p), 'wb') as f:
        np.save(f, L)
        print('Done !')
    