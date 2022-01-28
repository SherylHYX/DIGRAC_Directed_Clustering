import csv
import random
import math

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import numpy.random as rnd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split


def write_log(args, path):
    with open(path+'/settings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for para in args:
            writer.writerow([para, args[para]])
    return


def normalize_torch_adj(A):
    """Row-normalize torch Tensor matrix"""
    rowsum = np.array(A.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D_half = torch.diag(torch.FloatTensor(r_inv_sqrt))

    return torch.mm(torch.mm(D_half, A), D_half)


def DSBM(N, K, p, F, size_ratio=1, sizes='fix_ratio'):
    """A directed stochastic block model graph generator.
    Args:
        N: (int) Number of nodes.
        K: (int) Number of clusters.
        p: (float) Sparsity value, edge probability.
        F : meta-graph adjacency matrix to generate edges
        size_ratio: Only useful for sizes 'fix_ratio', with the largest size_ratio times the number of nodes of the smallest.
        sizes: (string) How to generate community sizes:
            'uniform': All communities are the same size (up to rounding).
            'fix_ratio': The communities have number of nodes multiples of each other, with the largest size_ratio times the number of nodes of the smallest.
            'random': Nodes are assigned to communities at random.
            'uneven': Communities are given affinities uniformly at random, and nodes are randomly assigned to communities weighted by their affinity.

    Returns:
        a,c where a is a sparse N by N matrix of the edges, c is an array of cluster membership.
    """

    assign = np.zeros(N, dtype=int)

    size = [0] * K

    if sizes == 'uniform':
        perm = rnd.permutation(N)
        size = [math.floor((i + 1) * N / K) - math.floor((i) * N / K)
                for i in range(K)]
        labels = []
        for i, s in enumerate(size):
            labels.extend([i]*s)
        labels = np.array(labels)
        # permutation
        assign = labels[perm]
    elif sizes == 'fix_ratio':
        perm = rnd.permutation(N)
        if size_ratio > 1:
            ratio_each = np.power(size_ratio, 1/(K-1))
            smallest_size = math.floor(
                N*(1-ratio_each)/(1-np.power(ratio_each, K)))
            size[0] = smallest_size
            if K > 2:
                for i in range(1, K-1):
                    size[i] = math.floor(size[i-1] * ratio_each)
            size[K-1] = N - np.sum(size)
        else:  # degenerate case, equaivalent to 'uniform' sizes
            size = [math.floor((i + 1) * N / K) -
                    math.floor((i) * N / K) for i in range(K)]
        labels = []
        for i, s in enumerate(size):
            labels.extend([i]*s)
        labels = np.array(labels)
        # permutation
        assign = labels[perm]

    elif sizes == 'random':
        for i in range(N):
            assign[i] = rnd.randint(0, K)
            size[assign[i]] += 1
        perm = [x for clus in range(K) for x in range(N) if assign[x] == clus]

    elif sizes == 'uneven':
        probs = rnd.ranf(size=K)
        probs = probs / probs.sum()
        for i in range(N):
            rand = rnd.ranf()
            cluster = 0
            tot = 0
            while rand > tot:
                tot += probs[cluster]
                cluster += 1
            assign[i] = cluster - 1
            size[cluster - 1] += 1
        perm = [x for clus in range(K) for x in range(N) if assign[x] == clus]
        print('Cluster sizes: ', size)

    else:
        raise ValueError('please select valid sizes')

    g = nx.stochastic_block_model(sizes=size, p=p*F, directed=True)
    A = nx.adjacency_matrix(g)[perm][:, perm]

    return A, assign


def fix_network(A, labels):
    '''find the largest connected component and then increase the degree of nodes with low degrees

    Parameters
    ----------
    A : scipy sparse adjacency matrix
    labels : an array of labels of the nodes in the original network

    Returns
    -------
    fixed-degree and connected network, submatrices of A, and subarray of labels
    '''
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph)
    largest_cc = max(nx.weakly_connected_components(G))
    A_new = A[list(largest_cc)][:, list(largest_cc)]
    labels_new = labels[list(largest_cc)]
    G0 = nx.from_scipy_sparse_matrix(A_new, create_using=nx.DiGraph)
    flag = True
    iter_num = 0
    while flag and iter_num < 10:
        iter_num += 1
        remove = [node for node, degree in dict(
            G0.degree()).items() if degree <= 1]
        keep = np.array([node for node, degree in dict(
            G0.degree()).items() if degree > 1])
        if len(remove):
            print(len(G0.nodes()), len(remove))
            G0.remove_nodes_from(remove)
        else:
            flag = False
    print('After {} iteration(s), we extract lcc with degree at least 2 for each node to have network with {} nodes, compared to {} nodes before.'.format(
        iter_num, len(keep), A.shape[0]))
    A_new = A[keep][:, keep]
    labels_new = labels[keep]
    return A_new, labels_new


def split_labels(labels):
    nclass = torch.max(labels) + 1
    labels_split = []
    labels_split_numpy = []
    for i in range(nclass):
        labels_split.append(torch.nonzero((labels == i)).view([-1]))
    for i in range(nclass):
        labels_split_numpy.append(labels_split[i].cpu().numpy())
    labels_split_dif = []
    for i in range(nclass):
        dif_type = [x for x in range(nclass) if x != i]
        labels_dif = torch.cat([labels_split[x] for x in dif_type])
        labels_split_dif.append(labels_dif)
    return nclass, labels_split, labels_split_numpy, labels_split_dif


def getClassMean(nclass, labels_split, logits):
    class_mean = torch.cat([torch.mean(
        logits[labels_split[x]], dim=0).view(-1, 1) for x in range(nclass)], dim=1)
    return class_mean


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)\



def load_model(net, name):
    state_dict = torch.load(name)
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)



def get_powers_sparse(A, hop=3, tau=0.1):
    '''
    function to get adjacency matrix powers
    inputs:
    A: directed adjacency matrix
    hop: the number of hops that would like to be considered for A to have powers.
    tau: the regularization parameter when adding self-loops to an adjacency matrix, i.e. A -> A + tau * I, 
        where I is the identity matrix. If tau=0, then we have no self-loops to add.
    output: (torch sparse tensors)
    A_powers: a list of A powers from 0 to hop
    '''
    A_powers = []

    shaping = A.shape
    adj0 = sp.eye(shaping[0])

    A_bar = normalize(A+tau*adj0, norm='l1')  # l1 row normalization
    tmp = A_bar.copy()
    adj0_new = sp.csc_matrix(adj0)
    ind_power = A.nonzero()
    A_powers.append(torch.sparse_coo_tensor(torch.LongTensor(
        adj0_new.nonzero()), torch.FloatTensor(adj0_new.data), shaping))
    A_powers.append(torch.sparse_coo_tensor(torch.LongTensor(
        ind_power), torch.FloatTensor(np.array(tmp[ind_power]).flatten()), shaping))
    if hop > 1:
        A_power = A.copy()
        for h in range(2, int(hop)+1):
            tmp = tmp.dot(A_bar)  # get A_bar powers
            A_power = A_power.dot(A)
            ind_power = A_power.nonzero()  # get indices for M matrix
            tmp = tmp.dot(A_bar)  # get A_bar powers
            A_powers.append(torch.sparse_coo_tensor(torch.LongTensor(
                ind_power), torch.FloatTensor(np.array(tmp[ind_power]).flatten()), shaping))

            # A_powers.append(torch.sparse_coo_tensor(torch.LongTensor(tmp.nonzero()), torch.FloatTensor(tmp.data), shaping))
    return A_powers



def BA(N, K, p, F, size_ratio=1, sizes='fix_ratio'):
    """A directed BA graph generator.
    Args:
        N: (int) Number of nodes.
        K: (int) Number of clusters.
        p: (float) Sparsity value, edge probability.
        F : meta-graph adjacency matrix
        size_ratio: Only useful for sizes 'fix_ratio', with the largest size_ratio times the number of nodes of the smallest.
        pout: (float) Sparsity value between clusters. By default, take the same value as pin.
        sizes: (string) How to generate community sizes:
            'uniform': All communities are the same size (up to rounding).
            'fix_ratio': The communities have number of nodes multiples of each other, with the largest size_ratio times the number of nodes of the smallest.
            'random': Nodes are assigned to communities at random.
            'uneven': Communities are given affinities uniformly at random, and nodes are randomly assigned to communities weighted by their affinity.
    Returns:
        a,c where a is a sparse N by N matrix of the edges, c is an array of cluster membership.
    """

    assign = np.zeros(N, dtype=int)

    size = [0] * K

    if sizes == 'uniform':
        perm = rnd.permutation(N)
        size = [math.floor((i + 1) * N / K) - math.floor((i) * N / K)
                for i in range(K)]
        tot = size[0]
        cluster = 0
        i = 0
        while i < N:
            if tot == 0:
                cluster += 1
                tot += size[cluster]
            else:
                tot -= 1
                assign[perm[i]] = cluster
                i += 1
    elif sizes == 'fix_ratio':
        perm = rnd.permutation(N)
        if size_ratio > 1:
            ratio_each = np.power(size_ratio, 1/(K-1))
            smallest_size = math.floor(
                N*(1-ratio_each)/(1-np.power(ratio_each, K)))
            size[0] = smallest_size
            if K > 2:
                for i in range(1, K-1):
                    size[i] = math.floor(size[i-1] * ratio_each)
            size[K-1] = N - np.sum(size)
        else:  # degenerate case, equaivalent to 'uniform' sizes
            size = [math.floor((i + 1) * N / K) -
                    math.floor((i) * N / K) for i in range(K)]
        tot = size[0]
        cluster = 0
        i = 0
        while i < N:
            if tot == 0:
                cluster += 1
                tot += size[cluster]
            else:
                tot -= 1
                assign[perm[i]] = cluster
                i += 1

    elif sizes == 'random':
        for i in range(N):
            assign[i] = rnd.randint(0, K)
            size[assign[i]] += 1
        perm = [x for clus in range(K) for x in range(N) if assign[x] == clus]

    elif sizes == 'uneven':
        probs = rnd.ranf(size=K)
        probs = probs / probs.sum()
        for i in range(N):
            rand = rnd.ranf()
            cluster = 0
            tot = 0
            while rand > tot:
                tot += probs[cluster]
                cluster += 1
            assign[i] = cluster - 1
            size[cluster - 1] += 1
        perm = [x for clus in range(K) for x in range(N) if assign[x] == clus]
        print('Cluster sizes: ', size)

    else:
        raise ValueError('please select valid sizes')

    m = math.ceil(N * p / 2)

    G = nx.generators.random_graphs.barabasi_albert_graph(N, m)  # undirected
    A_G = nx.adjacency_matrix(G)[perm][:, perm]
    original_A = A_G.todense()
    A = original_A.copy()

    accum_x, accum_y = 0, 0

    for i in range(K):
        accum_y = accum_x + size[i]
        for j in range(i, K):
            x, y = np.where(
                original_A[accum_x:size[i]+accum_x, accum_y:size[j]+accum_y])
            try:
                x1, x2, y1, y2 = train_test_split(x, y, test_size=F[i, j])

                A[x1+accum_x, y1+accum_y] = 0
                A[y2+accum_y, x2+accum_x] = 0
            except ValueError:
                print('Empty split!')

            accum_y += size[j]

        accum_x += size[i]
    # label assignment based on parameter size
    label = []
    for i, s in enumerate(size):
        label.extend([i]*s)
    label = np.array(label)
    # permutation
    label = label[perm]
    A = A[perm][:, perm]
    original_A = original_A[perm][:, perm]
    return np.array(original_A), np.array(A), label


def hermitian_feature(A, num_clusters):
    """ create Hermitian feature  (rw normalized)
    inputs:
    A : adjacency matrix
    num_clusters : number of clusters

    outputs: 
    features_SVD : a feature matrix from SVD of Hermitian matrix
    """
    H = (A-A.transpose()) * 1j
    H_abs = np.abs(H)  # (np.real(H).power(2) + np.imag(H).power(2)).power(0.5)
    D_abs_inv = sp.diags(1/np.array(H_abs.sum(1))[:, 0])
    H_rw = D_abs_inv.dot(H)
    u, s, vt = sp.linalg.svds(H_rw, k=num_clusters)
    features_SVD = np.concatenate((np.real(u), np.imag(u)), axis=1)
    scaler = StandardScaler().fit(features_SVD)
    features_SVD = scaler.transform(features_SVD)
    return features_SVD


def meta_graph_generation(F_style='cyclic', K=4, eta=0.05, ambient=False, fill_val=0.5):
    if eta == 0:
        eta = -1
    F = np.eye(K) * 0.5
    # path
    if F_style == 'path':
        for i in range(K-1):
            j = i + 1
            F[i, j] = 1 - eta
            F[j, i] = 1 - F[i, j]
    # cyclic structure
    elif F_style == 'cyclic':
        if K > 2:
            if ambient:
                for i in range(K-1):
                    j = (i + 1) % (K-1)
                    F[i, j] = 1 - eta
                    F[j, i] = 1 - F[i, j]
            else:
                for i in range(K):
                    j = (i + 1) % K
                    F[i, j] = 1 - eta
                    F[j, i] = 1 - F[i, j]
        else:
            if ambient:
                F = np.array([[0.5, 0.5], [0.5, 0.5]])
            else:
                F = np.array([[0.5, 1-eta], [eta, 0.5]])
    # cyclic structure but with reverse directions
    elif F_style == 'cyclic_reverse':
        if K > 2:
            if ambient:
                for i in range(K-1):
                    j = (i + 1) % (K-1)
                    F[i, j] = eta
                    F[j, i] = 1 - F[i, j]
            else:
                for i in range(K):
                    j = (i + 1) % K
                    F[i, j] = eta
                    F[j, i] = 1 - F[i, j]
        else:
            if ambient:
                F = np.array([[0.5, 0.5], [0.5, 0.5]])
            else:
                F = np.array([[0.5, eta], [1-eta, 0.5]])
    # complete meta-graph structure
    elif F_style == 'complete':
        if K > 2:
            for i in range(K-1):
                for j in range(i+1, K):
                    direction = np.random.randint(
                        2, size=1)  # random direction
                    F[i, j] = direction * (1 - eta) + (1-direction) * eta
                    F[j, i] = 1 - F[i, j]
        else:
            F = np.array([[0.5, 1-eta], [eta, 0.5]])
    # core-periphery L shape meta-graph structure
    elif F_style == 'L':
        if K < 3:
            raise Exception("Sorry, L shape requires K at least 3!")
        p1 = 1 - eta
        p2 = eta
        if ambient:
            if K < 4:
                raise Exception(
                    "Sorry, L shape with ambient nodes requires K at least 4!")
            F = np.ones([K, K])*p2
            F[:-2, 1] = p1
            F[-3, 1:] = p1
        else:
            F = np.ones([K, K])*p2
            F[:-1, 1] = p1
            F[-2, 1:] = p1
    elif F_style == 'star':
        if K < 3:
            raise Exception("Sorry, star shape requires K at least 3!")
        if ambient and K < 4:
            raise Exception(
                "Sorry, star shape with ambient nodes requires K at least 4!")
        center_ind = math.floor((K-1)/2)
        F[center_ind, ::2] = eta  # only even indices
        F[center_ind, 1::2] = 1-eta  # only odd indices
        F[::2, center_ind] = 1-eta
        F[1::2, center_ind] = eta
    elif F_style == 'multipartite':
        if K < 3:
            raise Exception("Sorry, multipartite shape requires K at least 3!")
        if ambient:
            if K < 4:
                raise Exception(
                    "Sorry, multipartite shape with ambient nodes requires K at least 4!")
            G1_ind = math.ceil((K-1)/9)
            G2_ind = math.ceil((K-1)*3/9)+G1_ind
        else:
            G1_ind = math.ceil(K/9)
            G2_ind = math.ceil(K*3/9)+G1_ind
        F[:G1_ind, G1_ind:G2_ind] = eta
        F[G1_ind:G2_ind, G2_ind:] = eta
        F[G2_ind:, G1_ind:G2_ind] = 1-eta
        F[G1_ind:G2_ind, :G1_ind] = 1-eta
    elif F_style == 'center':
        center_ind = math.floor((K-1)/2)
        F[center_ind] = eta
        F[:, center_ind] = 1-eta
        F[center_ind, center_ind] = 0.5
    else:
        raise Exception("Sorry, please give correct F style string!")
    if ambient:
        F[-1, :] = 0
        F[:, -1] = 0
    F[F == 0] = fill_val
    F[F == -1] = 0
    F[F == 2] = 1
    return F


def scipy_sparse_to_torch_sparse(A):
    A = sp.csr_matrix(A)
    return torch.sparse_coo_tensor(torch.LongTensor(A.nonzero()), torch.FloatTensor(A.data), A.shape)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
