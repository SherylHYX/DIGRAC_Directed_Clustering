import os
from argparse import Namespace

import networkx as nx
import oslom
import numpy as np
import leidenalg as la
import igraph as ig
from infomap import Infomap


def InfoMap(A):
    G = nx.convert_matrix.from_scipy_sparse_matrix(A ,create_using=nx.DiGraph)
    im = Infomap(silent=True)
    mapping = im.add_networkx_graph(G)
    mapping
    im.run()
    results = np.array([[node.module_id, mapping[node.node_id]] for node in im.nodes])
    return results[:,0][np.argsort(results[:,1])]

def leidenalg_modularity(A):
    G = nx.convert_matrix.from_scipy_sparse_matrix(A ,create_using=nx.DiGraph)
    g = ig.Graph.from_networkx(G)
    partition = la.find_partition(g, la.ModularityVertexPartition)
    return np.array(partition.membership)

def louvain_modularity(A):
    G = nx.convert_matrix.from_scipy_sparse_matrix(A ,create_using=nx.DiGraph)
    g = ig.Graph.from_networkx(G)
    optimiser = la.Optimiser()
    partition = la.ModularityVertexPartition(g)
    partition_agg = partition.aggregate_partition()
    while optimiser.move_nodes(partition_agg) > 0:
        partition.from_coarse_partition(partition_agg)
        partition_agg = partition_agg.aggregate_partition()
    return np.array(partition.membership)

def OSLOM(A):
    args = Namespace()
    args.min_cluster_size = 0
    args.oslom_exec = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), 'OSLOM2',oslom.DEF_OSLOM_EXEC)
    args.oslom_args = oslom.DEF_OSLOM_ARGS
    G = nx.convert_matrix.from_scipy_sparse_matrix(A ,create_using=nx.DiGraph)
    edges = G.edges.data("weight", default=1)
    clusters = oslom.run_in_memory(args, edges)
    cluster_list = clusters[0]['clusters']
    cluster_sizes = [len(cluster_list[i]['nodes']) for i in range(clusters[0]['num_found'])]
    print('The maximum cluster size is {}.'.format(max(cluster_sizes)))
    return cluster_list

def directed_modularity(A, labels):
    w = A.sum()
    K = int(labels.max() - labels.min() + 1)
    Q = 0
    for k in range(K):
        w_kk_norm = A[labels==k][:,labels==k].sum()/w
        w_k_in_norm = A[labels!=k][:,labels==k].sum()/w
        w_k_out_norm = A[labels==k][:,labels!=k].sum()/w
        Q += w_kk_norm - w_k_in_norm * w_k_out_norm
    return Q

'''
below are test code used in python console:
import os
from argparse import Namespace

import networkx as nx
import oslom
import numpy as np
import leidenalg as la
import igraph as ig
from infomap import Infomap

from src.utils import DSBM, meta_graph_generation
F = meta_graph_generation()
A, labels = DSBM(1000, 4, 0.1, F)
args = Namespace(); args.min_cluster_size = 0; args.oslom_args = oslom.DEF_OSLOM_ARGS
args.oslom_exec =  os.path.join(os.path.dirname(os.path.realpath(
        __file__)), 'OSLOM2',oslom.DEF_OSLOM_EXEC)

G = nx.convert_matrix.from_scipy_sparse_matrix(A ,create_using=nx.DiGraph); edges = G.edges.data("weight", default=1)
clusters = oslom.run_in_memory(args, edges)
'''