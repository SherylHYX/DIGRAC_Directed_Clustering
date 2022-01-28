import random
from typing import Optional, Union, Tuple

import torch
import numpy as np
import scipy.sparse as sp
from texttable import Texttable
import latextable
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import adjusted_rand_score, accuracy_score, normalized_mutual_info_score

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def triplet_loss_InnerProduct(nclass, labels_split, labels_split_dif, logits, n_samples):
    n_sample = n_samples

    n_sample_class = max((int)(n_sample / nclass), 32)
    thre = 0.1
    loss = 0
    for i in range(nclass):
        # python2: xrange, python3: range
        try:
            randInds1 = random.choices(labels_split[i], k=n_sample_class)
            randInds2 = random.choices(labels_split[i], k=n_sample_class)

            feats1 = logits[randInds1]
            feats2 = logits[randInds2]
            randInds_dif = random.choices(
                labels_split_dif[i], k=n_sample_class)

            feats_dif = logits[randInds_dif]
        except IndexError:
            n_sample_class = 32
            randInds1 = random.choices(labels_split[i], k=n_sample_class)
            randInds2 = random.choices(labels_split[i], k=n_sample_class)

            feats1 = logits[randInds1]
            feats2 = logits[randInds2]
            randInds_dif = random.choices(
                labels_split_dif[i], k=n_sample_class)

            feats_dif = logits[randInds_dif]
        # inner product: same class inner product should > dif class inner product
        inner_products = torch.sum(torch.mul(feats1, feats_dif-feats2), dim=1)
        dists = inner_products + thre
        mask = dists > 0

        loss += torch.sum(torch.mul(dists, mask.float()))

    loss /= n_sample_class*nclass
    return loss


def triplet_loss_InnerProduct_alpha(nclass, labels_split, labels_split_dif, logits, n_samples, thre=0.1):
    n_sample = n_samples

    n_sample_class = max((int)(n_sample / nclass), 32)
    # thre = 0.1
    loss = 0
    for i in range(nclass):
        # python2: xrange, python3: range
        try:
            randInds1 = random.choices(labels_split[i], k=n_sample_class)
            randInds2 = random.choices(labels_split[i], k=n_sample_class)

            feats1 = logits[randInds1]
            feats2 = logits[randInds2]
            randInds_dif = random.choices(
                labels_split_dif[i], k=n_sample_class)

            feats_dif = logits[randInds_dif]
        except IndexError:
            n_sample_class = 32
            randInds1 = random.choices(labels_split[i], k=n_sample_class)
            randInds2 = random.choices(labels_split[i], k=n_sample_class)

            feats1 = logits[randInds1]
            feats2 = logits[randInds2]
            randInds_dif = random.choices(
                labels_split_dif[i], k=n_sample_class)

            feats_dif = logits[randInds_dif]
        # inner product: same class inner product should > dif class inner product
        inner_products = torch.sum(torch.mul(feats1, feats_dif-feats2), dim=1)
        dists = inner_products + thre
        mask = dists > 0

        loss += torch.sum(torch.mul(dists, mask.float()))

    loss /= n_sample_class*nclass
    return loss


def get_W_sparse(P, A, K):
    '''Compute probabilistic imbalance value matrix to form a weight matrix
    inputs:
    P: prediction probability matrix made by the model
    A : scipy sparse adjacency matrix A
    K : number of clusters

    output:
    flow_mat : probablistic imbalance score matrix (NumPy array)
    W : a weight matrix of the same size as A (scipy sparse matrix)
    '''
    flow_mat = np.ones([K, K])*0.5
    if isinstance(P, torch.Tensor):
        P = P.to('cpu').detach().numpy()
    A_numpy = A.toarray()
    for k in range(K-1):
        for l in range(k+1, K):
            w_kl = np.matmul(
                np.matmul(np.transpose(P[:, k]), A_numpy), P[:, l])
            w_lk = np.matmul(
                np.matmul(np.transpose(P[:, l]), A_numpy), P[:, k])
            if (w_kl + w_lk) > 0:
                flow_mat[k, l] = w_kl/(w_kl + w_lk)
                flow_mat[l, k] = w_lk/(w_kl + w_lk)
    W = sp.csr_matrix(np.matmul(np.matmul(P, flow_mat), np.transpose(P)))
    W = A.multiply(W)
    return flow_mat, W


def sparsify_A_sparse(P, A, K, sparse_ratio=0.1):
    '''Compute probabilistic imbalance value matrix to form a weight matrix
    inputs:
    prob: prediction probability matrix made by the model
    A : adjacency matrix A
    K : number of clusters
    sparse_ratio : the ratio of entries to be zeroed out

    output:
    flow_mat : probablistic imbalance score matrix (NumPy array)
    W : a weight matrix of the same size as A (or A[idx][:,idx] if idx is not None)
    '''
    flow_mat = np.ones([K, K])*0.5
    if isinstance(P, torch.Tensor):
        P = P.to('cpu').detach().numpy()
    A_numpy = A.toarray()
    A = sp.lil_matrix(A)
    for k in range(K-1):
        for l in range(k+1, K):
            w_kl = np.matmul(
                np.matmul(np.transpose(P[:, k]), A_numpy), P[:, l])
            w_lk = np.matmul(
                np.matmul(np.transpose(P[:, l]), A_numpy), P[:, k])
            if (w_kl + w_lk) > 0:
                flow_mat[k, l] = w_kl/(w_kl + w_lk)
                flow_mat[l, k] = w_lk/(w_kl + w_lk)
    W = sp.csr_matrix(np.matmul(np.matmul(P, flow_mat), np.transpose(P)))
    W = A.multiply(W)
    threshold_ind = int(W[W > 0].shape[1]*sparse_ratio)
    if threshold_ind:  # otherwise do not sparsify
        threshold_value = np.sort(
            np.array(W[W > 0]).reshape(-1))[threshold_ind]
        A[W < threshold_value] = 0
    return flow_mat, sp.csr_matrix(A)


class Prob_Imbalance_Loss(torch.nn.Module):
    r"""An implementation of the probablistic imbalance loss function.
    Args:
        F (int or NumPy array, optional): Number of pairwise imbalance socres to consider, or the meta-graph adjacency matrix.
    """

    def __init__(self, F: Optional[Union[int, np.ndarray]] = None):
        super(Prob_Imbalance_Loss, self).__init__()
        if isinstance(F, int):
            self.sel = F
        elif F is not None:
            K = F.shape[0]
            self.sel = 0
            for i in range(K-1):
                for j in range(i+1, K):
                    if (F[i, j] + F[j, i]) > 0:
                        self.sel += 1

    def forward(self, P: torch.FloatTensor, A: Union[torch.FloatTensor, torch.sparse_coo_tensor], 
    K: int, normalization: str = 'vol_sum', threshold: str = 'sort') -> torch.FloatTensor:
        """Making a forward pass of the probablistic imbalance loss function.
        Args:
            prob: (PyTorch FloatTensor) Prediction probability matrix made by the model
            A: (PyTorch FloatTensor, can be sparse) Adjacency matrix A
            K: (int) Number of clusters
            normalization: (str, optional) normalization method:
                'vol_sum': Normalized by the sum of volumes, the default choice.
                'vol_max': Normalized by the maximum of volumes.            
                'vol_min': Normalized by the minimum of volumes.   
                'plain': No normalization, just CI.   
            threshold: (str, optional) normalization method:
                'sort': Picking the top beta imbalnace values, the default choice.
                'std': Picking only the terms 3 standard deviation away from null hypothesis.             
                'naive': No thresholding, suming up all K*(K-1)/2 terms of imbalance values.  
        Returns:
            loss value, roughly in [0,1].
        """
        assert normalization in ['vol_sum', 'vol_min', 'vol_max',
                                 'plain'], 'Please input the correct normalization method name!'
        assert threshold in [
            'sort', 'std', 'naive'], 'Please input the correct threshold method name!'

        device = A.device
        # avoid zero volumn to be denominator
        epsilon = torch.FloatTensor([1e-10]).to(device)
        # first calculate the probabilitis volumns for each cluster
        vol = torch.zeros(K).to(device)
        for k in range(K):
            vol[k] = torch.sum(torch.matmul(
                A + torch.transpose(A, 0, 1), P[:, k:k+1]))
        second_max_vol = torch.topk(vol, 2).values[1] + epsilon
        result = torch.zeros(1).to(device)
        imbalance = []
        if threshold == 'std':
            imbalance_std = []
        for k in range(K-1):
            for l in range(k+1, K):
                w_kl = torch.matmul(P[:, k], torch.matmul(A, P[:, l]))
                w_lk = torch.matmul(P[:, l], torch.matmul(A, P[:, k]))
                if (w_kl-w_lk).item() != 0:
                    if threshold != 'std' or np.power((w_kl-w_lk).item(), 2)-9*(w_kl+w_lk).item() > 0:
                        if normalization == 'vol_sum':
                            curr = torch.abs(w_kl-w_lk) / \
                                (vol[k] + vol[l] + epsilon) * 2
                        elif normalization == 'vol_min':
                            curr = torch.abs(
                                w_kl-w_lk)/(w_kl + w_lk)*torch.min(vol[k], vol[l])/second_max_vol
                        elif normalization == 'vol_max':
                            curr = torch.abs(
                                w_kl-w_lk)/(torch.max(vol[k], vol[l]) + epsilon)
                        elif normalization == 'plain':
                            curr = torch.abs(w_kl-w_lk)/(w_kl + w_lk + epsilon)
                        imbalance.append(curr)
                    else:  # below-threshold values in the 'std' thresholding scheme
                        if normalization == 'vol_sum':
                            curr = torch.abs(w_kl-w_lk) / \
                                (vol[k] + vol[l] + epsilon) * 2
                        elif normalization == 'vol_min':
                            curr = torch.abs(
                                w_kl-w_lk)/(w_kl + w_lk)*torch.min(vol[k], vol[l])/second_max_vol
                        elif normalization == 'vol_max':
                            curr = torch.abs(
                                w_kl-w_lk)/(torch.max(vol[k], vol[l]) + epsilon)
                        elif normalization == 'plain':
                            curr = torch.abs(w_kl-w_lk)/(w_kl + w_lk + epsilon)
                        imbalance_std.append(curr)
        imbalance_values = [curr.item() for curr in imbalance]
        if threshold == 'sort':
            # descending order
            ind_sorted = np.argsort(-np.array(imbalance_values))
            for ind in ind_sorted[:int(self.sel)]:
                result += imbalance[ind]
            # take negation to be minimized
            return torch.ones(1, requires_grad=True).to(device) - result/self.sel
        elif len(imbalance) > 0:
            return torch.ones(1, requires_grad=True).to(device) - torch.mean(torch.FloatTensor(imbalance))
        elif threshold == 'std':  # sel is 0, then disregard thresholding
            return torch.ones(1, requires_grad=True).to(device) - torch.mean(torch.FloatTensor(imbalance_std))
        else:  # nothing has positive imbalance
            return torch.ones(1, requires_grad=True).to(device)


def get_imbalance_distribution_and_flow(labels: Union[list, np.array, torch.LongTensor],
                                        num_clusters: int,
                                        A: Union[torch.FloatTensor, torch.sparse_coo_tensor],
                                        F: Union[int, np.ndarray],
                                        normalizations: list = ['vol_sum', 'plain'],
                                        thresholds: list = ['sort', 'std', 'naive']) -> Tuple[list, np.array, np.ndarray]:
    r"""Computes imbalance values, distribution of labels, 
    and predicted meta-graph flow matrix.

    Args:
        labels: (list, np.array, or torch.LongTensor) Predicted labels.
        num_clusters: (int) Number of clusters.
        A: (torch.FloatTensor or torch.sparse_coo_tensor) Adjacency matrix.
        F: (int or  np.ndarray) Number of selections in "sort" flavor or 
            the meta-graph adjacency matrix.
        normalizations: (list) Normalization methods to consider, 
            default is ['vol_sum','plain'].
        thresholds: (list) Thresholding methods to consider, 
            default is ['sort', 'std', 'naive'].

    :rtype: 
        imbalance_list: (list) List of imbalance values from different loss functions.
        labels_distribution: (np.array) Array of distribution of labels.
        flow_mat: (np.ndarray) Predicted meta-graph flow matrix.
        pairwise_imbalance_array: (np.array) Array of pairwise imbalance values from different objective functions,
            with shape (number of normalizations, num_clusters*(num_clusters)/2).
    """
    P = torch.zeros(labels.shape[0], num_clusters).to(A.device)
    loss = Prob_Imbalance_Loss(F)
    for k in range(num_clusters):
        P[labels == k, k] = 1
    labels_distribution = np.array(P.sum(0).to('cpu').numpy(), dtype=int)
    imbalance_list = []
    for threshold in thresholds:
        for normalization in normalizations:
            imbalance_list.append(
                1-loss(P, A, num_clusters, normalization, threshold).item())
    flow_mat = np.ones([num_clusters, num_clusters])*0.5
    for k in range(num_clusters-1):
        for l in range(k+1, num_clusters):
            w_kl = torch.matmul(P[:, k], torch.matmul(A, P[:, l])).item()
            w_lk = torch.matmul(P[:, l], torch.matmul(A, P[:, k])).item()
            if (w_kl + w_lk) > 0:
                flow_mat[k, l] = w_kl/(w_kl + w_lk)
                flow_mat[l, k] = w_lk/(w_kl + w_lk)
    pairwise_imbalance_array = get_pairwise_imbalance(labels=labels,
                                                    num_clusters=num_clusters,
                                                    A=A,
                                                    normalizations=normalizations)
    return imbalance_list, labels_distribution, flow_mat, pairwise_imbalance_array


default_compare_names_all = ['InfoMap',
                             'Bi_sym',
                             'DD_sym',
                             'DISG_LR',
                             'Herm',
                             'Herm_rw',
                             'DIGRAC']
default_metric_names = []
for threshold in ['sort', 'std', 'naive']:
    for normalization in ['vol\_sum', 'vol\_min', 'vol\_max', 'plain']:
        loss_choice = '$\mathcal{O}_\\text{'+normalization + '}^\\text{' + threshold + '}$'
        default_metric_names.append(loss_choice)
default_metric_names.extend(['size ratio','size std'])

def print_performance_mean_std(dataset:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               metric_names:list=default_metric_names, print_latex:bool=True, smaller_better:bool=False, print_std:bool=False):
    r"""Prints performance table (and possibly with latex) with mean and standard deviations.
        The best two performing methods are highlighted in \red and \blue respectively.

    Args:
        dataset: (string) Name of the data set considered.
        results: (np.array) Results with shape (num_trials, num_methods, num_metrics).
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (the last two are deemed better with smaller values, 
            while the rest should aim for higher values).
        print_latex: (bool, optional) Whether to print latex table also. Default True.
        smaller_better: (bool, optional) Whether for all metrics the smaller the better.
        print_std: (bool, optinoal) Whether to print standard deviations or just mean. Default False.
    """
    t = Texttable(max_width=120)
    t.set_deco(Texttable.HEADER)
    final_res_show = np.chararray(
        [len(metric_names)+1, len(compare_names_all)+1], itemsize=50)
    final_res_show[0, 0] = dataset+'Metric/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = metric_names
    std = np.chararray(
        [len(metric_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(results.std(0),2))
    results_mean = np.transpose(np.round(results.mean(0),2))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.2f}'.format(results_mean[i, j])
            std[i, j] = '{:.2f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(metric_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    else:
        plus_minus = np.chararray(
            [len(metric_names)-2, len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:-2, 1:] = final_res_show[1:-2, 1:] + plus_minus + std[:-2]
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i, metric in enumerate(metric_names):
            if smaller_better or (metric in ['size ratio', 'size std']): # the smaller, the better
                best_values = np.sort(results_mean[i])[:2]
            else: # the larger, the better.
                best_values = -np.sort(-results_mean[i])[:2]
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=dataset +
                                    " performance.", label="table:"+dataset) + "\n")

def print_performance_mean_std_analysis(dataset:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               metric_names:list=default_metric_names, print_latex:bool=True, print_std:bool=True):
    r"""Prints performance table (and possibly with latex) with mean and standard deviations.
        The best two performing methods are highlighted in \red and \blue respectively.
    Args:
        dataset: (string) Name of the data set considered.
        results: (np.array) Results with shape (num_trials, num_methods, num_metrics).
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (deemed better with larger values).
        print_latex: (bool, optional) Whether to print latex table also. Default True.
        print_std: (bool, optinoal) Whether to print standard deviations or just mean. Default False.
    """
    t = Texttable(max_width=120)
    t.set_deco(Texttable.HEADER)
    final_res_show = np.chararray(
        [len(metric_names)+1, len(compare_names_all)+1], itemsize=100)
    final_res_show[0, 0] = dataset+'Metric/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = metric_names
    std = np.chararray(
        [len(metric_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(np.nanstd(results,0),2))
    results_mean = np.transpose(np.round(np.nanmean(results,0),2))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.2f}'.format(results_mean[i, j])
            std[i, j] = '{:.2f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(metric_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            if metric_names[i] not in ['size ratio', 'size std']:
                best_values = -np.sort(-results_mean[i])[:2] # the bigger, the better
            else:
                best_values = np.sort(results_mean[i])[:2] # the smaller, the better
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=dataset +
                                    " performance.", label="table:"+dataset) + "\n")

def print_overall_performance_mean_std(title:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               dataset_names:list=['animal'], print_latex:bool=True, print_std:bool=True):
    r"""Prints performance table (and possibly with latex) with mean and standard deviations.
        The best two performing methods are highlighted in \red and \blue respectively.
    Args:
        dataset: (string) Name of the data set considered.
        results: (np.array) Results with shape (num_trials, num_methods, num_metrics).
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (deemed better with larger values).
        print_latex: (bool, optional) Whether to print latex table also. Default True.
        print_std: (bool, optinoal) Whether to print standard deviations or just mean. Default False.
    """
    t = Texttable(max_width=120)
    t.set_deco(Texttable.HEADER)
    final_res_show = np.chararray(
        [len(dataset_names)+1, len(compare_names_all)+1], itemsize=100)
    final_res_show[0, 0] = title+'Data/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = dataset_names
    std = np.chararray(
        [len(dataset_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(np.nanstd(results,0),2))
    results_mean = np.transpose(np.round(np.nanmean(results,0),2))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.2f}'.format(results_mean[i, j])
            std[i, j] = '{:.2f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(dataset_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            if title[:4] != 'size':
                best_values = -np.sort(-results_mean[i])[:2] # the bigger, the better
            else:
                best_values = np.sort(results_mean[i])[:2] # the smaller, the better
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=title +
                                    " performance.", label="table:"+title) + "\n")


def print_performance_by_trials(dataset, results, compare_names_all=default_compare_names_all,
                                metric_names=default_metric_names, print_latex=True):
    r"""Prints performance table (and possibly with latex) in details along with mean and standard deviations.
        The best three performing methods are also printed respectively.

    Args:
        dataset: (string) Name of the data set considered.
        results: (np.array) Results with shape (num_trials, num_methods, num_metrics).
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (the last two are deemed better with smaller values, 
            while the rest should aim for higher values).
        print_latex: (bool, optional) Whether to print latex table also. Default True.
    """
    num_trials = results.shape[0]
    final_res_show = np.chararray(
        [num_trials+6, len(compare_names_all)+1], itemsize=50)
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:num_trials+1, 0] = np.arange(num_trials)
    final_res_show[-5, 0] = "Best"
    final_res_show[-4, 0] = "Second best"
    final_res_show[-3, 0] = "Third best"
    final_res_show[-2, 0] = "Mean"
    final_res_show[-1, 0] = 'Std'
    assert results.shape[-1] == len(
        metric_names), 'Evaluation metric dimensions do not match!'
    for i in range(len(metric_names)):
        t = Texttable(max_width=120)
        final_res_show[0, 0] = dataset+metric_names[i] + "\Method"
        final_res_show[1:num_trials+1, 1:] = results[:, :, i]
        final_res_show[-2, 1:] = results[:, :, i].mean(0)
        final_res_show[-1, 1:] = results[:, :, i].std(0)
        if i < len(metric_names)-2:
            final_res_show[-5:-2, 1:] = np.argsort(-results[:, :, i], axis=0)[:3]
        else:
            final_res_show[-5:-2, 1:] = np.argsort(results[:, :, i], axis=0)[:3]
        t.add_rows(final_res_show)
        print(t.draw())
        if print_latex:
            print(latextable.draw_latex(t, caption=dataset+" detailed performance.",
                                        label="table:"+dataset+'_'+metric_names[i]+"_detailed") + "\n")


def label_size_ratio(labels_distributions, return_std=False, return_num_clusters=False):
    num_trials, num_methods, _ = labels_distributions.shape
    size_ratio = np.zeros([num_trials, num_methods])
    if return_num_clusters:
        num_clusters_pred = np.zeros([num_trials, num_methods])
        num_nodes_covered_in_recorded_clusters = np.zeros([num_trials, num_methods])
    if return_std:
        size_std = np.zeros([num_trials, num_methods])
    for i in range(num_trials):
        for j in range(num_methods):
            data = labels_distributions[i, j]
            data = data[data.nonzero()]
            size_ratio[i, j] = data.max()/data.min()
            if return_std:
                size_std[i, j] = np.nanstd(data)
            if return_num_clusters:
                num_clusters_pred[i, j] = len(data)
                num_nodes_covered_in_recorded_clusters[i, j] = np.sum(data)
    if return_std and not return_num_clusters:
        return size_ratio, size_std
    elif not return_std and return_num_clusters:
        return size_ratio, num_clusters_pred, num_nodes_covered_in_recorded_clusters
    elif return_std and return_num_clusters:
        return size_ratio, size_std, num_clusters_pred, num_nodes_covered_in_recorded_clusters
    else:
        return size_ratio

def plot_migration(tag, num_clusters, loc, method_name, save=False, plot_boundary=True):
    N = num_clusters # Number of labels

    # setup the plot
    fig, ax = plt.subplots(1,1, figsize=(14,6))
    # define the data
    x = loc[:, 0]
    y = loc[:, 1]

    if plot_boundary:
        boundary = np.loadtxt('../dataset/data/tmp/migration/USA_boundaries.txt')
        boundary = boundary[:,1:3]
        boundary = boundary[boundary[:,0]>=min(x)]
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    if num_clusters != 3:
        cmaplist = [cmap(i) for i in range(cmap.N)]
    else:
        cmaplist = ['yellow', 'red', 'blue']
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    if N <= 10:
        bounds = np.linspace(0,N,N+1)
    else:
        bounds = np.linspace(0,N,11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # make the scatter
    scat = ax.scatter(x,y,c=tag,cmap=cmap,     norm=norm, s=10)
    if plot_boundary:
        _ = ax.scatter(boundary[:,0], boundary[:,1],c='black',s=1)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cb.set_label('Label')
    if save:
        plt.savefig('../comparison_plots/real_plots/Migration_'+method_name+'.pdf')
    ax.set_title(method_name)
    plt.show()


def real_data_analysis(dataset, result_folder_name, name_base,
                    extra_result_folder_name=None, extra_name_base=None,
                    dir_name='../logs/migration_ratio/1000_0_80_10_1000/09-16-04:35:23/',
                    extra_dir_name=None,
                    compare_names_all=default_compare_names_all, metric_names=default_metric_names,
                    mean_std=True):
    r"""Conducts real data analysis.

    Args:
        dataset: (string) Name of the data set considered, e.g. 'migration'.
        (extra_)result_folder_name: (str) Directory to store ARI (wrt to labels), flow matrices, imbalance values, 
            e.g. '../result_arrays/migration/'. "extra" means for non-GNN and non-spectral methods.
        (extra_)name_base: (str) The invariant component in result array file names, 
            e.g. '0_200_50_0_0_1_3200_10_5000_sort_andvol_sum_SpectralDIGRAC.npy'.
        (extra_)dir_name: (str, optional) Directory to store predicted cluster assignments.
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (the last two are deemed better with smaller values, 
            while the rest should aim for higher values).
        mean_std: (bool, optinoal) Whether to print mean-std table instead of detailed tables. Default True.
    """

    if 'InfoMap' in compare_names_all:
        flow = np.concatenate((np.load(extra_result_folder_name+'flow'+extra_name_base)[:,:1], \
            np.load(result_folder_name+'flow'+name_base)),axis=1)
        dist = np.concatenate((np.load(extra_result_folder_name+'labels_distribution'+extra_name_base)[:,:1], \
            np.load(result_folder_name+'labels_distribution'+name_base)),axis=1)
        CI = np.concatenate((np.load(extra_result_folder_name+'CI'+extra_name_base)[:,:1], \
            np.load(result_folder_name+'CI'+name_base)),axis=1)
    else:
        flow = np.load(result_folder_name+'flow'+name_base)
        dist = np.load(result_folder_name+'labels_distribution'+name_base)
        CI = np.load(result_folder_name+'CI'+name_base)
    size_ratio, size_std = label_size_ratio(dist, True)
    num_clusters = dist.shape[-1]
    num_trials = CI.shape[0]


    for i in range(num_trials):
        for j in range(len(compare_names_all)):
            flow_mat = flow[i, j]
            plt.figure(figsize=[8,6])
            plt.rcParams.update({'font.size': 23.5})
            plt.matshow(flow_mat)
            plt.colorbar()
            save_name = '../comparison_plots/real_plots/flow_'+compare_names_all[j]+'_'+dataset+str(i)+'.pdf'
            plt.savefig(save_name)
            plt.title(
                'Meta graph adjacency matrix by '+compare_names_all[j]+' predicted labels, trial {}'.format(i))
            plt.show()

    if dataset != 'blog':
        if 'InfoMap' not in compare_names_all:
            res = np.load(result_folder_name+'pairwise'+name_base)
        else:
            res = np.concatenate((np.load(extra_result_folder_name+'pairwise'+extra_name_base)[:,:1], \
                np.load(result_folder_name+'pairwise'+name_base)),axis=1)
        for trial in range(num_trials):
            pairwise_imbalance_array_full = res[trial]
            figure_markers = ['*','P','<','s','8','+','H','|','D','>','v','^','d']
            fig, axs = plt.subplots(2, 2, figsize=(18,15), sharex=True)
            x_val = np.arange(int(num_clusters*(num_clusters-1)/2))+1
            for i in range(len(compare_names_all)):
                plt.rcParams.update({'font.size': 23.5})
                mpl.rc('xtick', labelsize=20) 
                mpl.rc('ytick', labelsize=20) 
                if i != len(compare_names_all)-1:
                    axs[0,0].scatter(x_val,pairwise_imbalance_array_full[i,0],marker=figure_markers[i], ls='None',label=compare_names_all[i])
                    axs[0,1].scatter(x_val,pairwise_imbalance_array_full[i,1],marker=figure_markers[i], ls='None', label=compare_names_all[i])
                    axs[1,0].scatter(x_val,pairwise_imbalance_array_full[i,2],marker=figure_markers[i], ls='None', label=compare_names_all[i])
                    axs[1,1].scatter(x_val,pairwise_imbalance_array_full[i,3],marker=figure_markers[i], ls='None', label=compare_names_all[i])
                else:
                    axs[0,0].plot(x_val,pairwise_imbalance_array_full[i,0], 'm', label=compare_names_all[i])
                    axs[0,0].legend(loc='best', fontsize=20)
                    axs[0,0].set_ylabel(r'$CI^{vol\_sum}$ ranked pairs', fontsize=20)
                    axs[0,1].plot(x_val,pairwise_imbalance_array_full[i,1], 'm', label=compare_names_all[i])
                    axs[0,1].set_ylabel(r'$CI^{vol\_min}$ ranked pairs', fontsize=20)
                    axs[1,0].plot(x_val,pairwise_imbalance_array_full[i,2], 'm', label=compare_names_all[i])
                    axs[1,0].set_ylabel(r'$CI^{vol\_max}$ ranked pairs', fontsize=20)
                    axs[1,0].set_xlabel('rank', fontsize=20)
                    axs[1,1].plot(x_val,pairwise_imbalance_array_full[i,3], 'm', label=compare_names_all[i])
                    axs[1,1].set_ylabel(r'$CI^{plain}$ ranked pairs')
                    axs[1,1].set_xlabel('rank', fontsize=20)
            plt.savefig('../comparison_plots/ranked_pairs/Top_pairs_'+dataset+str(trial)+'.pdf')
            fig.suptitle('Ranked pairs for '+dataset+' trial '+str(trial))
            plt.show()


    if mean_std:
        print_performance_mean_std(dataset, np.concatenate(
            (CI, size_ratio[:, :, None], size_std[:, :, None]), 2), compare_names_all, metric_names)
    else:
        print_performance_by_trials(dataset, np.concatenate(
            (CI, size_ratio[:, :, None], size_std[:, :, None]), 2), compare_names_all, metric_names, False)


    if dataset in ['migration', 'migration_ratio']:
        spectral_names = ['Bi_sym',
                        'DD_sym',
                        'DISG_LR',
                        'Herm',
                        'Herm_rw']
        coord = pd.read_csv('../dataset/data/tmp/migration/Coord.csv', header=None)
        loc = coord.values
        for trial in range(num_trials):
            title = 'spectral_pred' + str(trial)
            spectral_pred = np.load(dir_name+title+'.npy')
            for i in range(len(spectral_names)):
                pred = spectral_pred[i]
                plot_migration(pred, num_clusters, loc, dataset+spectral_names[i]+str(trial), True)
            DIGRAC_pred = np.load(dir_name+'DIGRAC_vol_sum_sort_pred' + str(trial)+'.npy')
            plot_migration(DIGRAC_pred, num_clusters, loc, dataset+'DIGRAC'+str(trial), True)
            if 'InfoMap' in compare_names_all:
                pred = np.load(extra_dir_name+'extra_pred' + str(trial)+'.npy')[0]
                pred = pred - pred.min()
                plot_migration(pred, int(pred.max()+1), loc, dataset+'InfoMap'+str(trial), True)





def pairwise_imbalance(P: torch.FloatTensor, A: Union[torch.FloatTensor, torch.sparse_coo_tensor], 
K: int, normalization: str = 'vol_sum') -> list:
    """Probablistic pairwise imbalance score calculation function.
    Args:
        prob: (PyTorch FloatTensor) Prediction probability matrix made by the model
        A: (PyTorch FloatTensor, can be sparse) Adjacency matrix A
        K: (int) Number of clusters
        normalization: (str, optional) normalization method:
            'vol_sum': Normalized by the sum of volumes, the default choice.
            'vol_max': Normalized by the maximum of volumes.            
            'vol_min': Normalized by the minimum of volumes.   
            'plain': No normalization, just CI.

    Returns:
        list of sorted pairwise imbalance scores, each is roughly in [0,1].
    """
    assert normalization in ['vol_sum', 'vol_min', 'vol_max',
                                'plain'], 'Please input the correct normalization method name!'

    device = A.device
    # avoid zero volumn to be denominator
    epsilon = torch.FloatTensor([1e-8]).to(device)
    # first calculate the probabilitis volumns for each cluster
    vol = torch.zeros(K).to(device)
    total_vol = torch.zeros(1).to(device)
    for k in range(K):
        vol[k] = torch.sum(torch.matmul(
            A + torch.transpose(A, 0, 1), P[:, k:k+1]))
        total_vol += vol[k]
    avg_vol = total_vol/K
    imbalance = []
    for k in range(K-1):
        for l in range(k+1, K):
            w_kl = torch.matmul(P[:, k], torch.matmul(A, P[:, l]))
            w_lk = torch.matmul(P[:, l], torch.matmul(A, P[:, k]))
            if (w_kl-w_lk).item():
                if normalization == 'vol_sum':
                    curr = torch.abs(w_kl-w_lk) / \
                        (vol[k] + vol[l] + epsilon) * 2 * K
                elif normalization == 'vol_min':
                    curr = torch.abs(
                        w_kl-w_lk)/(w_kl + w_lk)*torch.min(vol[k], vol[l])/avg_vol
                elif normalization == 'vol_max':
                    curr = torch.abs(
                        w_kl-w_lk)/(torch.max(vol[k], vol[l]) + epsilon) * K
                elif normalization == 'plain':
                    curr = torch.abs(w_kl-w_lk)/(w_kl + w_lk)
            else:
                curr = torch.zeros(1).to(device)
            imbalance.append(curr)
    imbalance_values = [curr.item() for curr in imbalance]
    imbalance_values.sort(reverse=True)
    return imbalance_values

def get_pairwise_imbalance(labels: Union[list, np.array, torch.LongTensor],
                        num_clusters: int,
                        A: Union[torch.FloatTensor, torch.sparse_coo_tensor],
                        normalizations: list = ['vol_sum', 'vol_min', 'vol_max','plain']) -> np.array:
    r"""Computes pairwise imbalance scores for different normalizations.

    Args:
        labels: (list, np.array, or torch.LongTensor) Predicted labels.
        num_clusters: (int) Number of clusters.
        A: (torch.FloatTensor or torch.sparse_coo_tensor) Adjacency matrix.
        normalizations: (list) Normalization methods to consider, 
            default is ['vol_sum','plain'].

    :rtype: 
        pairwise_imbalance_array: (np.array) Array of pairwise imbalance values from different objective functions,
            with shape (number of normalizations, num_clusters*(num_clusters)/2).
    """
    P = torch.zeros(labels.shape[0], num_clusters).to(A.device)
    for k in range(num_clusters):
        P[labels == k, k] = 1
    pairwise_imbalance_array = np.zeros((len(normalizations), int(num_clusters*(num_clusters-1)/2)))
    for i, normalization in enumerate(normalizations):
        pairwise_imbalance_array[i] = pairwise_imbalance(P, A, num_clusters, normalization)
    return pairwise_imbalance_array

def real_data_select_trial(dataset, result_folder_name, name_base,
                    compare_names_all=default_compare_names_all, metric_names=default_metric_names,
                    mean_std=True):
    r"""Conducts real data analysis by trials to help select the best performing trials.

    Args:
        dataset: (string) Name of the data set considered, e.g. 'migration'.
        result_folder_name: (str) Directory to store ARI (wrt to labels), flow matrices, imbalance values, 
            e.g. '../result_arrays/migration/'.
        name_base: (str) The invariant component in result array file names, 
            e.g. '0_200_50_0_0_1_3200_10_5000_sort_andvol_sum_SpectralDIGRAC.npy'.
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (the last two are deemed better with smaller values, 
            while the rest should aim for higher values).
    """
    dist = np.load(result_folder_name+'labels_distribution'+name_base)
    CI = np.load(result_folder_name+'CI'+name_base)
    size_ratio, size_std = label_size_ratio(dist, True)
    print_performance_by_trials(dataset, np.concatenate(
        (CI, size_ratio[:, :, None], size_std[:, :, None]), 2), compare_names_all, metric_names, False)
    print_performance_mean_std(dataset, np.concatenate(
            (CI, size_ratio[:, :, None], size_std[:, :, None]), 2), compare_names_all, metric_names)

def link_prediction_evaluation(out_val, out_test, y_val, y_test):
    r"""Evaluates link prediction results.

    Args:
        out_val: (torch.FloatTensor) Log probabilities of validation edge output, with 2 or 3 columns.
        out_test: (torch.FloatTensor) Log probabilities of test edge output, with 2 or 3 columns.
        y_val: (torch.LongTensor) Validation edge labels (with 2 or 3 possible values).
        y_test: (torch.LongTensor) Test edge labels (with 2 or 3 possible values).

    :rtype: 
        result_array: (np.array) Array of evaluation results, with shape (2, 5).
    """
    out = torch.exp(out_val).detach().to('cpu').numpy()
    y_val = y_val.detach().to('cpu').numpy()
    # possibly three-class evaluation
    pred_label = np.argmax(out, axis = 1)
    val_acc_full = accuracy_score(pred_label, y_val)
    # two-class evaluation
    out = out[y_val < 2, :2]
    y_val = y_val[y_val < 2]


    prob = out[:,0]/(out[:,0]+out[:,1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    val_auc = roc_auc_score(y_val, prob)
    pred_label = np.argmax(out, axis = 1)
    val_acc = accuracy_score(pred_label, y_val)
    val_f1_macro = f1_score(pred_label, y_val, average='macro')
    val_f1_micro = f1_score(pred_label, y_val, average='micro')

    out = torch.exp(out_test).detach().to('cpu').numpy()
    y_test = y_test.detach().to('cpu').numpy()
    # possibly three-class evaluation
    pred_label = np.argmax(out, axis = 1)
    test_acc_full = accuracy_score(pred_label, y_test)
    # two-class evaluation
    out = out[y_test < 2, :2]
    y_test = y_test[y_test < 2]
    

    prob = out[:,0]/(out[:,0]+out[:,1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    test_auc = roc_auc_score(y_test, prob)
    pred_label = np.argmax(out, axis = 1)
    test_acc = accuracy_score(pred_label, y_test)
    test_f1_macro = f1_score(pred_label, y_test, average='macro')
    test_f1_micro = f1_score(pred_label, y_test, average='micro')
    return [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
            [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro]]

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

def clustering_evaluation(A, pred, labels):
    dir_mod = directed_modularity(A, pred)
    if labels[0] != -1:      
        ARI = adjusted_rand_score(pred, labels)
        nmi = normalized_mutual_info_score(pred, labels)
        acc = accuracy_score(pred, labels)
    else:
        ARI = np.nan
        nmi = np.nan
        acc = np.nan
    return dir_mod, ARI, nmi, acc