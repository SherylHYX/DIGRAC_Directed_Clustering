# external files
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import csv
import math
import random
import pickle as pk
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from texttable import Texttable

# internal files
from utils import write_log, scipy_sparse_to_torch_sparse
from metrics import get_imbalance_distribution_and_flow
from metrics import Prob_Imbalance_Loss, print_performance_mean_std, label_size_ratio
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from param_parser import parameter_parser
from extra_comparison import InfoMap, leidenalg_modularity, louvain_modularity, directed_modularity, OSLOM
from preprocess import load_data

args = parameter_parser()
args.all_methods = ['extra']
torch.manual_seed(args.seed)
device = args.device
if args.cuda:
    print("Using cuda")
    torch.cuda.manual_seed(args.seed)
no_magnet = True
if 'extra' in args.all_methods:
    compare_names = ['infomap', 'louvain_modularity', 'leidenalg_modularity','GT'] # also with ground_truth labels
compare_names_all = compare_names


class Trainer(object):
    """
    Object to train and score different models.
    """

    def __init__(self, args):
        """
        Constructing the trainer instance.
        :param args: Arguments object.
        """
        self.args = args
        self.device = args.device

        label, self.train_mask, self.val_mask, self.test_mask, self.seed_mask, comb = load_data(
            args, args.q, args.K_m, True, args.load_only, True)

        # normalize label, the minimum should be 0 as class index
        _label_ = label - np.amin(label)
        self.label = torch.from_numpy(_label_[np.newaxis]).to(device)
        self.cluster_dim = np.amax(_label_)+1
        self.num_clusters = self.cluster_dim

        self.features, self.A = comb
        self.nfeat = self.features.shape[1]


        date_time = datetime.now().strftime('%m-%d-%H:%M:%S')

        if args.dataset == 'DSBM/':
            param_values = [args.p, args.K, args.N, args.hop, args.tau, args.size_ratio,
                            args.seed_ratio, args.eta, args.F_style, args.sp_style, args.loss_choice]
            save_name = '_'.join([str(int(100*value)) for value in param_values[:-3]]) + \
                '_' + args.F_style + '_' + args.sp_style + '_' + args.loss_choice
        else:
            param_values = [self.num_clusters, args.seed_ratio,
                            args.train_ratio, args.test_ratio, args.num_trials]
            save_name = '_'.join([str(int(100*value))
                                  for value in param_values])
        if args.seed != 31:
            save_name = 'Seed' + str(args.seed) + '_' + save_name
        self.log_path = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), args.log_root, args.dataset[:-1], save_name, date_time)

        if os.path.isdir(self.log_path) == False:
            try:
                os.makedirs(self.log_path)
            except FileExistsError:
                print('Folder exists!')

        self.splits = self.train_mask.shape[1]
        if len(self.test_mask.shape) == 1:
            #data.test_mask = test_mask.unsqueeze(1).repeat(1, splits)
            self.test_mask = np.repeat(
                self.test_mask[:, np.newaxis], self.splits, 1)
        write_log(vars(args), self.log_path)  # write the setting

    def gen_results_extra(self):
        # torch.FloatTensor(np.array(self.A.todense())).to(self.args.device)
        A = scipy_sparse_to_torch_sparse(self.A).to(self.args.device)
        num_trials = self.splits
        res_full = np.zeros([num_trials, len(compare_names)])
        res_all_full = np.zeros([num_trials, len(compare_names)])
        acc_full = np.zeros([num_trials, len(compare_names)])
        acc_all_full = np.zeros([num_trials, len(compare_names)])
        nmi_full = np.zeros([num_trials, len(compare_names)]) # nomalized mutual information
        nmi_all_full = np.zeros([num_trials, len(compare_names)])
        dir_mod_full = np.zeros([num_trials, len(compare_names)]) # directed modularity
        dir_mod_all_full = np.zeros([num_trials, len(compare_names)])
        CI_full = np.zeros([self.splits, len(compare_names), len(
            self.args.report_normalizations)*len(self.args.report_thresholds)])
        labels_distribution_full = np.zeros(
            [self.splits, len(compare_names), self.num_clusters])
        flow_mat_full = np.zeros(
            [self.splits, len(compare_names), self.num_clusters, self.num_clusters])
        pairwise_imbalance_array_full = np.zeros(
            [self.splits, len(compare_names), len(self.args.report_normalizations), int(self.num_clusters*(self.num_clusters-1)/2)])
        for split in range(self.splits):
            if self.args.SavePred:
                pred_all = np.zeros(
                    [len(compare_names), len(self.label.view(-1).to('cpu'))])
            test_index = self.test_mask[:, split]
            test_A = self.A[test_index][:,test_index]
            res = []
            res_all = []
            acc = []
            acc_all = []
            nmi = []
            nmi_all = []
            dir_mod = []
            dir_mod_all = []
            labels_test_cpu = (self.label.view(-1)[test_index]).to('cpu')
            labels_cpu = self.label.view(-1).to('cpu')
            idx_test_cpu = test_index
            # now append results for comparison methods
            for i, pred in enumerate([InfoMap(self.A), louvain_modularity(self.A), 
                                        leidenalg_modularity(self.A), labels_cpu]):
                pred = np.array(pred - pred.min(), dtype=int)
                num_clusters = pred.max() + 1
                if num_clusters > 1:
                    try:
                        imbalance_list, labels_distribution, flow_mat, pairwise_imbalance_array = get_imbalance_distribution_and_flow(pred,
                                                                                                            num_clusters, A, self.args.F, self.args.report_normalizations, self.args.report_thresholds)
                    except RuntimeError:
                        imbalance_list = np.nan
                        labels_distribution = np.nan
                        flow_mat = np.nan
                        pairwise_imbalance_array = np.nan
                    CI_full[split, i] = imbalance_list
                    min_num_clusters = int(min(num_clusters, self.num_clusters))
                    min_pairwise_num = int(min_num_clusters * (min_num_clusters - 1)/2)
                    labels_distribution_full[split, i, :min_num_clusters] = labels_distribution[:min_num_clusters]
                    flow_mat_full[split, i, :min_num_clusters, :min_num_clusters] = flow_mat[:min_num_clusters, :min_num_clusters]
                    pairwise_imbalance_array_full[split, i, :, :min_pairwise_num] = pairwise_imbalance_array[:, :min_pairwise_num]
                else:
                    CI_full[split, i] = np.nan
                    labels_distribution_full[split, i, 0] = A.shape[0]
                    flow_mat_full[split, i] = np.nan
                    pairwise_imbalance_array_full[split, i] = np.nan
                res.append(adjusted_rand_score(
                    pred[idx_test_cpu], labels_test_cpu))
                res_all.append(adjusted_rand_score(pred, labels_cpu))
                acc.append(accuracy_score(labels_test_cpu,
                    pred[idx_test_cpu]))
                acc_all.append(accuracy_score(labels_cpu, pred))
                nmi.append(normalized_mutual_info_score(labels_test_cpu,
                    pred[idx_test_cpu]))
                nmi_all.append(normalized_mutual_info_score(labels_cpu, pred))
                dir_mod.append(directed_modularity(test_A,
                    pred[idx_test_cpu]))
                dir_mod_all.append(directed_modularity(self.A, pred))
                if self.args.SavePred:
                    pred_all[i] = pred
            if self.args.SavePred:
                np.save(self.log_path + '/extra_pred'+str(split), pred_all)
            res_full[split] = res
            res_all_full[split] = res_all
            acc_full[split] = acc
            acc_all_full[split] = acc_all
            nmi_full[split] = nmi
            nmi_all_full[split] = nmi_all
            dir_mod_full[split] = dir_mod
            dir_mod_all_full[split] = dir_mod_all
        print('Test ARI for methods to compare:{}'.format(res_full))
        print('All data ARI for methods to compare:{}'.format(res_all_full))
        return res_full, res_all_full, acc_full, acc_all_full, \
            nmi_full, nmi_all_full, dir_mod_full, dir_mod_all_full, \
            CI_full, labels_distribution_full, flow_mat_full, pairwise_imbalance_array_full


# train and grap results
if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/debug/'+args.dataset[:-1]+'/')
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../extra_results/'+args.dataset[:-1]+'/')
if os.path.isdir(dir_name) == False:
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        print('Folder exists!')

args.loss_choice = ''
for threshold in args.thresholds:
    args.loss_choice += threshold + '_'
args.loss_choice += 'and'
for normalization in args.normalizations:
    args.loss_choice += normalization + '_'

trainer = Trainer(args)
final_res = np.zeros([args.num_trials, len(compare_names_all)])
final_res_all = np.zeros([args.num_trials, len(compare_names_all)])
final_acc = np.zeros([args.num_trials, len(compare_names_all)])
final_acc_all = np.zeros([args.num_trials, len(compare_names_all)])
final_nmi = np.zeros([args.num_trials, len(compare_names_all)])
final_nmi_all = np.zeros([args.num_trials, len(compare_names_all)])
final_dir_mod = np.zeros([args.num_trials, len(compare_names_all)])
final_dir_mod_all = np.zeros([args.num_trials, len(compare_names_all)])
CI_full = np.zeros([args.num_trials, len(compare_names_all), len(
    args.report_normalizations)*len(args.report_thresholds)])
labels_distribution_full = np.zeros(
    [args.num_trials, len(compare_names_all), trainer.num_clusters])
flow_mat_full = np.zeros([args.num_trials, len(
    compare_names_all), trainer.num_clusters, trainer.num_clusters])
pairwise_imbalance_array_full = np.zeros(
            [args.num_trials, len(compare_names_all), len(args.report_normalizations), int(trainer.num_clusters*(trainer.num_clusters-1)/2)])
        
method_str = ''
if 'extra' in args.all_methods:
    method_str += 'extra'
    # extra methods
    final_res[:, :len(compare_names)], \
    final_res_all[:, :len(compare_names)], \
    final_acc[:, :len(compare_names)], \
    final_acc_all[:, :len(compare_names)], \
    final_nmi[:, :len(compare_names)], \
    final_nmi_all[:, :len(compare_names)], \
    final_dir_mod[:, :len(compare_names)], \
    final_dir_mod_all[:, :len(compare_names)], \
    CI_full[:, :len(compare_names)], \
    labels_distribution_full[:, :len(compare_names)], \
    flow_mat_full[:, :len(compare_names)], \
    pairwise_imbalance_array_full[:, :len(compare_names)] = trainer.gen_results_extra()

# print results and save results to arrays
t = Texttable(max_width=120)
if args.dataset == 'DSBM/':
    param_values = [args.fill_val, args.ambient, args.initial_loss_epochs, args.p,args.eta,args.K,args.N,args.hop,args.tau, args.size_ratio, 
    args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, args.supervised_loss_ratio]
    t.add_rows([["Parameter","filled value","ambient","initial loss epochs","p","eta","K","N","hop","tau","size ratio",
    "seed ratio", "alpha", "lr", "hidden","triplet loss ratio","supervised loss ratio","F style"],
    ["Values",args.fill_val, args.ambient, args.initial_loss_epochs, args.p,args.eta,args.K,args.N,args.hop,args.tau, args.size_ratio, 
    args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, args.supervised_loss_ratio, args.F_style]])
    save_name = '_'.join([str(int(100*value)) for value in param_values]) + \
        '_' + args.F_style + '_' + args.sp_style + '_' + args.loss_choice + method_str
else:
    param_values = [args.initial_loss_epochs, args.hop,args.tau, 
    args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, args.supervised_loss_ratio]
    t.add_rows([["Parameter","initial loss epochs","hop","tau",
    "seed ratio", "alpha", "lr", "hidden","triplet loss ratio","supervised loss ratio"],
    ["Values",args.initial_loss_epochs, args.hop,args.tau, 
    args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, args.supervised_loss_ratio]])
    save_name = '_'.join([str(int(100*value)) for value in param_values]) + '_' + args.loss_choice + method_str
if args.seed != 31:
    save_name = 'Seed' + str(args.seed) + '_' + save_name
print(t.draw())

metric_names = ['test ARI', 'all ARI', 'test_nmi', 'all_nmi', 'test_dir_mod', 'all_dir_mod']
for threshold in args.report_thresholds:
    for normalization in args.report_normalizations:
        metric_names.append(normalization + '_' + threshold)
metric_names.append('size_ratio')

size_ratio = label_size_ratio(labels_distribution_full)
print_performance_mean_std(args.dataset[:-1]+'_best', np.concatenate((final_res[:, :, None], final_res_all[:, :, None], 
final_nmi[:, :, None], final_nmi_all[:, :, None], final_dir_mod[:, :, None], final_dir_mod_all[:, :, None],
CI_full, size_ratio[:, :, None]), 2), compare_names_all, metric_names, False)

np.save(dir_name+'test'+save_name, final_res)
np.save(dir_name+'all'+save_name, final_res_all)
np.save(dir_name+'test_acc'+save_name, final_acc)
np.save(dir_name+'all_acc'+save_name, final_acc_all)
np.save(dir_name+'test_nmi'+save_name, final_nmi)
np.save(dir_name+'all_nmi'+save_name, final_nmi_all)
np.save(dir_name+'test_dir_mod'+save_name, final_dir_mod)
np.save(dir_name+'all_dir_mod'+save_name, final_dir_mod_all)
np.save(dir_name+'CI'+save_name, CI_full)
np.save(dir_name+'labels_distribution'+save_name, labels_distribution_full)
np.save(dir_name+'flow'+save_name, flow_mat_full)
np.save(dir_name+'pairwise'+save_name, pairwise_imbalance_array_full)
