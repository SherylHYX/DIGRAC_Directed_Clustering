# external files
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import torch.nn.functional as F
from texttable import Texttable
from sklearn.metrics import adjusted_rand_score, accuracy_score, normalized_mutual_info_score
from torch_geometric.nn import GCNConv

# internal files
from utils import write_log, scipy_sparse_to_torch_sparse, get_powers_sparse
from metrics import print_performance_mean_std, Prob_Imbalance_Loss, label_size_ratio
from metrics import directed_modularity, clustering_evaluation, get_imbalance_distribution_and_flow
from GNN_models import DIGRAC, DiGCN_Inception_Block, DGCN
from param_parser import parameter_parser
from preprocess import load_data
from features_in_out import directed_features_in_out
from get_adj import get_second_directed_adj, cal_fast_appr
from comparison import Cluster
from MagNet import MagNet
from model_digcl import Encoder, drop_feature, Model
from eval_digcl import pred_digcl
from extra_comparison import InfoMap, leidenalg_modularity, louvain_modularity


typical_GNN_names = ['DIGRAC', 'DiGCN', 'DGCN', 'MagNet']
args = parameter_parser()
torch.manual_seed(args.seed)
device = args.device
if args.cuda:
    print("Using cuda")
    torch.cuda.manual_seed(args.seed)
compare_names_all = []
for model_name in args.all_methods:
    if model_name not in typical_GNN_names:
        compare_names_all.append(model_name)
    else:
        for normalization in args.normalizations:
            for threshold in args.thresholds:
                compare_names_all.append(model_name+'_'+normalization+'_'+threshold)

criterion = torch.nn.NLLLoss()
class Trainer(object):
    """
    Object to train and score different models.
    """

    def __init__(self, args, random_seed, save_name_base):
        """
        Constructing the trainer instance.
        :param args: Arguments object.
        """
        self.args = args
        self.device = args.device
        self.random_seed = random_seed

        self.label, self.train_mask, self.val_mask, self.test_mask, self.features, self.A = load_data(args, random_seed)
        self.features = torch.FloatTensor(self.features).to(args.device)
        self.args.N = self.A.shape[0]
        self.A_torch = scipy_sparse_to_torch_sparse(self.A).to(args.device)
        
        self.nfeat = self.features.shape[1]
        if self.label is None:
            self.label = -np.ones(self.A.shape[0])
            assert self.args.CE_loss_coeff == 0, 'Cannot conduct supervised training without label!'
        self.label = torch.LongTensor(self.label).to(args.device)
        
        self.num_clusters = self.args.K
        self.c =  Cluster(self.A, int(self.num_clusters))

        date_time = datetime.now().strftime('%m-%d-%H:%M:%S')

        

        save_name = save_name_base + 'Seed' + str(random_seed)

        self.log_path = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), args.log_root, args.dataset, save_name, date_time)

        if os.path.isdir(self.log_path) == False:
            try:
                os.makedirs(self.log_path)
            except FileExistsError:
                print('Folder exists!')

        self.splits = self.args.num_trials
        if self.test_mask is not None and self.test_mask.ndim == 1:
            self.train_mask = np.repeat(
                self.train_mask[:, np.newaxis], self.splits, 1)
            self.val_mask = np.repeat(
                self.val_mask[:, np.newaxis], self.splits, 1)
            self.test_mask = np.repeat(
                self.test_mask[:, np.newaxis], self.splits, 1)
        write_log(vars(args), self.log_path)  # write the setting

    def DiGCL(self, train_index):
        edge_index = torch.LongTensor(self.A.nonzero()).to(self.args.device)
        edge_weights = torch.FloatTensor(self.A.data).to(self.args.device)
        alpha_1 = 0.1
        drop_feature_rate_1 = 0.3
        drop_feature_rate_2 = 0.4

        edge_index_init, edge_weight_init = cal_fast_appr(
            alpha_1, edge_index, self.features.shape[0], self.features.dtype, edge_weight=edge_weights)
        x = self.features
        encoder = Encoder(self.nfeat, self.args.hidden*2, F.relu,
                        base_model=GCNConv, k=2).to(self.args.device)
        model = Model(encoder, self.args.hidden*2, self.args.hidden, 0.4).to(self.args.device)
        a = 0.9
        b = 0.1
        alpha_2 = a - (a-b)*(1/3*np.log(self.args.epochs/(self.args.epochs+1)+np.exp(-3)))
        edge_index_1, edge_weight_1 = cal_fast_appr(
            alpha_1, edge_index, x.shape[0], x.dtype, edge_weight=edge_weights)
        edge_index_2, edge_weight_2 = cal_fast_appr(
            alpha_2, edge_index, x.shape[0], x.dtype, edge_weight=edge_weights)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=0.00001)
        for _ in range(600):
            x_1 = drop_feature(x, drop_feature_rate_1)
            x_2 = drop_feature(x, drop_feature_rate_2)

            z1 = model(x_1, edge_index_1, edge_weight_1)
            z2 = model(x_2, edge_index_2, edge_weight_2)

            loss = model.loss(z1, z2, batch_size=0)
            loss.backward()
            optimizer.step()
        # test
        model.eval()
        z = model(x, edge_index_init, edge_weight_init)
        pred = pred_digcl(z, self.label.view(-1), train_index)
        return pred

    def train(self, model_name, normalization='plain', threshold='sort'):
        #################################
        # training and evaluation
        #################################
        if model_name not in typical_GNN_names:
            res_full, CI_full, labels_distribution_full, \
            flow_mat_full, pairwise_imbalance_array_full = self.others(model_name)
            results = (res_full, res_full)
            imbalance = (CI_full, CI_full)
            labels_distributions = (labels_distribution_full,
                                    labels_distribution_full)
            flow_matrices = (flow_mat_full, flow_mat_full)
            pairwise_imbalance = (pairwise_imbalance_array_full, pairwise_imbalance_array_full)
        else:
            if self.args.CE_loss_coeff + self.args.imbalance_coeff == 0:
                raise ValueError('Incorrect loss combination!')

            # (the last two dimensions) rows: test, val, all; cols: ARI, NMI, accuracy, directed modularity
            res_full = np.zeros([self.splits, 3, 4])
            res_full[:] = np.nan
            res_full_latest = res_full.copy()
            CI_full = np.zeros([self.splits, len(
                self.args.report_normalizations)*len(self.args.report_thresholds)])
            CI_full_latest = CI_full.copy()
            labels_distribution_full = np.zeros(
                [self.splits, self.num_clusters])
            labels_distribution_full_latest = labels_distribution_full.copy()
            flow_mat_full = np.zeros(
                [self.splits, self.num_clusters, self.num_clusters])
            flow_mat_full_latest = flow_mat_full.copy()
            pairwise_imbalance_array_full = np.zeros(
                [self.splits, len(self.args.report_normalizations), int(self.num_clusters*(self.num_clusters-1)/2)]
            )
            pairwise_imbalance_array_full_latest = pairwise_imbalance_array_full.copy()
            
            args = self.args
            device = args.device
            A = scipy_sparse_to_torch_sparse(self.A).to(self.args.device)
            if model_name == 'DiGCN':
                edge_index = torch.LongTensor(self.A.nonzero())
                edge_weights = torch.FloatTensor(self.A.data)
                edge_index1 = edge_index.clone().to(self.args.device)
                edge_weights1 = edge_weights.clone().to(self.args.device)
                edge_index2, edge_weights2 = get_second_directed_adj(edge_index, self.features.shape[0],self.features.dtype,
                edge_weights)
                edge_index2 = edge_index2.to(self.args.device)
                edge_weights2 = edge_weights2.to(self.args.device)
                edge_index = (edge_index1, edge_index2)
                edge_weights = (edge_weights1, edge_weights2)
                del edge_index2, edge_weights2
            elif model_name == 'DGCN':
                edge_index = torch.LongTensor(self.A.nonzero())
                edge_weights = torch.FloatTensor(self.A.data)
                edge_index, edge_in, in_weight, edge_out, out_weight = directed_features_in_out(edge_index, self.A.shape[0], edge_weights)
                edge_index = edge_index.to(self.args.device)
                edge_in, in_weight, edge_out, out_weight = edge_in.to(device), in_weight.to(device), edge_out.to(device), out_weight.to(device)
                edge_index = (edge_index, edge_in, edge_out)
                edge_weights = (in_weight, out_weight)
            else:
                edge_index = torch.LongTensor(self.A.nonzero()).to(self.args.device)
                edge_weights = torch.FloatTensor(self.A.data).to(self.args.device)
            for split in range(self.splits):
                if model_name == 'DIGRAC':
                    model = DIGRAC(nfeat=self.nfeat, dropout=self.args.dropout, hop=self.args.hop, fill_value=self.args.tau, 
                        hidden=self.args.hidden, nclass=self.num_clusters).to(self.args.device)
                elif model_name == 'DiGCN':
                    model = DiGCN_Inception_Block(num_features=self.nfeat, dropout=self.args.dropout,
                        embedding_dim=self.args.hidden*2, prob_dim=self.num_clusters).to(self.args.device)
                elif model_name == 'DGCN':
                    model = DGCN(self.nfeat, self.args.hidden*2, self.num_clusters,
                        self.args.dropout_d).to(self.args.device)
                elif model_name == 'MagNet':
                    model = MagNet(in_channels=self.nfeat, num_filter=int(self.args.hidden/2), q=0.25, K=2, label_dim=self.num_clusters, \
                        activation=self.args.activation, dropout=self.args.dropout).to(self.args.device)
                else:
                    raise NameError('Please input the correct model name instead of {}!'.format(model_name))

                if self.args.optimizer == 'Adam':
                    opt = optim.Adam(model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
                elif self.args.optimizer == 'SGD':
                    opt = optim.SGD(model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
                else:
                    raise NameError('Please input the correct optimizer name, Adam or SGD!')

                if self.test_mask is not None:
                    train_index = self.train_mask[:, split]
                    val_index = self.val_mask[:, split]
                    test_index = self.test_mask[:, split]
                    if args.AllTrain:
                        # to use all nodes
                        train_index[:] = True
                        val_index[:] = True
                        test_index[:] = True
                    train_A = scipy_sparse_to_torch_sparse(self.A[train_index][:, train_index]).to(self.args.device)
                    val_A = scipy_sparse_to_torch_sparse(self.A[val_index][:, val_index]).to(self.args.device)
                    test_A = scipy_sparse_to_torch_sparse(self.A[test_index][:, test_index]).to(self.args.device)
                else:
                    train_index = np.ones(args.N, dtype=bool)
                    val_index = train_index
                    test_index = train_index
                    train_A = scipy_sparse_to_torch_sparse(self.A).to(self.args.device)
                    val_A = train_A
                    test_A = train_A
                #################################
                # Train/Validation/Test
                #################################
                best_val_loss = 1000.0
                early_stopping = 0
                log_str_full = ''
                imbalance_loss_func = Prob_Imbalance_Loss(args.F)
                imbalance_loss_pretrain = Prob_Imbalance_Loss(3)

                for epoch in range(args.epochs):
                    start_time = time.time()
                    ####################
                    # Train
                    ####################
                    model.train()
                    log_prob = model(self.features, edge_index, edge_weights)
                    prob = torch.exp(log_prob)
                       
                    if self.args.CE_loss_coeff > 0:
                        train_loss_CE = criterion(log_prob[train_index], self.label[train_index])
                    else:
                        train_loss_CE = torch.ones(1, requires_grad=True).to(device)
                    
                    if self.args.imbalance_coeff > 0:
                        if epoch < self.args.pretrain_epochs and threshold == 'std':
                            train_loss_imbalance = imbalance_loss_pretrain(prob[train_index], train_A, self.num_clusters, normalization, 'sort')
                        else:
                            train_loss_imbalance = imbalance_loss_func(prob[train_index], train_A, self.num_clusters, normalization, threshold)
                    else:
                        train_loss_imbalance = torch.ones(1, requires_grad=True).to(device)

                    train_loss = self.args.CE_loss_coeff * train_loss_CE + self.args.imbalance_coeff * train_loss_imbalance

                    outstrtrain = 'Train loss:, {:.6f}, imbalance loss: {:.6f}, CE loss: {:6f},'.format(train_loss.detach().item(),
                    train_loss_imbalance.detach().item(), train_loss_CE.detach().item())
                    opt.zero_grad()
                    try:
                        train_loss.backward()
                    except RuntimeError:
                        log_str = '{} trial {} RuntimeError!'.format(model_name, split)
                        log_str_full += log_str + '\n'
                        print(log_str)
                        if not os.path.isfile(self.log_path + '/'+model_name+'_'+normalization+'_'+threshold+'_model'+str(split)+'.t7'):
                                torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_'+normalization+'_'+threshold+'_model'+str(split)+'.t7')
                        torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_'+normalization+'_'+threshold+'_model_latest'+str(split)+'.t7')
                        break
                    opt.step()
                    ####################
                    # Validation
                    ####################
                    model.eval()

                    log_prob = model(self.features, edge_index, edge_weights)
                    prob = torch.exp(log_prob)
                       
                    if self.args.CE_loss_coeff > 0:
                        val_loss_CE = criterion(log_prob[val_index], self.label[val_index])
                    else:
                        val_loss_CE = torch.ones(1, requires_grad=True).to(device)
                    
                    if self.args.imbalance_coeff > 0:
                        if epoch < self.args.pretrain_epochs and threshold == 'std':
                            val_loss_imbalance = imbalance_loss_pretrain(prob[val_index], val_A, self.num_clusters, normalization, 'sort')
                        else:
                            val_loss_imbalance = imbalance_loss_func(prob[val_index], val_A, self.num_clusters, normalization, threshold)
                    else:
                        val_loss_imbalance = torch.ones(1, requires_grad=True).to(device)

                    val_loss = self.args.CE_loss_coeff * val_loss_CE + self.args.imbalance_coeff * val_loss_imbalance

                    outstrval = 'val loss:, {:.6f}, imbalance loss: {:.6f}, CE loss: {:6f},,'.format(val_loss.detach().item(),
                    val_loss_imbalance.detach().item(), val_loss_CE.detach().item())
                    duration = "---, {:.4f}, seconds ---".format(
                        time.time() - start_time)
                    log_str = ("{}, / {} epoch,".format(epoch, args.epochs)) + \
                        outstrtrain+outstrval+duration
                    log_str_full += log_str + '\n'
                    print(log_str)
                    
                    ####################
                    # Save weights
                    ####################
                    save_perform = val_loss.detach().item()
                    if save_perform <= best_val_loss:
                        early_stopping = 0
                        best_val_loss = save_perform
                        torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_'+normalization+'_'+threshold+'_model'+str(split)+'.t7')
                    else:
                        early_stopping += 1
                    if early_stopping > args.early_stopping or epoch == (args.epochs-1):
                        torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_'+normalization+'_'+threshold+'_model_latest'+str(split)+'.t7')
                        break

                status = 'w'
                if os.path.isfile(self.log_path + '/'+model_name+'_'+normalization+'_'+threshold+'_log'+str(split)+'.csv'):
                    status = 'a'
                with open(self.log_path + '/'+model_name+'_'+normalization+'_'+threshold+'_log'+str(split)+'.csv', status) as file:
                    file.write(log_str_full)
                    file.write('\n')
                    status = 'a'

                ####################
                # Testing
                ####################
                base_save_path = self.log_path + '/'+model_name+'_'+normalization+'_'+threshold
                logstr = ''
                model.load_state_dict(torch.load(
                    self.log_path + '/'+model_name+'_'+normalization+'_'+threshold+'_model'+str(split)+'.t7'))
                model.eval()

                log_prob = model(self.features, edge_index, edge_weights)
                prob = torch.exp(log_prob)
                    
                if self.args.CE_loss_coeff > 0:
                    val_loss_CE = criterion(log_prob[val_index], self.label[val_index])
                    test_loss_CE = criterion(log_prob[test_index], self.label[test_index])
                    all_loss_CE = criterion(log_prob, self.label)
                else:
                    val_loss_CE = torch.ones(1, requires_grad=True).to(device)
                    test_loss_CE = val_loss_CE
                    all_loss_CE = val_loss_CE
                
                if self.args.imbalance_coeff > 0:
                    val_loss_imbalance = imbalance_loss_func(prob[val_index], val_A, self.num_clusters, normalization, threshold)
                    test_loss_imbalance = imbalance_loss_func(prob[test_index], test_A, self.num_clusters, normalization, threshold)
                    all_loss_imbalance = imbalance_loss_func(prob, self.A_torch, self.num_clusters, normalization, threshold)
                else:
                    val_loss_imbalance = torch.ones(1, requires_grad=True).to(device)
                    test_loss_imbalance = val_loss_imbalance
                    all_loss_imbalance = val_loss_imbalance

                val_loss = self.args.CE_loss_coeff * val_loss_CE + self.args.imbalance_coeff * val_loss_imbalance
                test_loss = self.args.CE_loss_coeff * test_loss_CE + self.args.imbalance_coeff * test_loss_imbalance
                all_loss = self.args.CE_loss_coeff * all_loss_CE + self.args.imbalance_coeff * all_loss_imbalance


                logstr += 'Final results for {}:,'.format(model_name)
                logstr += 'Best val loss: ,{:.3f}, test loss: ,{:.3f}, all loss: ,{:.3f},'.format(val_loss.detach().item(), test_loss.detach().item(), all_loss.detach().item())

                prob_np = prob.detach().to('cpu')
                pred_label = np.argmax(prob_np, axis = 1)

                if self.args.SavePred:
                    np.save(self.log_path + '/'+model_name+'_pred' +
                            str(split), pred_label.view(-1).to('cpu'))


                val_dir_mod, val_ARI, val_NMI, val_acc = clustering_evaluation(self.A[val_index][:,val_index],
                    pred_label.view(-1).to('cpu')[val_index], self.label.view(-1).to('cpu')[val_index])
                test_dir_mod, test_ARI, test_NMI, test_acc = clustering_evaluation(self.A[test_index][:,test_index],
                    pred_label.view(-1).to('cpu')[test_index], self.label.view(-1).to('cpu')[test_index])
                all_dir_mod, all_ARI, all_NMI, all_acc = clustering_evaluation(self.A,
                    pred_label.view(-1).to('cpu'), self.label.view(-1).to('cpu'))
                imbalance_list, labels_distribution, flow_mat, pairwise_imbalance_array = get_imbalance_distribution_and_flow(pred_label.view(-1).to('cpu'),
                                                                                                    self.num_clusters, A, self.args.F, self.args.report_normalizations, self.args.report_thresholds)
                CI_full[split] = imbalance_list
                labels_distribution_full[split] = labels_distribution
                flow_mat_full[split] = flow_mat
                pairwise_imbalance_array_full[split] = pairwise_imbalance_array
                logstr = 'best imbalance values, {}'.format(imbalance_list)+'\n'
                logstr += 'labels distribution is,{}'.format(
                    labels_distribution)+'\n'
                logstr += 'Predicted meta-graph flow matrix is {}'.format(
                    flow_mat)+'\n'

                # latest
                model.load_state_dict(torch.load(self.log_path +
                                '/'+model_name+'_'+normalization+'_'+threshold+'_model_latest'+str(split)+'.t7'))
                model.eval()
                preds = model(self.features, edge_index, edge_weights)
                pred_label = preds.max(dim=1)[1]

                val_dir_mod_latest, val_ARI_latest, val_NMI_latest, val_acc_latest = clustering_evaluation(self.A[val_index][:,val_index],
                    pred_label.view(-1).to('cpu')[val_index], self.label.view(-1).to('cpu')[val_index])
                test_dir_mod_latest, test_ARI_latest, test_NMI_latest, test_acc_latest = clustering_evaluation(self.A[test_index][:,test_index],
                    pred_label.view(-1).to('cpu')[test_index], self.label.view(-1).to('cpu')[test_index])
                all_dir_mod_latest, all_ARI_latest, all_NMI_latest, all_acc_latest = clustering_evaluation(self.A,
                    pred_label.view(-1).to('cpu'), self.label.view(-1).to('cpu'))
                imbalance_list, labels_distribution, flow_mat, pairwise_imbalance_array = get_imbalance_distribution_and_flow(pred_label.view(-1).to('cpu'),
                                                                                                    self.num_clusters, A, self.args.F, self.args.report_normalizations, self.args.report_thresholds)
                CI_full_latest[split] = imbalance_list
                labels_distribution_full_latest[split] = labels_distribution
                flow_mat_full_latest[split] = flow_mat
                pairwise_imbalance_array_full_latest[split] = pairwise_imbalance_array
                logstr += 'latest imbalance values, {}'.format(imbalance_list)+'\n'
                logstr += 'labels distribution is,{}'.format(
                    labels_distribution)+'\n'
                logstr += 'Predicted meta-graph flow matrix is {}'.format(
                    flow_mat)+'\n'

                ####################
                # Save testing results
                ####################
                logstr += 'val_ARI:, '+str(np.round(val_ARI, 3))+' ,test_ARI: ,'+str(np.round(test_ARI, 3))+' ,val_ARI_latest: ,'+str(
                    np.round(val_ARI_latest, 3))+' ,test_ARI_latest: ,'+str(np.round(test_ARI_latest, 3))
                logstr += ' ,all_ARI: ,' + \
                    str(np.round(all_ARI, 3))+', all_ARI_latest: ,' + \
                    str(np.round(all_ARI_latest, 3))
                logstr += 'val_acc:, '+str(np.round(val_acc, 3))+' ,test_acc: ,'+str(np.round(test_acc, 3))+' ,val_acc_latest: ,'+str(
                    np.round(val_acc_latest, 3))+' ,test_acc_latest: ,'+str(np.round(test_acc_latest, 3))
                logstr += ' ,all_acc: ,' + \
                    str(np.round(all_acc, 3))+', all_acc_latest: ,' + \
                    str(np.round(all_acc_latest, 3))
                print(logstr)
                res_full[split] = [[val_ARI, val_NMI, val_acc, val_dir_mod],
                [test_ARI, test_NMI, test_acc, test_dir_mod],
                [all_ARI, all_NMI, all_acc, all_dir_mod]]
                res_full_latest[split] = [[val_ARI_latest, val_NMI_latest, val_acc_latest, val_dir_mod_latest],
                [test_ARI_latest, test_NMI_latest, test_acc_latest, test_dir_mod_latest],
                [all_ARI_latest, all_NMI_latest, all_acc_latest, all_dir_mod_latest]]
                with open(self.log_path + '/'+model_name+'_'+normalization+'_'+threshold+ '_log'+str(split)+'.csv', status) as file:
                    file.write(logstr)
                    file.write('\n')
                torch.cuda.empty_cache()
                results = (res_full, res_full_latest)
                imbalance = (CI_full, CI_full_latest)
                labels_distributions = (labels_distribution_full,
                                        labels_distribution_full_latest)
                flow_matrices = (flow_mat_full, flow_mat_full_latest)
                pairwise_imbalance = (pairwise_imbalance_array_full, pairwise_imbalance_array_full_latest)
        return results, imbalance, labels_distributions, flow_matrices, pairwise_imbalance

    def others(self, model_name):
        #################################
        # training and evaluation for non-NN methods
        #################################
        # (the last two dimensions) rows: test, val, all; cols: ARI, NMI, accuracy, directed modularity
        res_full = np.zeros([self.splits, 3, 4])
        res_full[:] = np.nan
        CI_full = np.zeros([self.splits, len(
            self.args.report_normalizations)*len(self.args.report_thresholds)])
        labels_distribution_full = np.zeros(
            [self.splits, self.num_clusters])
        flow_mat_full = np.zeros(
            [self.splits, self.num_clusters, self.num_clusters])
        pairwise_imbalance_array_full = np.zeros(
            [self.splits, len(self.args.report_normalizations), int(self.num_clusters*(self.num_clusters-1)/2)]
        )
        
        for split in range(self.splits):
            if self.test_mask is not None:
                train_index = self.train_mask[:, split]
                val_index = self.val_mask[:, split]
                test_index = self.test_mask[:, split]
                if args.AllTrain:
                    # to use all nodes
                    train_index[:] = True
                    val_index[:] = True
                    test_index[:] = True
            else:
                val_index = np.ones(args.N, dtype=bool)
                test_index = val_index
                train_index = val_index

            ####################
            # Testing
            ####################
            logstr = ''
            try:
                if model_name == 'Bi_sym':
                    pred = self.c.spectral_cluster_Bi_sym()
                elif model_name == 'DD_sym':
                    pred = self.c.spectral_cluster_DD_sym()
                elif model_name == 'DISG_LR':
                    pred = self.c.spectral_cluster_DISG_LR()
                elif model_name == 'Herm':
                    pred = self.c.spectral_cluster_herm()
                elif model_name == 'Herm_rw':
                    pred = self.c.spectral_cluster_herm_rw()
                elif model_name == 'InfoMap':
                    pred = InfoMap(self.A)
                elif model_name == 'Louvain':
                    pred = louvain_modularity(self.A)
                elif model_name == 'Leiden':
                    pred = leidenalg_modularity(self.A)
                elif model_name == 'DiGCL':
                    pred = self.DiGCL(train_index)
                pred = np.array(pred - pred.min(), dtype=int)
                num_clusters = pred.max() + 1
                if num_clusters > 1:
                    try:
                        imbalance_list, labels_distribution, flow_mat, pairwise_imbalance_array = get_imbalance_distribution_and_flow(pred,
                                                                                                        num_clusters, self.A_torch, self.args.F, self.args.report_normalizations, self.args.report_thresholds)
                        min_num_clusters = int(min(num_clusters, self.num_clusters))
                        min_pairwise_num = int(min_num_clusters * (min_num_clusters - 1)/2)
                        CI_full[split] = imbalance_list
                        labels_distribution_full[split][:min_num_clusters] = labels_distribution[:min_num_clusters]
                        flow_mat_full[split][:min_num_clusters] = flow_mat[:min_num_clusters]
                        pairwise_imbalance_array_full[split][:, :min_pairwise_num] = pairwise_imbalance_array[:, :min_pairwise_num]

                    except RuntimeError:
                        CI_full[split] = np.nan
                        labels_distribution_full[split] = np.nan
                        flow_mat_full[split] = np.nan
                        pairwise_imbalance_array_full[split] = np.nan
                else:
                    CI_full[split] = np.nan
                    labels_distribution_full[split] = np.nan
                    flow_mat_full[split] = np.nan
                    pairwise_imbalance_array_full[split] = np.nan
            except Exception:
                CI_full[split] = np.nan
                labels_distribution_full[split] = np.nan
                flow_mat_full[split] = np.nan
                pairwise_imbalance_array_full[split] = np.nan
                pred = np.zeros(self.A.shape[0], dtype=int)


            val_dir_mod, val_ARI, val_NMI, val_acc = clustering_evaluation(self.A[val_index][:,val_index],
                    pred[val_index], self.label.view(-1).to('cpu')[val_index])
            test_dir_mod, test_ARI, test_NMI, test_acc = clustering_evaluation(self.A[test_index][:,test_index],
                pred[test_index], self.label.view(-1).to('cpu')[test_index])
            all_dir_mod, all_ARI, all_NMI, all_acc = clustering_evaluation(self.A,
                pred, self.label.view(-1).to('cpu'))
            res_full[split] = [[val_ARI, val_NMI, val_acc, val_dir_mod],
                [test_ARI, test_NMI, test_acc, test_dir_mod],
                [all_ARI, all_NMI, all_acc, all_dir_mod]]
            if self.args.SavePred:
                np.save(self.log_path + '/'+model_name+'_pred'+str(split), pred)
            logstr = model_name + ': '
            logstr += 'val_ARI:, '+str(np.round(val_ARI, 3))+' ,test_ARI: ,'+str(np.round(test_ARI, 3))
            logstr += ' ,all_ARI: ,' + str(np.round(all_ARI, 3))
            logstr += ',val_acc:, '+str(np.round(val_acc, 3))+' ,test_acc: ,'+str(np.round(test_acc, 3))
            logstr += ' ,all_acc: ,' + str(np.round(all_acc, 3))
            print(logstr)
            with open(self.log_path + '/' + model_name + '_log'+str(split)+'.csv', 'w') as file:
                file.write(logstr)
                file.write('\n')
        return res_full, CI_full, labels_distribution_full, \
            flow_mat_full, pairwise_imbalance_array_full


# train and grap results
if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/debug/'+args.dataset)
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/'+args.dataset)


res = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), 3, 4])
res[:] = np.nan
res_latest = res.copy()

CI_full = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), len(
    args.report_normalizations)*len(args.report_thresholds)])
labels_distribution_full = np.zeros([len(compare_names_all),
    args.num_trials*len(args.seeds),  args.K])
flow_mat_full = np.zeros([len(
    compare_names_all), args.num_trials*len(args.seeds), args.K, args.K])
pairwise_imbalance_array_full = np.zeros(
            [len(compare_names_all), args.num_trials*len(args.seeds), len(args.report_normalizations), int(args.K*(args.K-1)/2)])
CI_full_latest = CI_full.copy()
labels_distribution_full_latest = labels_distribution_full.copy()
flow_mat_full_latest = flow_mat_full.copy()
pairwise_imbalance_array_full_latest = pairwise_imbalance_array_full.copy()

method_str = ''
for model_name in args.all_methods:
    method_str += model_name
if len(list(set(args.all_methods).intersection(set(typical_GNN_names)))) > 0:
    method_str += 'normalizations_'
    for normalization in args.normalizations:
        method_str += normalization
    method_str += 'thresholds_'
    for threshold in args.thresholds:
        method_str += threshold   

default_name_base = ''
if len(list(set(args.all_methods).intersection(set(typical_GNN_names)))) > 0:
    default_name_base += 'dropout' + str(int(100*args.dropout))
    default_name_base += 'imb_coe' + str(int(100*args.imbalance_coeff)) + 'CE_coe' + str(int(100*args.CE_loss_coeff))
    default_name_base += 'optimizer' + str(args.optimizer) + 'hid' + str(args.hidden) + 'lr' + str(int(1000*args.lr))
    if args.pretrain_epochs > 0:
        default_name_base +=  'pre' + str(int(args.pretrain_epochs))
save_name_base = default_name_base

default_name_base +=  'trials' + str(args.num_trials) + 'train_r' + str(int(100*args.train_ratio)) + 'test_r' + str(int(100*args.test_ratio)) + 'All' + str(args.AllTrain)
save_name_base = default_name_base
if args.dataset[:4] == 'DSBM':
    default_name_base += 'seeds' + '_'.join([str(value) for value in np.array(args.seeds).flatten()])
save_name = default_name_base


current_seed_ind = 0
for random_seed in args.seeds:
    current_ind = 0
    trainer = Trainer(args, random_seed, save_name_base)
    for model_name in args.all_methods:
        if model_name not in typical_GNN_names:
            results, imbalance, labels_distributions, flow_matrices, pairwise_imbalance = trainer.train(model_name)
            res[current_ind, current_seed_ind: current_seed_ind + args.num_trials], res_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = results
            CI_full[current_ind, current_seed_ind: current_seed_ind + args.num_trials], CI_full_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = imbalance
            labels_distribution_full[current_ind, current_seed_ind: current_seed_ind + args.num_trials], labels_distribution_full_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = labels_distributions
            flow_mat_full[current_ind, current_seed_ind: current_seed_ind + args.num_trials], flow_mat_full_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = flow_matrices
            pairwise_imbalance_array_full[current_ind, current_seed_ind: current_seed_ind + args.num_trials], pairwise_imbalance_array_full_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = pairwise_imbalance
            current_ind = current_ind + 1
        else:
            for normalization in args.normalizations:
                for threshold in args.thresholds:
                    results, imbalance, labels_distributions, flow_matrices, pairwise_imbalance = trainer.train(model_name, normalization, threshold)
                    res[current_ind, current_seed_ind: current_seed_ind + args.num_trials], res_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = results
                    CI_full[current_ind, current_seed_ind: current_seed_ind + args.num_trials], CI_full_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = imbalance
                    labels_distribution_full[current_ind, current_seed_ind: current_seed_ind + args.num_trials], labels_distribution_full_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = labels_distributions
                    flow_mat_full[current_ind, current_seed_ind: current_seed_ind + args.num_trials], flow_mat_full_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = flow_matrices
                    pairwise_imbalance_array_full[current_ind, current_seed_ind: current_seed_ind + args.num_trials], pairwise_imbalance_array_full_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = pairwise_imbalance
                    current_ind = current_ind + 1
    current_seed_ind += args.num_trials

# print results and save results to arrays
t = Texttable(max_width=120)
t.add_rows([["Parameter","pretrain epochs","hop","tau",
"train ratio",  "lr", "hidden","imbalance loss coeff","CE loss coeff"],
["Values",args.pretrain_epochs, args.hop,args.tau, 
args.train_ratio, args.lr, args.hidden, args.imbalance_coeff, args.CE_loss_coeff]])

print(t.draw())

for save_dir_name in ['res', 'CI', 'labels_distribution', 'flow', 'pairwise']:
    if os.path.isdir(os.path.join(dir_name,save_dir_name,method_str)) == False:
        try:
            os.makedirs(os.path.join(dir_name,save_dir_name,method_str))
        except FileExistsError:
            print('Folder exists for best {}!'.format(save_dir_name))
    if os.path.isdir(os.path.join(dir_name,save_dir_name+'_latest',method_str)) == False:
        try:
            os.makedirs(os.path.join(dir_name,save_dir_name+'_latest',method_str))
        except FileExistsError:
            print('Folder exists for latest {}!'.format(save_dir_name))

np.save(os.path.join(dir_name,'res',method_str,save_name), res)
np.save(os.path.join(dir_name,'res_latest',method_str,save_name), res_latest)
np.save(os.path.join(dir_name,'CI',method_str,save_name), CI_full)
np.save(os.path.join(dir_name,'CI_latest',method_str,save_name), CI_full_latest)
np.save(os.path.join(dir_name,'labels_distribution',method_str,save_name), labels_distribution_full)
np.save(os.path.join(dir_name,'labels_distribution_latest',method_str,save_name), labels_distribution_full_latest)
np.save(os.path.join(dir_name,'flow',method_str,save_name), flow_mat_full)
np.save(os.path.join(dir_name,'flow_latest',method_str,save_name), flow_mat_full_latest)
np.save(os.path.join(dir_name,'pairwise',method_str,save_name), pairwise_imbalance_array_full)
np.save(os.path.join(dir_name,'pairwise_latest',method_str,save_name), pairwise_imbalance_array_full_latest)


new_shape_res = (args.num_trials*len(args.seeds), len(compare_names_all), 12)

metric_names = ['test ARI', 'test NMI', 'test acc', 'test dir mod', \
'val ARI', 'val NMI', 'val acc', 'val dir mod', \
'all ARI', 'all NMI', 'all acc', 'all dir mod']
for threshold in args.report_thresholds:
    for normalization in args.report_normalizations:
        metric_names.append(normalization + '_' + threshold)
metric_names.append('size_ratio')

size_ratio = label_size_ratio(labels_distribution_full_latest.swapaxes(0,1))
print_performance_mean_std(args.dataset[:-1]+'_latest', np.concatenate((res_latest.swapaxes(0,1).reshape(new_shape_res),
CI_full_latest.swapaxes(0,1), size_ratio[:, :, None]), 2), compare_names_all, metric_names, False)

print('\n')

size_ratio = label_size_ratio(labels_distribution_full.swapaxes(0,1))
print_performance_mean_std(args.dataset[:-1]+'_best', np.concatenate((res.swapaxes(0,1).reshape(new_shape_res),
CI_full.swapaxes(0,1), size_ratio[:, :, None]), 2), compare_names_all, metric_names, False)
