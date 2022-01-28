import argparse
import os

import torch

from utils import meta_graph_generation

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run DIGRAC on DSBMs.")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--debug', '-D',action='store_true', default=False,
                        help='Debugging mode, minimal setting.')
    parser.add_argument('--seed', type=int, default=31, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, #default = 0.01
                        help='Initial learning rate.')
    parser.add_argument('--samples', type=int, default=10000, # with few seeds, set the number of samples smaller
                        help='Samples per triplet loss.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--normalizations",
                        nargs="+",
                        type=str,
                        help="Normalization methods to choose from: vol_min, vol_sum, vol_max and None.")
    parser.add_argument("--thresholds",
                        nargs="+",
                        type=str,
                        help="Thresholding methods to choose from: sort, std and None.")
    parser.set_defaults(normalizations=['vol_sum'])
    parser.set_defaults(thresholds=['sort'])
    parser.add_argument("--all_methods",
                        nargs="+",
                        type=str,
                        help="Methods to use.")
    parser.set_defaults(all_methods=['Herm_rw','DIGRAC'])
    parser.add_argument("--report_normalizations",
                        nargs="+",
                        type=str,
                        help="Normalization methods to generate report.")
    parser.add_argument("--report_thresholds",
                        nargs="+",
                        type=str,
                        help="Thresholding methods to generate report.")
    parser.set_defaults(report_normalizations=['vol_sum','vol_min','vol_max','plain'])
    parser.set_defaults(report_thresholds=['sort', 'std', 'naive'])
    parser.add_argument('--alpha', type=float, default=0,
                        help='Threshold in triplet loss for seeds.') 

    # synthetic model hyperparameters below
    parser.add_argument('--p', type=float, default=0.02,
                        help='Probability of the existence of a link within communities, with probability (1-p), we have 0.')
    parser.add_argument('--N', type=int, default=1000,
                        help='Number of nodes in the directed stochastic block model.')
    parser.add_argument('--K', type=int, default=3,
                        help='Number of clusters.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training ratio during data split.')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test ratio during data split.')
    parser.add_argument('--hop', type=int, default=2,
                        help='Number of hops to consider for the random walk.') 
    parser.add_argument('--tau', type=float, default=0.5,
                        help='The regularization parameter when adding self-loops to an adjacency matrix, i.e. A -> A + tau * I, where I is the identity matrix.')
    parser.add_argument('--imbalance_coeff', type=float, default=1,
                        help='Imbalance loss coefficient.')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use. Adam or SGD in our case.')
    parser.add_argument('--CE_loss_coeff', type=float, default=0,
                        help='Ratio of factor of supervised loss part, cross-entropy loss.')
    parser.add_argument('--seed_ratio', type=float, default=0.1,
                        help='The ratio in the training set of each cluster to serve as seed nodes.')
    parser.add_argument('--size_ratio', type=float, default=1.5,
                        help='The size ratio of the largest to the smallest block. 1 means uniform sizes. should be at least 1.')
    parser.add_argument('--num_trials', type=int, default=2,
                        help='Number of trials to generate results.')      
    parser.add_argument('--F', default=9,
                        help='Meta-graph adjacency matrix or the number of pairs to consider, array or int.')
    parser.add_argument('--F_style', type=str, default='cyclic',
                        help='Meta-graph adjacency matrix style.')
    parser.add_argument('--ambient', type=int, default=0,
                        help='whether to include ambient nodes in the meta-graph.')
    parser.add_argument('--sp_style', type=str, default='random',
                        help='Spasifying style. Only "random" is supported for now.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Direction noise level in the meta-graph adjacency matrix, less than 0.5.')
    parser.add_argument('--w', type=float, default=0.1,
                        help='Weight of change to the adjacency matrix.')
    parser.add_argument('--sparse_ratio', type=float, default=0.1,
                        help='Ratio to be zeroed of the adjacency matrix.')

    parser.add_argument('--pretrain_epochs', type=int, default=50,
                        help='Number of initial epochs before thresholding in loss.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Number of iterations to consider for early stopping.')
    parser.add_argument('--fill_val', type=float, default=0.5,
                        help='The value to be filled when we originally have 0, from meta-graph adj to meta-graph to generate data.')
    parser.add_argument('--regenerate_data', action='store_true', help='Whether to force creation of data splits.')
    parser.add_argument('--load_only', action='store_true', help='Whether not to store generated data.')
    parser.add_argument('-IsSparsed', '-S', action='store_true', help='Whether to sparsift adjacency matrix or downweigh its entries.')
    parser.add_argument('-AllTrain', '-All', action='store_true', help='Whether to use all data to do gradient descent.')
    parser.add_argument('-SavePred', '-SP', action='store_true', help='Whether to save predicted labels.')
    parser.add_argument('--log_root', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../logs/'), 
                        help='The path saving model.t7 and the training process')
    parser.add_argument('--data_path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../dataset/data/tmp/'), 
                        help='Data set folder.')
    parser.add_argument('--dataset', type=str, default='DSBM/', help='Data set selection.')
    parser.add_argument("--seeds",
                        nargs="+",
                        type=int,
                        help="seeds to generate random graphs.")
    parser.set_defaults(seeds=[10, 20, 30, 40, 50])
    # below are for MagNet
    parser.add_argument('--epochs_m', type=int, default=3000, help='Training epochs for MagNet.')
    parser.add_argument('--q', type=float, default=0.25, help='q value for the phase matrix')

    parser.add_argument('--K_m', type=int, default=1, help='K for cheb series for magnet')
    parser.add_argument('--layer', type=int, default=2, help='How many layers of gcn in the model, only 1 or 2 layers.')
    parser.add_argument('--dropout_m', type=float, default=0.5, help='Dropout prob for magnet')

    parser.add_argument('--lr_m', type=float, default=5e-3, help='Learning rate for magnet')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')

    parser.add_argument('--early_stopping_m', type=int, default=500, help='Number of MagNet iterations to consider for early stopping.')
    
    parser.add_argument('-activation', '-a', action='store_true', help='Whether to use activation function in MagNet.')
    parser.add_argument('--num_filter', type=int, default=16, help='Num of MagNet filters.')
    parser.add_argument('--to_radians', type=str, default='none', help='Whether to transform real and imaginary numbers to modulus and radians')
    parser.add_argument('-not_norm', '-n', action='store_false', help='Whether to use normalized laplacian or not, default: yes')
    # below are for DiGCN
    parser.add_argument('--epochs_d', type=int, default=1000)
    parser.add_argument('--lr_d', type=float, default=0.05)
    parser.add_argument('--weight_decay_d', type=float, default=0.0001)
    parser.add_argument('--early_stopping_d', type=int, default=200)
    parser.add_argument('--hidden_d', type=int, default=64)
    parser.add_argument('--hidden_ib', type=int, default=32)
    parser.add_argument('--dropout_d', type=float, default=0.5)
    parser.add_argument('--alpha_d', type=float, default=0.1,
                        help='alpha used in approximate personalized page rank.') 
    
    # lead-lag data
    parser.add_argument('--year_index', type=int, default=2,
                        help='Index of the year when using lead-lag data.') 
    args = parser.parse_args()

    if args.dataset[-1]!='/':
        args.dataset += '/'
    
    if args.dataset[:4] != 'DSBM':
        args.SavePred = True
        args.AllTrain = True
        args.train_ratio = 1
        args.test_ratio = 1
        args.num_trials = args.num_trials * len(args.seeds)
        args.seeds = [10]
        
    
    if args.dataset[:4] == 'DSBM':
        # calculate the meta-graph adjacency matrix F and the one to generate data: F_data
        args.F = meta_graph_generation(args.F_style, args.K, args.eta, args.ambient, 0)
        args.F_data = meta_graph_generation(args.F_style, args.K, args.eta, args.ambient, args.fill_val)
        default_name_base = args.F_style+ '_' + args.sp_style
        default_name_base += 'p' + str(int(100*args.p)) + 'K' + str(args.K) + 'N' + str(args.N) + 'size_r' + str(int(100*args.size_ratio))
        default_name_base += 'eta' + str(int(100*args.eta)) + 'ambient' + str(args.ambient)
        args.dataset = 'DSBM/' + default_name_base
    elif args.dataset[:4] == 'blog':
        args.F = 1
        args.K = 2
    elif args.dataset[:4] == 'tele':
        args.F = 5
        args.K = 4
    elif args.dataset[:4] == 'migr':
        args.F = 9
        args.K = 10
    elif args.dataset[:4] == 'wiki':
        args.F = 10
        args.K = 10
    elif args.dataset[:8].lower() == 'lead_lag':
        args.dataset = 'lead_lag/'+str(2001 + args.year_index)+'/'
        args.K = 10
        args.F = 3
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    if args.debug:
        args.seeds = [10]
        args.num_trials = 2
        args.epochs_m = 2
        args.epochs_e = 2
        args.pretrain_epochs = 1
        args.epochs = 2
        args.log_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../debug_logs/')
    return args
