import torch
import torch.nn as nn

import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torch_geometric.loader import DataLoader as GraphDataLoader
from utils import ShowerFeatures, ShowerFeaturesPared
# from voxelBasedNoise import ShowerFeaturesFromVoxels

from torch_geometric.nn import GCNConv, EdgeConv
from models import *

def trainLoop(model, dataloader, optimizer,
              epoch,
              verbose = True,
              enable_sdp = False,
              **kwargs):
    if verbose:
        pbar = tqdm(dataloader)
        iterable = pbar
    else:
        iterable = dataloader

    train_loss = []
    
    model.train()
    for i, batch in enumerate(iterable):
        batch.to(device)
        
        if enable_sdp and batch.x.shape[0] > 20:
            continue

        # # burn in without jac propagation first
        # if epoch < 1:
        #     nodeInf, edgeInf = model(batch.to(device))
        # else:
        # nodeInf, edgeInf = model.predict_autograd(batch.to(device))

        nodeInf, edgeInf = model.forward_from_tensors(batch.x[:,:10],
                                                      batch.edge_attr[:,:6],
                                                      batch.edge_index,
        )

        model.eval()
        jac = torch.func.jacrev(model.forward_from_tensors,
                                argnums = (0, 1))(batch.x[:,:10],
                                                  batch.edge_attr[:,:6],
                                                  batch.edge_index)
        (J_node_node, J_node_edge), (J_edge_node, J_edge_edge) = jac
        model.train()
        
        J_node_node = J_node_node[:,0,:,:].flatten(start_dim = 1, end_dim = 2)
        J_node_edge = J_node_edge[:,0,:,:].flatten(start_dim = 1, end_dim = 2)
        J_edge_node = J_edge_node[:,0,:,:].flatten(start_dim = 1, end_dim = 2)
        J_edge_edge = J_edge_edge[:,0,:,:].flatten(start_dim = 1, end_dim = 2)

        # sdp_scale = args.alpha
        sdp_scale = 1
        epsilon = 1.e-3
        
        node_var = sdp_scale*torch.pow(torch.flatten(torch.exp(batch.x[:,10:])), 2) + epsilon
        edge_var = sdp_scale*torch.pow(torch.flatten(torch.exp(batch.edge_attr[:,6:])), 2) + epsilon

        cov_node_node = torch.diag(torch.matmul(J_node_node*node_var, J_node_node.T))
        cov_node_edge = torch.diag(torch.matmul(J_node_edge*edge_var, J_node_edge.T))
        cov_edge_node = torch.diag(torch.matmul(J_edge_node*node_var, J_edge_node.T))
        cov_edge_edge = torch.diag(torch.matmul(J_edge_edge*edge_var, J_edge_edge.T))

        std_node_pred = torch.sqrt(cov_node_node + cov_node_edge)
        std_edge_pred = torch.sqrt(cov_edge_node + cov_edge_edge)

        node_score = approx_cdf(nodeInf.flatten()/std_node_pred)
        edge_score = approx_cdf(edgeInf.flatten()/std_edge_pred)

        print (np.min(node_score), np.max(node_score))
        print (np.min(edge_score), np.max(edge_score))

        nodeTarget = batch.y.clone().detach().float()
        nodeLoss = F.binary_cross_entropy(node_score,
                                          nodeTarget)
        
        edgeTarget = batch.edge_label.clone().detach().float()
        edgeLoss = F.binary_cross_entropy(edge_score,
                                          edgeTarget)

        loss = nodeLoss + edgeLoss
        train_loss.append(loss.item())
        if verbose:
            pbarMessage = " ".join(["train loss:",
                                    str(round(loss.item(), 4))])
            pbar.set_description(pbarMessage)
        
        loss.backward()

        # if args.enable_sdp:
        #     # for i in model.parameters():
        #     #     print (i)
        #     clip = 1.e4
        #     clip_res = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        #     print ("total norm", clip_res)
        #     clip_res = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        #     print ("total norm", clip_res)

        optimizer.step()

    return train_loss

def testLoop(model, dataloader,
             epoch,
             verbose = True,
             enable_sdp = False,
             **kwargs):
    if verbose:
        pbar = tqdm(dataloader)
        iterable = pbar
    else:
        iterable = dataloader

    test_loss = []
    test_node_acc = []
    test_edge_acc = []
    
    model.eval()
    for i, batch in enumerate(iterable):
        if enable_sdp and batch.x.shape[0] > 30:
            # print ('skipping')
            continue
        # else:
        #     print ('not skipping')

        # if epoch < 1:
        #     nodeInf, edgeInf = model(batch)
        # else:
        # nodeInf, edgeInf = model.predict_autograd(batch)
        # nodeInf, edgeInf = model.predict_jacfwd(batch)
        if enable_sdp:
            nodeInf, edgeInf = model.predict_jacfwd(batch.to(device))
        else:
            nodeInf, edgeInf = model(batch.to(device))
             
        # print (edgeInf)
        nodeTarget = batch.y.clone().detach().float()
        nodeLoss = F.binary_cross_entropy(nodeInf[:, 0],
                                          nodeTarget)

        decision_boundary = 0.5
        node_decision = nodeInf > decision_boundary
        TP = node_decision[:,0] == nodeTarget
        node_acc = torch.sum(TP)/len(node_decision)
        
        edgeTarget = batch.edge_label.clone().detach().float()
        edgeLoss = F.binary_cross_entropy(edgeInf[:,0],
                                          edgeTarget)

        decision_boundary = 0.5
        edge_decision = edgeInf > decision_boundary
        TP = edge_decision[:,0] == edgeTarget

        edge_acc = torch.sum(TP)/len(edge_decision)
        
        loss = nodeLoss + edgeLoss
        # loss = edgeLoss

        test_loss.append(loss.item())
        test_node_acc.append(node_acc.item())
        test_edge_acc.append(edge_acc.item())
        
        if verbose:
            pbarMessage = " ".join(["test loss:",
                                    str(round(loss.item(), 4)),
                                    "node acc:",
                                    str(round(node_acc.item(), 4)),
                                    "edge acc:",
                                    str(round(edge_acc.item(), 4))])
            pbar.set_description(pbarMessage)

    # print ("test loss:", np.mean(test_loss), np.std(test_loss))
    return test_loss, test_node_acc, test_edge_acc
        

def main(args):
    num_workers = 0

    if args.enable_sdp:
        batch_size = 1
    else:
        batch_size = 64

    # batch_size = 16
    train_data = ShowerFeaturesPared(file_path = args.train,
                                     mode = 'UA',
                                     noise_interval = (args.noise_lower_bound,
                                                       args.noise_upper_bound),
    )
    train_dataloader = GraphDataLoader(train_data,
                                       shuffle     = True,
                                       num_workers = num_workers,
                                       batch_size  = batch_size
    )

    test_data = ShowerFeaturesPared(file_path = args.test,
                                    mode = 'UA',
                                    noise_interval = (args.noise_lower_bound,
                                                      args.noise_upper_bound),
    ) 
    test_dataloader = GraphDataLoader(test_data,
                                      shuffle     = True,
                                      num_workers = num_workers,
                                      batch_size  = batch_size
    )

    model = EdgeConvNet(input_node_feats = 10,
                        input_edge_feats = 6,
                        aggr = 'add').to(device)
    # model = EdgeConv_SDP(input_node_feats = 10,
    #                      input_edge_feats = 6,
    #                      sdp_scale = 10,
    #                      epsilon = 1.e-2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-6, weight_decay=5e-4)

    if args.checkpoint:
        with open(args.checkpoint, 'rb') as f:
            checkpoint = torch.load(f,
                                    map_location = device)
            model.load_state_dict(checkpoint['model'], strict=False)
    
    train_loss = []
    test_mean_loss = []
    test_std_loss = []
    test_mean_node_acc = []
    test_std_node_acc = []
    test_mean_edge_acc = []
    test_std_edge_acc = []

    for n_epoch in range(args.nEpochs):
        print ("epoch " + str(n_epoch))
        print ("training...")
        epoch_train_loss = trainLoop(model,
                                     train_dataloader,
                                     optimizer,
                                     n_epoch,
                                     args.verbose,
                                     args.enable_sdp)

        if n_epoch%args.checkpoint_period == 0:
            checkpoint_path = os.path.join(args.output, 'checkpoint_epoch_'+str(n_epoch)+'.ckpt')
            torch.save(dict(model = model.state_dict()), checkpoint_path)

        print ("testing...")
        test_metrics = testLoop(model,
                                test_dataloader,
                                n_epoch,
                                args.verbose,
                                args.enable_sdp)

        epoch_test_loss, epoch_test_node_accuracy, epoch_test_edge_accuracy = test_metrics

        train_loss = train_loss + epoch_train_loss

        test_mean_loss.append(np.mean(epoch_train_loss))
        test_std_loss.append(np.std(epoch_train_loss))

        test_mean_node_acc.append(np.mean(epoch_test_node_accuracy))
        test_std_node_acc.append(np.std(epoch_test_node_accuracy))

        test_mean_edge_acc.append(np.mean(epoch_test_edge_accuracy))
        test_std_edge_acc.append(np.std(epoch_test_edge_accuracy))

    plt.figure()
    plt.plot(np.linspace(0, args.nEpochs, len(train_loss)),
             train_loss,
             label = 'Train Loss')
    plt.errorbar(np.arange(len(test_mean_loss)) + 1,
                 test_mean_loss,
                 yerr = test_std_loss,
                 fmt = 'o',
                 label = 'Test Loss')
    plt.legend()
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Edge Loss')

    plt.savefig(os.path.join(args.output,
                             'train_curve.png'))

    plt.figure()
    plt.errorbar(np.arange(len(test_mean_node_acc)) + 1,
                 test_mean_node_acc,
                 yerr = test_std_node_acc,
                 fmt = 'o',
                 # label = 'Test Loss',
    )
    plt.legend()
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Node Classification Accuracy (TP/TP + FP)')

    plt.savefig(os.path.join(args.output,
                             'test_node_accuracy.png'))

    plt.figure()
    plt.errorbar(np.arange(len(test_mean_edge_acc)) + 1,
                 test_mean_edge_acc,
                 yerr = test_std_edge_acc,
                 fmt = 'o',
                 # label = 'Test Loss',
    )
    plt.legend()
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Edge Classification Accuracy (TP/TP + FP)')

    plt.savefig(os.path.join(args.output,
                             'test_edge_accuracy.png'))
    
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type = str,
                        default = '/sdf/home/d/dougl215/studies/GNN_uq/data/if-graph-train.h5',
                        help = "input train data (hdf5)")
    parser.add_argument('--test', type = str,
                        default = '/sdf/home/d/dougl215/studies/GNN_uq/data/if-graph-test.h5',
                        help = "input test data (hdf5)")

    parser.add_argument('-n', '--nEpochs', type = int,
                        default = 5,
                        help = "maximum number of epochs to train")
    parser.add_argument('-lb', '--noise_lower_bound', type = float,
                        default = 0.2,
                        help = "lower bound (fractional) noise per input feature")
    parser.add_argument('-ub', '--noise_upper_bound', type = float,
                        default = 0.5,
                        help = "upper bound (fractional) noise per input feature")
    parser.add_argument('-c', '--checkpoint', type = str,
                        default = '',
                        help = "checkpoint to load")
    parser.add_argument('-f', '--checkpoint_period', type = int,
                        default = 1,
                        help = "save a checkpoint every N epochs")
    parser.add_argument('-o', '--output', type = str,
                        default = '.',
                        help = "output prefix directory")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    parser.add_argument('-s', '--enable_sdp',
                        action = 'store_true',
                        help = "providing this flag enables SDP propagation of input error instead of directly predicting a Bernoulli score")
    
    args = parser.parse_args()

    main(args)

