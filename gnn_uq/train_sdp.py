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
              verbose = True, **kwargs):
    if verbose:
        pbar = tqdm(dataloader)
        iterable = pbar
    else:
        iterable = dataloader

    train_loss = []
    
    model.train()
    for i, batch in enumerate(iterable):

        batch = batch.to(device)
        optimizer.zero_grad()
        nodeInf, edgeInf = model(batch)

        model.predict(batch)
        
        nodeTarget = batch.y.clone().detach().float()
        nodeLoss = F.binary_cross_entropy(nodeInf[:, 0],
                                          nodeTarget)
        
        edgeTarget = batch.edge_label.clone().detach().float()
        edgeLoss = F.binary_cross_entropy(edgeInf[:,0],
                                          edgeTarget)

        loss = nodeLoss + edgeLoss
        train_loss.append(loss.item())
        if verbose:
            pbarMessage = " ".join(["train loss:",
                                    str(round(loss.item(), 4))])
            pbar.set_description(pbarMessage)
        
        loss.backward()
        optimizer.step()

    return train_loss

def testLoop(model, dataloader,
             verbose = True, **kwargs):
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
        batch = batch.to(device)
        nodeInf, edgeInf = model(batch)
        
        nodeTarget = batch.y.clone().detach().float()
        nodeLoss = F.binary_cross_entropy(nodeInf[:, 0],
                                          nodeTarget)

        decision_boundary = 0.5
        node_decision = nodeInf > decision_boundary
        TP = node_decision[:,0] == nodeTarget
        node_acc = torch.sum(TP)/len(node_decision)
        
        edgeTarget = batch_mean.edge_label.clone().detach().float()
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
    # batch_size = 64
    batch_size = 1
    train_data = ShowerFeaturesPared(file_path = args.train,
                                     mode = args.mode,
                                     noise_interval = (args.noise_lower_bound,
                                                       args.noise_upper_bound),
    )
    # train_data = ShowerFeaturesFromVoxels(file_path = args.train,
    #                                       mode = args.mode,
    #                                       # dropout_eff = 1,
    #                                       # dropout_eff = 0.8,
    #                                       dropout_eff = 0.75,
    # )
    train_dataloader = GraphDataLoader(train_data,
                                       shuffle     = True,
                                       num_workers = num_workers,
                                       batch_size  = batch_size
    )

    test_data = ShowerFeaturesPared(file_path = args.test,
                                    mode = args.mode,
                                    noise_interval = (args.noise_lower_bound,
                                                      args.noise_upper_bound),
    ) 
    # test_data = ShowerFeaturesFromVoxels(file_path = args.test,
    #                                      mode = args.mode,
    #                                      # dropout_eff = 1,
    #                                      # dropout_eff = 0.8,
    #                                      dropout_eff = 0.75,
    # ) 
    test_dataloader = GraphDataLoader(test_data,
                                      shuffle     = True,
                                      num_workers = num_workers,
                                      batch_size  = batch_size
    )

    # model = EdgeConvNet().to(device)
    # model = EdgeConvNet(input_node_feats = 32,
    #                     input_edge_feats = 38).to(device)
    # model = EdgeConvNet(input_node_feats = 2*10,
    #                     input_edge_feats = 2*6).to(device)
    # model = EdgeConv_SDP(input_node_feats = 10,
    #                      input_edge_feats = 6).to(device)
    model = EdgeConv_SDP(input_node_feats = 10,
                         input_edge_feats = 6).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-5, weight_decay=5e-4)

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
                                     args.verbose)

        if n_epoch%args.checkpoint_period == 0:
            checkpoint_path = os.path.join(args.output, 'checkpoint_epoch_'+str(n_epoch)+'.ckpt')
            torch.save(dict(model = model.state_dict()), checkpoint_path)

        print ("testing...")
        test_metrics = testLoop(model,
                                test_dataloader,
                                args.verbose)

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

    parser.add_argument('-m', '--mode', type = str,
                        default = 'UA',
                        help = "mode for noise input: {UA, blind, nonoise}")
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
                        default = '.',
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
    
    args = parser.parse_args()

    main(args)

