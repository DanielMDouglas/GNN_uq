import torch
import torch.nn as nn

import numpy as np
import os
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torch_geometric.loader import DataLoader as GraphDataLoader
from utils import ShowerFeatures

from torch_geometric.nn import GCNConv, EdgeConv
from models import *

def trainLoop(model, dataloader, optimizer, **kwargs):
    pbar = tqdm(dataloader)
    iterable = pbar

    model.train()
    for i, batch in enumerate(iterable):
        batch = batch.to(device)
        optimizer.zero_grad()
        nodeInf, edgeInf = model(batch)
        
        nodeTarget = batch.y.clone().detach().float()
        nodeLoss = F.binary_cross_entropy(nodeInf[:, 0],
                                          nodeTarget)
        
        edgeTarget = batch.edge_label.clone().detach().float()
        edgeLoss = F.binary_cross_entropy(edgeInf[:,0],
                                          edgeTarget)
        
        loss = nodeLoss + edgeLoss

        verbose = True
        if verbose:
            pbarMessage = " ".join(["train loss:",
                                    str(round(loss.item(), 4))])
            pbar.set_description(pbarMessage)
        
        loss.backward()
        optimizer.step()

def testLoop(model, dataloader, **kwargs):
    pbar = tqdm(dataloader)
    iterable = pbar

    test_loss = []
    
    model.eval()
    for i, batch in enumerate(iterable):
        batch = batch.to(device)
        nodeInf, edgeInf = model(batch)
        
        nodeTarget = batch.y.clone().detach().float()
        nodeLoss = F.binary_cross_entropy(nodeInf[:, 0],
                                          nodeTarget)
        
        edgeTarget = batch.edge_label.clone().detach().float()
        edgeLoss = F.binary_cross_entropy(edgeInf[:,0],
                                          edgeTarget)

        # print (nodeInf)
        # print (nodeTarget)
        
        loss = nodeLoss + edgeLoss

        # test_loss.append(loss)
        
        verbose = True
        if verbose:
            pbarMessage = " ".join(["test loss:",
                                    str(round(loss.item(), 4))])
            pbar.set_description(pbarMessage)

    # print ("test loss:", torch.mean(test_loss), torch.std(test_loss))
        
def main(args):
    train_data = ShowerFeatures(file_path = args.train)
    train_dataloader = GraphDataLoader(train_data,
                                       shuffle     = True,
                                       num_workers = 0,
                                       batch_size  = 64
    )

    test_data = ShowerFeatures(file_path = args.test)
    test_dataloader = GraphDataLoader(test_data,
                                      shuffle     = True,
                                      num_workers = 0,
                                      batch_size  = 64
    )

    model = EdgeConvNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4, weight_decay=5e-4)

    for n_epoch in range(args.nEpochs):
        trainLoop(model, train_dataloader, optimizer)
        testLoop(model, test_dataloader)
        
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
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    
    args = parser.parse_args()

    main(args)

