import torch
import torch.nn as nn

import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torch_geometric.loader import DataLoader as GraphDataLoader
from utils import ShowerFeatures

from torch_geometric.nn import GCNConv, EdgeConv
from models import *

def main(args):
    data = ShowerFeatures(file_path = args.test,
                          mode = args.mode,
                          noise_interval = (0.1, 0.3))
    dataloader = GraphDataLoader(data,
                                 shuffle     = True,
                                 num_workers = 0,
                                 batch_size  = 64
    )

    model = EdgeConvNet(input_node_feats = 32,
                        input_edge_feats = 38).to(device)

    checkpoint = torch.load(args.checkpoint,
                            map_location = device)
    model.load_state_dict(checkpoint['model'], strict = False)

    pbar = tqdm(dataloader)
    model.eval()

    node_scores = torch.Tensor([]).to(device)
    edge_scores = torch.Tensor([]).to(device)

    node_label = torch.Tensor([]).to(device)
    edge_label = torch.Tensor([]).to(device)

    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        nodeInf, edgeInf = model(batch)
        
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

        node_scores = torch.cat((node_scores, nodeInf))
        edge_scores = torch.cat((edge_scores, edgeInf))

        node_label = torch.cat((node_label, nodeTarget))
        edge_label = torch.cat((edge_label, edgeTarget))

        if node_scores.shape[0] > 40000:
            break

        # print (node_scores)
        # print (node_scores.shape)

        
    print (node_scores, edge_scores)
    print (node_label, edge_label)

    nodeLoss = F.binary_cross_entropy(nodeInf[:,0],
                                      nodeTarget)
    edgeLoss = F.binary_cross_entropy(edgeInf[:,0],
                                      edgeTarget)

    print ("node loss", nodeLoss)
    print ("edge loss", edgeLoss)
    
    node_scores = node_scores.cpu().detach().numpy()
    edge_scores = edge_scores.cpu().detach().numpy()

    node_label = node_label.cpu().detach().numpy()
    edge_label = edge_label.cpu().detach().numpy()

    print (np.min(node_scores), np.max(node_scores))
    print (np.min(edge_scores), np.max(edge_scores))

    score_bins = np.linspace(0, 1, 51)
    fig = plt.figure()
    plt.hist(node_scores[node_label == 0],
             bins = score_bins,
             histtype = 'step',
             label = 'label == 0',
             )
    plt.hist(node_scores[node_label == 1],
             bins = score_bins,
             histtype = 'step',
             label = 'label == 1',
             )
    plt.semilogy()
    plt.legend()
    plt.xlabel('Node score')
    plt.savefig(os.path.join(args.output,
                             'node_score.png'))

    fig = plt.figure()
    plt.hist(edge_scores[edge_label == 0],
             bins = score_bins,
             histtype = 'step',
             label = 'label == 0'
             )
    plt.hist(edge_scores[edge_label == 1],
             bins = score_bins,
             histtype = 'step',
             label = 'label == 1'
             )
    plt.legend()
    plt.semilogy()
    plt.xlabel('Edge score')
    plt.savefig(os.path.join(args.output,
                             'edge_score.png'))

    weighted_counts, bins = np.histogram(edge_scores.flatten(), bins = score_bins,
                                         weights = edge_label.flatten())
    unweighted_counts, bins = np.histogram(edge_scores.flatten(), bins = score_bins)
    fig = plt.figure()
    plt.plot(0.5*(score_bins[1:] + score_bins[:-1]),
             weighted_counts/unweighted_counts,
             )
    plt.plot(np.linspace(0, 1, 1000),
             np.linspace(0, 1, 1000),
             ls = '--',
             color = 'red',
             )
    plt.xlabel(r'Edge Score')
    plt.ylabel(r'Fraction of True Edges')
    plt.savefig(os.path.join(args.output,
                             'edge_calib.png'))

    weighted_counts, bins = np.histogram(node_scores.flatten(), bins = score_bins,
                                         weights = node_label.flatten())
    unweighted_counts, bins = np.histogram(node_scores.flatten(), bins = score_bins)
    fig = plt.figure()
    plt.plot(0.5*(score_bins[1:] + score_bins[:-1]),
             weighted_counts/unweighted_counts,
             )
    plt.plot(np.linspace(0, 1, 1000),
             np.linspace(0, 1, 1000),
             ls = '--',
             color = 'red',
             )
    plt.xlabel(r'Node Score')
    plt.ylabel(r'Fraction of True Nodes')
    plt.savefig(os.path.join(args.output,
                             'node_calib.png'))

    
    return

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type = str,
                        help = "model weight file")
    parser.add_argument('-m', '--mode', type = str,
                        default = 'UA',
                        help = "mode for noise input: {UA, blind, nonoise}")
    parser.add_argument('--test', type = str,
                        default = '/sdf/home/d/dougl215/studies/GNN_uq/data/if-graph-test.h5',
                        help = "input test data (hdf5)")
    parser.add_argument('-o', '--output', type = str,
                        default = '.',
                        help = "output prefix directory")

    args = parser.parse_args()

    main(args)

