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

train = '/sdf/home/d/dougl215/studies/GNN_uq/data/if-graph-train.h5'
test = '/sdf/home/d/dougl215/studies/GNN_uq/data/if-graph-test.h5'

data = ShowerFeatures(file_path = train)

dataloader = GraphDataLoader(data,
                             shuffle     = True,
                             num_workers = 0,
                             batch_size  = 64
)

model = EdgeConvNet().to(device)

pbar = tqdm(dataloader)
iterable = pbar

optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3, weight_decay=5e-4)
    
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
        pbarMessage = " ".join(["loss:",
                                str(round(loss.item(), 4))])
        pbar.set_description(pbarMessage)
        
    # lossHistory.append(loss.item())

    loss.backward()
    optimizer.step()

