import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv

class EdgeUpdate(torch.nn.Module):
    def __init__(self, nn, **kwargs):
        super(EdgeUpdate, self).__init__(**kwargs)
        self.nn = nn

    def forward(self, node_feats, edge_feats, edge_indices):
        new_edge_feats = self.nn(torch.cat([edge_feats,
                                            node_feats[edge_indices[0]],
                                            node_feats[edge_indices[1]]],
                                           dim = 1))

        return new_edge_feats

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return torch.sigmoid(x)

class EdgeConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        nodeFeatSize = (16, 64, 128, 256, 1)
        edgeFeatSize = (19, 64, 128, 256, 1)

        self.nodeMessageMap1 = nn.Sequential(nn.BatchNorm1d(2*nodeFeatSize[0]),
                                             nn.Linear(2*nodeFeatSize[0], 
                                                       nodeFeatSize[1]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(nodeFeatSize[1]),
                                             nn.Linear(nodeFeatSize[1], 
                                                       nodeFeatSize[1]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(nodeFeatSize[1]),
                                             nn.Linear(nodeFeatSize[1], 
                                                       nodeFeatSize[1]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(nodeFeatSize[1]),
        )
        self.nodeMP1 = EdgeConv(self.nodeMessageMap1)

        self.edgeMessageMap1 = nn.Sequential(nn.BatchNorm1d(2*nodeFeatSize[1] + edgeFeatSize[0]),
                                             nn.Linear(2*nodeFeatSize[1] + edgeFeatSize[0], 
                                                       edgeFeatSize[1]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(edgeFeatSize[1]),
                                             nn.Linear(edgeFeatSize[1], 
                                                       edgeFeatSize[1]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(edgeFeatSize[1]),
                                             nn.Linear(edgeFeatSize[1], 
                                                       edgeFeatSize[1]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(edgeFeatSize[1]),
        )
        self.edgeMP1 = EdgeUpdate(self.edgeMessageMap1)

        self.nodeMessageMap2 = nn.Sequential(nn.Linear(2*nodeFeatSize[1], 
                                                       nodeFeatSize[2]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(nodeFeatSize[2]),
                                             nn.Linear(nodeFeatSize[2], 
                                                       nodeFeatSize[2]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(nodeFeatSize[2]),
                                             nn.Linear(nodeFeatSize[2], 
                                                       nodeFeatSize[2]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(nodeFeatSize[2]),
        )
        self.nodeMP2 = EdgeConv(self.nodeMessageMap2)

        self.edgeMessageMap2 = nn.Sequential(nn.BatchNorm1d(2*nodeFeatSize[2] + edgeFeatSize[1]),
                                             nn.Linear(2*nodeFeatSize[2] + edgeFeatSize[1], 
                                                       edgeFeatSize[2]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(edgeFeatSize[2]),
                                             nn.Linear(edgeFeatSize[2], 
                                                       edgeFeatSize[2]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(edgeFeatSize[2]),
                                             nn.Linear(edgeFeatSize[2], 
                                                       edgeFeatSize[2]),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(edgeFeatSize[2]),
        )
        self.edgeMP2 = EdgeUpdate(self.edgeMessageMap2)

        self.nodeMLPhead = nn.Sequential(nn.Linear(nodeFeatSize[2], 
                                                   nodeFeatSize[3]),
                                         nn.ReLU(),
                                         nn.Linear(nodeFeatSize[3], 
                                                   nodeFeatSize[3]),
                                         nn.ReLU(),
                                         nn.Linear(nodeFeatSize[3], 
                                                   nodeFeatSize[3]),
                                         nn.Linear(nodeFeatSize[3], 
                                                   nodeFeatSize[4]),
        )

        self.edgeMLPhead = nn.Sequential(nn.Linear(edgeFeatSize[2], 
                                                   edgeFeatSize[3]),
                                         nn.Linear(edgeFeatSize[3], 
                                                   edgeFeatSize[3]),
                                         nn.ReLU(),
                                         nn.Linear(edgeFeatSize[3], 
                                                   edgeFeatSize[3]),
                                         nn.ReLU(),
                                         nn.Linear(edgeFeatSize[3], 
                                                   edgeFeatSize[3]),
                                         nn.Linear(edgeFeatSize[3], 
                                                   edgeFeatSize[4]),
        )
        
    def forward(self, data):
        node_feats, edge_feats, edge_index = data.x, data.edge_attr, data.edge_index

        node_feats = self.nodeMP1(node_feats, edge_index)
        edge_feats = self.edgeMP1(node_feats, edge_feats, edge_index)
        node_feats = self.nodeMP2(node_feats, edge_index)
        edge_feats = self.edgeMP2(node_feats, edge_feats, edge_index)

        node_feats = self.nodeMLPhead(node_feats)
        edge_feats = self.edgeMLPhead(edge_feats)

        return torch.sigmoid(node_feats), torch.sigmoid(edge_feats)
