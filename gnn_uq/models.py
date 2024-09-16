import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv
from torch_geometric.data import Data as GraphData

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
    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = GCNConv(n_feat, 2*n_feat)
        self.conv2 = GCNConv(2*n_feat, 4*n_feat)
        self.conv3 = GCNConv(4*n_feat, 1)

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
    def __init__(self, input_node_feats = 16, input_edge_feats = 19, final_sigmoid = True):
        super().__init__()

        self.final_sigmoid = final_sigmoid
        # nodeFeatSize = (16, 64, 128, 256, 1)
        # edgeFeatSize = (19, 64, 128, 256, 1)
        nodeFeatSize = (input_node_feats,
                        4*input_node_feats,
                        8*input_node_feats,
                        16*input_node_feats,
                        1)
        edgeFeatSize = (input_edge_feats,
                        4*input_edge_feats,
                        8*input_edge_feats,
                        16*input_edge_feats,
                        1)

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

        if self.final_sigmoid:
            return torch.sigmoid(node_feats), torch.sigmoid(edge_feats)
        else:
            return node_feats, edge_feats

class EdgeConv_SDP(torch.nn.Module):
    def __init__(self, input_node_feats = 16, input_edge_feats = 19):
        super().__init__()
        self.input_node_feats = input_node_feats
        self.input_edge_feats = input_edge_feats
        self.inner_GNN = EdgeConvNet(input_node_feats,
                                     input_edge_feats,
                                     final_sigmoid = False)

    def forward(self, inpt):
        # print (inpt.x.shape)
        # print (inpt.edge_attr.shape)
        # print (inpt.edge_index.shape)
        mean = GraphData(x = inpt.x[:,:self.input_node_feats],
                         edge_index = inpt.edge_index,
                         edge_attr = inpt.edge_attr[:,:self.input_edge_feats],
                         y = inpt.y,
                         edge_label = inpt.edge_label,
                         index = inpt.index)
        node_unc = torch.exp(inpt.x[:,self.input_node_feats:])
        edge_unc = torch.exp(inpt.edge_attr[:,self.input_edge_feats:])
        print (node_unc)

        node_inf, edge_inf =  self.inner_GNN(mean)

        # print (node_inf)
        # print (node_inf[0])

        node_inf_unc = torch.empty_like(node_inf)
        edge_inf_unc = torch.empty_like(edge_inf)
        
        # node_cov = torch.diag(node_unc)
        # edge_cov = torch.diag(edge_unc)

        # print ("node_cov", node_cov)

        
        
        for i, this_node_inf in enumerate(node_inf):
            node_node_jac = torch.autograd.grad(this_node_inf, mean.x,
                                                create_graph=True,
                                                retain_graph=True,
                                                allow_unused=True,
            )[0]
            
            this_node_inf_unc = torch.sqrt(torch.sum(torch.pow(node_node_jac, 2)*torch.pow(node_unc, 2)))
            node_inf_unc[i] = this_node_inf_unc

        for i, this_edge_inf in enumerate(edge_inf):
            node_edge_jac = torch.autograd.grad(this_edge_inf, mean.x,
                                                create_graph=True,
                                                retain_graph=True,
                                                allow_unused=True,
            )[0]

            this_edge_inf_unc = torch.sum(torch.pow(node_edge_jac, 2)*torch.pow(node_unc, 2))

            edge_edge_jac = torch.autograd.grad(this_edge_inf, mean.edge_attr,
                                                create_graph=True,
                                                retain_graph=True,
                                                allow_unused=True,
            )[0]

            this_edge_inf_unc += torch.sum(torch.pow(edge_edge_jac, 2)*torch.pow(edge_unc, 2))
            this_edge_inf_unc = torch.sqrt(this_edge_inf_unc)
            
            edge_inf_unc[i] = this_edge_inf_unc

        # print ("inference")
        # print (node_inf, node_inf.shape,
        #        edge_inf, edge_inf.shape)

        # print ("uncertainty")
        # print (node_inf_unc, node_inf_unc.shape,
        #        edge_inf_unc, edge_inf_unc.shape)
            
        # print (node_inf.shape)
        # print (edge_inf.shape)
        
        return torch.sigmoid(node_inf), torch.sigmoid(edge_inf)
