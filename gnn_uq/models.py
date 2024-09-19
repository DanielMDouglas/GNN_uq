import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    def __init__(self, input_node_feats = 16, input_edge_feats = 19, final_sigmoid = True, aggr = 'max'):
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

        self.aggr = aggr

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
        self.nodeMP1 = EdgeConv(self.nodeMessageMap1, aggr = self.aggr)
        # self.nodeMP1 = self.nodeMessageMap1

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
        self.nodeMP2 = EdgeConv(self.nodeMessageMap2, aggr = self.aggr)

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

    def forward_from_tensors(self, node_feats, edge_feats, edge_index):

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
                                     final_sigmoid = False,
                                     aggr = 'add')

    def forward(self, inpt, final_sigmoid = True):
        mean = GraphData(x = inpt.x[:,:self.input_node_feats],
                         edge_index = inpt.edge_index,
                         edge_attr = inpt.edge_attr[:,:self.input_edge_feats],
                         y = inpt.y,
                         edge_label = inpt.edge_label,
                         index = inpt.index)
        # node_unc = torch.exp(inpt.x[:,self.input_node_feats:])
        # edge_unc = torch.exp(inpt.edge_attr[:,self.input_edge_feats:])
        
        node_inf, edge_inf =  self.inner_GNN(mean)

        return torch.sigmoid(node_inf), torch.sigmoid(edge_inf)

    def forward_from_tensors(self, node_feats, edge_feats, edge_index):
        node_mean = node_feats[:,:self.input_node_feats]
        edge_mean = edge_feats[:,:self.input_edge_feats]
        
        node_inf, edge_inf =  self.inner_GNN.forward_from_tensors(node_mean, edge_mean, edge_index)

        return torch.sigmoid(node_inf), torch.sigmoid(edge_inf)

    def forward_from_node_tensor(self, edge_feats, edge_index):

        def single_parameter_function(node_feats):
            return self.forward_from_tensors(node_feats, edge_feats, edge_index)

        return single_parameter_function

    def forward_from_edge_tensor(self, node_feats, edge_index):

        def single_parameter_function(edge_feats):
            return self.forward_from_tensors(node_feats, edge_feats, edge_index)

        return single_parameter_function

    def predict (self, inpt):
        print ("jacobian calculating...")
        # J = torch.vmap(torch.func.jacrev(self.forward_from_node_tensor(inpt.edge_attr, inpt.index)))(inpt.x)
        node_mean = inpt.x[:,:self.input_node_feats]
        edge_mean = inpt.edge_attr[:,:self.input_edge_feats]
        
        node_unc = torch.flatten(torch.exp(inpt.x[:,self.input_node_feats:]))
        edge_unc = torch.flatten(torch.exp(inpt.edge_attr[:,self.input_edge_feats:]))

        node_inpt_cov = torch.diag(node_unc*node_unc)
        edge_inpt_cov = torch.diag(edge_unc*edge_unc)

        self.eval()
        xp = inpt.x.clone().requires_grad_()
        ep = inpt.edge_attr.clone().requires_grad_()

        mean_node_pred, mean_edge_pred = self.forward(inpt, final_sigmoid = False)

        # calculate jacobian w.r.t. node inputs
        J_node_all = torch.func.jacrev(self.forward_from_node_tensor(inpt.edge_attr, inpt.edge_index))(xp)
        J_node_node = torch.flatten(J_node_all[0][:,0,:,:self.input_node_feats],
                                    start_dim = 1,
                                    end_dim = 2)
        J_node_edge = torch.flatten(J_node_all[1][:,0,:,:self.input_node_feats],
                                    start_dim = 1,
                                    end_dim = 2)

        JT_node_node = J_node_node.T
        JT_node_edge = J_node_edge.T

        cov_node_edge = torch.matmul(torch.matmul(J_node_edge, node_inpt_cov), JT_node_edge)
        cov_node_node = torch.matmul(torch.matmul(J_node_node, node_inpt_cov), JT_node_node)

        # calculate jacobian w.r.t. edge inputs
        J_edge_all = torch.func.jacrev(self.forward_from_edge_tensor(inpt.x, inpt.edge_index))(ep)
        J_edge_node = torch.flatten(J_edge_all[0][:,0,:,:self.input_edge_feats],
                                    start_dim = 1,
                                    end_dim = 2)
        J_edge_edge = torch.flatten(J_edge_all[1][:,0,:,:self.input_edge_feats],
                                    start_dim = 1,
                                    end_dim = 2)

        JT_edge_node = J_edge_node.T
        JT_edge_edge = J_edge_edge.T

        cov_edge_edge = torch.matmul(torch.matmul(J_edge_edge, edge_inpt_cov), JT_edge_edge)
        cov_edge_node = torch.matmul(torch.matmul(J_edge_node, edge_inpt_cov), JT_edge_node)

        std_node_pred = torch.sqrt(torch.diag(cov_node_node + cov_edge_node))
        std_edge_pred = torch.sqrt(torch.diag(cov_node_edge + cov_edge_edge))

        node_score = 0.5*(1 + torch.erf(mean_node_pred.flatten()/(std_node_pred*np.sqrt(2))))
        edge_score = 0.5*(1 + torch.erf(mean_edge_pred.flatten()/(std_edge_pred*np.sqrt(2))))

        return node_score[:,None], edge_score[:,None]
