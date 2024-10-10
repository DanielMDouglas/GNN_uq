import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, EdgeConv
from torch_geometric.data import Data as GraphData

from utils import approx_erf, approx_cdf

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

    def forward_from_data_tensors(self, edge_index):

        return lambda node_feats, edge_feats: self.forward_from_tensors(node_feats, edge_feats, edge_index)

    def forward_from_node_tensor(self, edge_feats, edge_index):

        return lambda node_feats: self.forward_from_tensors(node_feats, edge_feats, edge_index)

    def forward_from_edge_tensors(self, node_feats, edge_index):

        return lambda edge_feats: self.forward_from_tensors(node_feats, edge_feats, edge_index)
        
class EdgeConv_SDP(torch.nn.Module):
    def __init__(self, input_node_feats = 16, input_edge_feats = 19, sdp_scale = 1, epsilon = 1.e-3):
        super().__init__()
        self.input_node_feats = input_node_feats
        self.input_edge_feats = input_edge_feats
        self.inner_GNN = EdgeConvNet(input_node_feats,
                                     input_edge_feats,
                                     final_sigmoid = False,
                                     aggr = 'add')

        self.sdp_scale = sdp_scale
        self.epsilon = epsilon
        
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

        if final_sigmoid:
            return torch.sigmoid(node_inf), torch.sigmoid(edge_inf)
        else:
            return node_inf, edge_inf

    def forward_from_tensors(self, node_feats, edge_feats, edge_index):
        node_inf, edge_inf = self.inner_GNN.forward_from_tensors(node_feats, edge_feats, edge_index)

        return node_inf, edge_inf

    def forward_from_node_tensor(self, edge_feats, edge_index):

        def single_parameter_function(node_feats):
            return self.forward_from_tensors(node_feats, edge_feats, edge_index)

        return single_parameter_function

    def forward_from_edge_tensor(self, node_feats, edge_index):

        def single_parameter_function(edge_feats):
            return self.forward_from_tensors(node_feats, edge_feats, edge_index)

        return single_parameter_function

    def predict (self, inpt):
        # print ("jacobian calculating...")
        # J = torch.vmap(torch.func.jacrev(self.forward_from_node_tensor(inpt.edge_attr, inpt.index)))(inpt.x)
        # node_mean = inpt.x[:,:self.input_node_feats]
        # edge_mean = inpt.edge_attr[:,:self.input_edge_feats]
        
        # node_unc = torch.flatten(torch.exp(inpt.x[:,self.input_node_feats:]))
        # edge_unc = torch.flatten(torch.exp(inpt.edge_attr[:,self.input_edge_feats:]))

        # node_inpt_cov = torch.diag(node_unc*node_unc)
        # edge_inpt_cov = torch.diag(edge_unc*edge_unc)

        self.eval()

        # (J_node_node, J_node_edge), (J_edge_node, J_edge_edge) = torch.func.jacfwd(self.forward_from_tensors,
        #                                                                            argnums = (0, 1))(inpt.x[:,:self.input_node_feats],
        #                                                                                              inpt.edge_attr[:,:self.input_edge_feats],
        #                                                                                              inpt.edge_index)

        print (inpt.x.shape)

        # J_node_node = J_node_node[:,0,:,:].flatten(start_dim = 1, end_dim = 2)
        # J_node_edge = J_node_edge[:,0,:,:].flatten(start_dim = 1, end_dim = 2)
        # J_edge_node = J_edge_node[:,0,:,:].flatten(start_dim = 1, end_dim = 2)
        # J_edge_edge = J_edge_edge[:,0,:,:].flatten(start_dim = 1, end_dim = 2)

        # node_var = torch.pow(torch.flatten(torch.exp(inpt.x[:,self.input_node_feats:])), 2)
        # edge_var = torch.pow(torch.flatten(torch.exp(inpt.edge_attr[:,self.input_edge_feats:])), 2)

        pred_node_mean, pred_edge_mean = self.forward(inpt, final_sigmoid = False)
        
        # print (J_node_node.shape)
        print (pred_node_mean)
        print (inpt.y)
        
        # print ("nodes", inpt.x.shape)
        # print ("edges", inpt.edge_attr.shape)
        # xp = inpt.x.clone().requires_grad_()
        # ep = inpt.edge_attr.clone().requires_grad_()

        # mean_node_pred, mean_edge_pred = self.forward(inpt, final_sigmoid = False)

        # # calculate jacobian w.r.t. node inputs
        # J_node_all = torch.func.jacrev(self.forward_from_node_tensor(inpt.edge_attr, inpt.edge_index))(xp)
        # # J_node_node = torch.flatten(J_node_all[0][:,0,:,:self.input_node_feats],
        # #                             start_dim = 1,
        # #                             end_dim = 2)
        # J_node_edge = torch.flatten(J_node_all[1][:,0,:,:self.input_node_feats],
        #                             start_dim = 1,
        #                             end_dim = 2)

        # # JT_node_node = J_node_node.T
        # JT_node_edge = J_node_edge.T
        
        # cov_node_edge = self.sdp_scale*torch.matmul(torch.matmul(J_node_edge, node_inpt_cov), JT_node_edge)
        # # cov_node_node = self.sdp_scale*torch.matmul(torch.matmul(J_node_node, node_inpt_cov), JT_node_node)

        # # calculate jacobian w.r.t. edge inputs
        # J_edge_all = torch.func.jacrev(self.forward_from_edge_tensor(inpt.x, inpt.edge_index))(ep)
        # # J_edge_node = torch.flatten(J_edge_all[0][:,0,:,:self.input_edge_feats],
        # #                             start_dim = 1,
        # #                             end_dim = 2)
        # J_edge_edge = torch.flatten(J_edge_all[1][:,0,:,:self.input_edge_feats],
        #                             start_dim = 1,
        #                             end_dim = 2)

        # # JT_edge_node = J_edge_node.T
        # JT_edge_edge = J_edge_edge.T

        # cov_edge_edge = self.sdp_scale*torch.matmul(torch.matmul(J_edge_edge, edge_inpt_cov), JT_edge_edge)
        # # cov_edge_node = self.sdp_scale*torch.matmul(torch.matmul(J_edge_node, edge_inpt_cov), JT_edge_node)

        # # std_node_pred = torch.sqrt(torch.diag(cov_node_node + cov_edge_node))
        # std_node_pred = torch.ones_like(mean_node_pred.flatten()) + self.epsilon
        # std_edge_pred = torch.sqrt(torch.diag(cov_node_edge + cov_edge_edge)) + self.epsilon

        # # print ("mean_edge_pred", mean_edge_pred) 
        # # print ("std_edge_pred", std_edge_pred)
        # # node_score = 0.5*(1 + approx_erf(mean_node_pred.flatten()/(std_node_pred*np.sqrt(2))))
        # # edge_score = 0.5*(1 + approx_erf(mean_edge_pred.flatten()/(std_edge_pred*np.sqrt(2))))
        # node_score = approx_cdf(mean_node_pred.flatten()/std_node_pred)
        # edge_score = approx_cdf(mean_edge_pred.flatten()/std_edge_pred)
        
        # return node_score[:,None], edge_score[:,None]

        # return mean_node_pred, mean_edge_pred

    def predict_jacfwd (self, inpt):
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
        J_node_all = torch.func.jacfwd(self.forward_from_node_tensor(inpt.edge_attr, inpt.edge_index))(xp)
        # J_node_node = torch.flatten(J_node_all[0][:,0,:,:self.input_node_feats],
        #                             start_dim = 1,
        #                             end_dim = 2)
        J_node_edge = torch.flatten(J_node_all[1][:,0,:,:self.input_node_feats],
                                    start_dim = 1,
                                    end_dim = 2)

        # JT_node_node = J_node_node.T
        JT_node_edge = J_node_edge.T
        
        cov_node_edge = self.sdp_scale*torch.matmul(torch.matmul(J_node_edge, node_inpt_cov), JT_node_edge)
        # cov_node_node = self.sdp_scale*torch.matmul(torch.matmul(J_node_node, node_inpt_cov), JT_node_node)

        # calculate jacobian w.r.t. edge inputs
        J_edge_all = torch.func.jacfwd(self.forward_from_edge_tensor(inpt.x, inpt.edge_index))(ep)
        # J_edge_node = torch.flatten(J_edge_all[0][:,0,:,:self.input_edge_feats],
        #                             start_dim = 1,
        #                             end_dim = 2)
        J_edge_edge = torch.flatten(J_edge_all[1][:,0,:,:self.input_edge_feats],
                                    start_dim = 1,
                                    end_dim = 2)

        # JT_edge_node = J_edge_node.T
        JT_edge_edge = J_edge_edge.T

        cov_edge_edge = self.sdp_scale*torch.matmul(torch.matmul(J_edge_edge, edge_inpt_cov), JT_edge_edge)
        # cov_edge_node = self.sdp_scale*torch.matmul(torch.matmul(J_edge_node, edge_inpt_cov), JT_edge_node)

        # std_node_pred = torch.sqrt(torch.diag(cov_node_node + cov_edge_node))
        std_node_pred = torch.ones_like(mean_node_pred.flatten()) + self.epsilon
        std_edge_pred = torch.sqrt(torch.diag(cov_node_edge + cov_edge_edge)) + self.epsilon

        # node_score = 0.5*(1 + approx_erf(mean_node_pred.flatten()/(std_node_pred*np.sqrt(2))))
        # edge_score = 0.5*(1 + approx_erf(mean_edge_pred.flatten()/(std_edge_pred*np.sqrt(2))))
        node_score = approx_cdf(mean_node_pred.flatten()/std_node_pred)
        edge_score = approx_cdf(mean_edge_pred.flatten()/std_edge_pred)

        # print ("node", node_score, mean_node_pred.flatten(), std_node_pred)
        # print ("edge", edge_score, mean_edge_pred.flatten(), std_edge_pred)
        
        self.train()
        
        return node_score[:,None], edge_score[:,None]

        # return mean_node_pred, mean_edge_pred

    def predict_autograd (self, inpt):
        # print ("jacobian calculating...")
        # J = torch.vmap(torch.func.jacrev(self.forward_from_node_tensor(inpt.edge_attr, inpt.index)))(inpt.x)
        node_mean = inpt.x[:,:self.input_node_feats]
        edge_mean = inpt.edge_attr[:,:self.input_edge_feats]
        
        node_unc = torch.exp(inpt.x[:,self.input_node_feats:])
        edge_unc = torch.exp(inpt.edge_attr[:,self.input_edge_feats:])

        node_inpt_cov = self.sdp_scale*torch.diag(node_unc*node_unc)
        edge_inpt_cov = self.sdp_scale*torch.diag(edge_unc*edge_unc)

        self.eval()
        # print ("nodes", inpt.x.shape)
        # print ("edges", inpt.edge_attr.shape)
        # xp = inpt.x.clone().requires_grad_()
        # ep = inpt.edge_attr.clone().requires_grad_()
        inpt.x.requires_grad_()
        inpt.edge_attr.requires_grad_()

        mean_node_pred, mean_edge_pred = self.forward(inpt, final_sigmoid = False)

        # print ("mean node pred shape", mean_node_pred.shape)

        std_node_pred = torch.empty_like(mean_node_pred)
        for i in range(mean_node_pred.shape[0]):
            node_jac = torch.autograd.grad(mean_node_pred[i,:].sum(0), inpt.x,
                                           create_graph = True,
                                           retain_graph = True)[0][:,:self.input_node_feats]
            node_pred_unc_node = (node_jac*node_jac*node_unc*node_unc).sum()

            # edge_jac = torch.autograd.grad(mean_node_pred[i,:].sum(0), inpt.edge_attr,
            #                                create_graph = True,
            #                                retain_graph = True)[0][:,:self.input_edge_feats]
            # node_pred_unc_edge = (edge_jac*edge_jac*edge_unc*edge_unc).sum()

            std_node_pred[i] = torch.sqrt(node_pred_unc_node)
            # print (mean_node_pred[i,:], torch.sqrt(node_pred_unc_node))

        std_edge_pred = torch.empty_like(mean_edge_pred)
        # print (std_edge_pred.shape)
        for i in range(mean_edge_pred.shape[0]):
            node_jac = torch.autograd.grad(mean_edge_pred[i,:].sum(0), inpt.x,
                                           create_graph = False,
                                           retain_graph = True)[0][:,:self.input_node_feats]
            edge_pred_unc_node = (node_jac*node_jac*node_unc*node_unc).sum()

            edge_jac = torch.autograd.grad(mean_edge_pred[i,:].sum(0), inpt.edge_attr,
                                           create_graph = False,
                                           retain_graph = True)[0][:,:self.input_edge_feats]
            edge_pred_unc_edge = (edge_jac*edge_jac*edge_unc*edge_unc).sum()

            std_edge_pred[i] = torch.sqrt(edge_pred_unc_node + edge_pred_unc_edge)
            # print (mean_edge_pred[i,:], torch.sqrt(edge_pred_unc_node + edge_pred_unc_edge))

        # calculate jacobian w.r.t. node inputs
        # J_node_all = torch.func.jacrev(self.forward_from_node_tensor(inpt.edge_attr, inpt.edge_index))(xp)
        # # J_node_node = torch.flatten(J_node_all[0][:,0,:,:self.input_node_feats],
        # #                             start_dim = 1,
        # #                             end_dim = 2)
        # J_node_edge = torch.flatten(J_node_all[1][:,0,:,:self.input_node_feats],
        #                             start_dim = 1,
        #                             end_dim = 2)

        # # JT_node_node = J_node_node.T
        # JT_node_edge = J_node_edge.T
        
        # cov_node_edge = torch.matmul(torch.matmul(J_node_edge, node_inpt_cov), JT_node_edge)
        # # cov_node_node = torch.matmul(torch.matmul(J_node_node, node_inpt_cov), JT_node_node)

        # # calculate jacobian w.r.t. edge inputs
        # J_edge_all = torch.func.jacrev(self.forward_from_edge_tensor(inpt.x, inpt.edge_index))(ep)
        # # J_edge_node = torch.flatten(J_edge_all[0][:,0,:,:self.input_edge_feats],
        # #                             start_dim = 1,
        # #                             end_dim = 2)
        # J_edge_edge = torch.flatten(J_edge_all[1][:,0,:,:self.input_edge_feats],
        #                             start_dim = 1,
        #                             end_dim = 2)





        
        # # JT_edge_node = J_edge_node.T
        # JT_edge_edge = J_edge_edge.T

        # cov_edge_edge = torch.matmul(torch.matmul(J_edge_edge, edge_inpt_cov), JT_edge_edge)
        # # cov_edge_node = torch.matmul(torch.matmul(J_edge_node, edge_inpt_cov), JT_edge_node)

        # # std_node_pred = torch.sqrt(torch.diag(cov_node_node + cov_edge_node))
        # std_node_pred = torch.ones_like(mean_node_pred.flatten())
        # std_edge_pred = torch.sqrt(torch.diag(cov_node_edge + cov_edge_edge))

        # # print ("mean_edge_pred", mean_edge_pred)
        # # print ("std_edge_pred", std_edge_pred)
        # # node_score = 0.5*(1 + approx_erf(mean_node_pred.flatten()/(std_node_pred*np.sqrt(2))))
        # # edge_score = 0.5*(1 + approx_erf(mean_edge_pred.flatten()/(std_edge_pred*np.sqrt(2))))

        print ("pre CDF", mean_node_pred, std_node_pred)
        
        node_score = approx_cdf(mean_node_pred.flatten()/std_node_pred.flatten())
        edge_score = approx_cdf(mean_edge_pred.flatten()/std_edge_pred.flatten())

        print (node_score)
        
        return node_score[:,None], edge_score[:,None]

class EdgeConv_SDP_blindtrained(torch.nn.Module):
    def __init__(self, input_node_feats = 16, input_edge_feats = 19, sdp_scale = 1, epsilon = 1.e-3):
        super().__init__()
        self.input_node_feats = input_node_feats
        self.input_edge_feats = input_edge_feats
        self.inner_GNN = EdgeConvNet(2*input_node_feats,
                                     2*input_edge_feats,
                                     final_sigmoid = False,
                                     aggr = 'add')

        self.sdp_scale = sdp_scale
        self.epsilon = epsilon
        
    def forward(self, inpt, final_sigmoid = True):
        # mean = GraphData(x = inpt.x[:,:self.input_node_feats],
        #                  edge_index = inpt.edge_index,
        #                  edge_attr = inpt.edge_attr[:,:self.input_edge_feats],
        #                  y = inpt.y,
        #                  edge_label = inpt.edge_label,
        #                  index = inpt.index)

        # node_unc = torch.exp(inpt.x[:,self.input_node_feats:])
        # edge_unc = torch.exp(inpt.edge_attr[:,self.input_edge_feats:])
        
        # node_inf, edge_inf =  self.inner_GNN(mean)
        node_inf, edge_inf =  self.inner_GNN(inpt)

        if final_sigmoid:
            return torch.sigmoid(node_inf), torch.sigmoid(edge_inf)
        else:
            return node_inf, edge_inf

    def forward_from_tensors(self, node_feats, edge_feats, edge_index):
        # node_mean = node_feats[:,:self.input_node_feats]
        # edge_mean = edge_feats[:,:self.input_edge_feats]
        
        # node_inf, edge_inf =  self.inner_GNN.forward_from_tensors(node_mean, edge_mean, edge_index)
        node_inf, edge_inf =  self.inner_GNN.forward_from_tensors(node_feats, edge_feats, edge_index)

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
        
        node_var = torch.pow(torch.flatten(torch.exp(inpt.x[:,self.input_node_feats:])), 2)
        edge_var = torch.pow(torch.flatten(torch.exp(inpt.edge_attr[:,self.input_edge_feats:])), 2)

        pred_node_mean, pred_edge_mean = self.forward(inpt, final_sigmoid = False)

        self.eval()

        J_node_all = torch.func.jacrev(self.forward_from_tensors, argnums = 0)(inpt.x, inpt.edge_attr, inpt.edge_index)
        J_node_node = torch.flatten(J_node_all[0][:,0,:,:self.input_node_feats],
                                    start_dim = 1,
                                    end_dim = 2)
        J_node_edge = torch.flatten(J_node_all[1][:,0,:,:self.input_node_feats],
                                    start_dim = 1,
                                    end_dim = 2)

        J_edge_all = torch.func.jacrev(self.forward_from_tensors, argnums = 1)(inpt.x, inpt.edge_attr, inpt.edge_index)
        J_edge_node = torch.flatten(J_edge_all[0][:,0,:,:self.input_edge_feats],
                                    start_dim = 1,
                                    end_dim = 2)
        J_edge_edge = torch.flatten(J_edge_all[1][:,0,:,:self.input_edge_feats],
                                    start_dim = 1,
                                    end_dim = 2)

        pred_node_std = torch.sqrt(self.sdp_scale*torch.diag((J_node_node*node_var)@J_node_node.T) + torch.diag((J_edge_node*edge_var)@J_edge_node.T))
        pred_node_std = pred_node_std[:,None]
        pred_edge_std = torch.sqrt(self.sdp_scale*torch.diag((J_node_edge*node_var)@J_node_edge.T) + torch.diag((J_edge_edge*edge_var)@J_edge_edge.T))
        pred_edge_std = pred_edge_std[:,None]
        
        pred_node_mean = pred_node_mean[:, :self.input_node_feats]
        pred_edge_mean = pred_edge_mean[:, :self.input_edge_feats]

        # print (pred_node_mean.shape, pred_node_std.shape)
        # print (pred_edge_mean.shape, pred_edge_std.shape)
        
        node_score = approx_cdf(pred_node_mean/pred_node_std)
        edge_score = approx_cdf(pred_edge_mean/pred_edge_std)

        # print ("node", node_score, pred_node_mean, pred_node_std.shape)
        # print ("edge", edge_score, pred_edge_mean, pred_edge_std.shape)
        
        # print ("node", node_score)
        # print ("edge", edge_score)
        
        self.train()
        
        return node_score, edge_score
