import numpy as np
import matplotlib.pyplot as plt
import h5py
import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ShowerVoxels(Dataset):
    """
    class: an interface to access shower fragment voxels.  This Dataset is designed to produce list of shower voxels
    """
    def __init__(self, file_path):
        """
        Args: file_path ...... path to the HDF5 file that contains the feature data
        """
        self._file_path = file_path
        self._file_handle = None
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['voxels'])

    # def __del__(self):

    #     if self._file_handle is not None:
    #         self._file_handle.close()
    #         self._file_handle = None

    def __len__(self):

        return self._entries

    def __getitem__(self, idx):

        # get the subset of voxels that correspond to the requested entry
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)
            
        voxel_info = self._file_handle['voxels'][idx].reshape(-1,6)
        clusts = [np.where(voxel_info[:,3]==c)[0] for c in np.unique(voxel_info[:,3])]
        voxels = [torch.tensor(voxel_info[c,:3], dtype=torch.float32) for c in clusts]
        frag_ids = np.array([voxel_info[c[0],3] for c in clusts], dtype=np.int64)
        shower_ids = np.array([voxel_info[c[0],4] for c in clusts], dtype=np.int64)
        primary_ids = np.array([voxel_info[c[0],5] for c in clusts], dtype=np.int64)
            
        return dict(
            voxels = voxels,
            frag_ids = frag_ids,
            shower_ids = shower_ids,
            primary_ids = primary_ids
        )

class ShowerFeatures(GraphDataset):
    """
    class: an interface for shower fragment data files. This Dataset is designed to produce a batch of
           of node and edge feature data.
    """
    def __init__(self, file_path, mode):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._mode = mode
        self._file_handle = None
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['node_features'])

    # def __del__(self):
        
    #     if self._file_handle is not None:
    #         self._file_handle.close()
    #         self._file_handle = None
            
    def __len__(self):
        return self._entries

    def len(self):
        return len(self._entries)

    def __getitem__(self, idx):
        
        # Get the subset of node and edge features that correspond to the requested event ID
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)
            
        node_info = torch.tensor(self._file_handle['node_features'][idx].reshape(-1, 19), dtype=torch.float32)
        node_features = node_info[:,:-3]
        group_ids = node_info[:,-2].long()
        node_labels = node_info[:,-1].long()

        edge_info = torch.tensor(self._file_handle['edge_features'][idx].reshape(-1, 22),
                                 dtype=torch.float32)
        edge_features = edge_info[:,:-3]
        edge_index = edge_info[:,-3:-1].long().t()
        edge_labels = edge_info[:,-1].long()

        return GraphData(x = node_features,
                         edge_index = edge_index,
                         edge_attr = edge_features,
                         y = node_labels,
                         edge_label = edge_labels,
                         index = idx)

    def get(self,idx):
        return self[idx]

class ShowerFeaturesFromVoxels(GraphDataset):
    """
    class: an interface for shower fragment data files. This Dataset is designed to produce a batch of
           of node and edge feature data.
    """
    def __init__(self, file_path, mode, dropout_eff = 1):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._mode = mode
        # self._current = 0
        self._dropout_eff = dropout_eff
        self._file_handle = None
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['node_features'])

    # def __del__(self):
        
    #     if self._file_handle is not None:
    #         self._file_handle.close()
    #         self._file_handle = None
            
    def __len__(self):
        return self._entries

    def len(self):
        return len(self._entries)

    # def __next__(self):
    #     if self._current >= self._entries:
    #         raise StopIteration

    #     try:
    #         yield self.__getitem__(self._current)
    #     except ValueError:
    #         self._current += 1

    #     self._current += 1
            
    def __getitem__(self, idx):

        # Get the subset of node and edge features that correspond to the requested event ID
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)

        voxel_info = self._file_handle['voxels'][idx].reshape(-1,6)
        clusts = [np.where(voxel_info[:,3]==c)[0] for c in np.unique(voxel_info[:,3])]
        voxels = [torch.tensor(voxel_info[c,:3], dtype=torch.float32).to(device) for c in clusts]
        frag_ids = np.array([voxel_info[c[0],3] for c in clusts], dtype=np.int64)
        shower_ids = np.array([voxel_info[c[0],4] for c in clusts], dtype=np.int64)
        primary_ids = np.array([voxel_info[c[0],5] for c in clusts], dtype=np.int64)

        # drop out certain voxels from the image
        # treat each voxel independently, with a certain
        # efficiency to remain in the sample
        # smaller fragments will be affected more
        voxel_mask = [(torch.rand(frag.shape[0]) <= self._dropout_eff).to(device) for frag in voxels]

        # make the node (fragment) features
        N_frags = len(voxels)
        center_of_mass = torch.stack([torch.mean(frag[mask], dim = 0) for frag, mask in zip(voxels, voxel_mask)])
        covariance = torch.stack([torch.cov(frag[mask].T, correction = 1) for frag, mask in zip(voxels, voxel_mask)])
        eigh = torch.linalg.eigh(covariance)
        # L = eigh.eigenvalues
        V = eigh.eigenvectors
        principal_axis = V[:,:,2]
        N_voxels = torch.Tensor([frag[mask].shape[0] for frag, mask in zip(voxels, voxel_mask)]).to(device)

        node_features = torch.cat((center_of_mass,
                                   covariance.reshape(N_frags, 9),
                                   principal_axis,
                                   N_voxels.reshape(N_frags, 1),
                                   ), dim = 1)
        
        node_info = torch.tensor(self._file_handle['node_features'][idx].reshape(-1, 19), dtype=torch.float32)
        group_ids = node_info[:,-2].long()
        node_labels = node_info[:,-1].long()

        # make the edge features 
        edge_info = torch.tensor(self._file_handle['edge_features'][idx].reshape(-1, 22),
                                 dtype=torch.float32)
        # edge_features_old = edge_info[:,:-3]
        edge_index = edge_info[:,-3:-1].long().t()
        edge_labels = edge_info[:,-1].long()

        edge_features = torch.empty((0,19)).to(device)
        for nodeA_ind, nodeB_ind in zip(edge_index[0], edge_index[1]):
            nodeA = voxels[nodeA_ind][voxel_mask[nodeA_ind]]
            nodeB = voxels[nodeB_ind][voxel_mask[nodeB_ind]]
            
            cdistA = torch.cdist(nodeA, center_of_mass[None,nodeB_ind])
            cdistB = torch.cdist(center_of_mass[None,nodeA_ind], nodeB)
            # cdistB= torch.cdist(nodeA, nodeB)
            # closestA = nodeA[torch.max(cdist == torch.min(cdist), dim = 1).values][0]
            # closestB = nodeB[torch.max(cdist == torch.min(cdist), dim = 0).values][0]
            closestA = nodeA[torch.max(cdistA == torch.min(cdistA), dim = 1).values][0]
            closestB = nodeB[torch.max(cdistB == torch.min(cdistB), dim = 0).values][0]
            # print (closestA)
            dV = closestA - closestB
            dV_norm = torch.norm(dV)
            if torch.all(dV == 0):
                dV_unit = dV
            else:
                dV_unit = dV/dV_norm

            dV_outer = torch.outer(dV_unit, dV_unit).flatten()

            # for i, var in enumerate([closestA, closestB, dV_unit, dV_norm.reshape(1), dV_outer]):
            #     if torch.any(torch.isnan(var)):
            #         print (i, var)
            #         print (torch.cat((closestA,
            #                           closestB,
            #                           dV_unit,
            #                           dV_norm.reshape(1),
            #                           dV_outer,
            #         )))

            #         raise ValueError

            this_edge_feat = torch.cat((closestA,
                                        closestB,
                                        dV_unit,
                                        dV_norm.reshape(1),
                                        dV_outer,
                                        ))

            edge_features = torch.cat((edge_features, this_edge_feat.reshape((1, 19))))

        # if torch.any(torch.isnan(node_features)) or torch.any(torch.isnan(edge_features)):
        #     print(torch.any(torch.isnan(node_features)))
        #     print(torch.any(torch.isnan(edge_features)))
        #     print (edge_features.shape)
        #     print(torch.argmax(torch.isnan(edge_features).int(), dim = 0))
        #     print(torch.argmax(torch.isnan(edge_features).int(), dim = 1))
        #     # print(edge_features[:,0])
        #     print ("FUCK")
        #     raise ValueError
        return GraphData(x = node_features,
                         edge_index = edge_index,
                         edge_attr = edge_features,
                         y = node_labels,
                         edge_label = edge_labels,
                         index = idx)

    def get(self,idx):
        return self[idx]
