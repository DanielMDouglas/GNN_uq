import numpy as np
import matplotlib.pyplot as plt
import h5py
import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset

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

    def __getitem__(self, idx):
        
        # Get the subset of node and edge features that correspond to the requested event ID
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)

        voxel_info = self._file_handle['voxels'][idx].reshape(-1,6)
        clusts = [np.where(voxel_info[:,3]==c)[0] for c in np.unique(voxel_info[:,3])]
        voxels = [torch.tensor(voxel_info[c,:3], dtype=torch.float32) for c in clusts]
        frag_ids = np.array([voxel_info[c[0],3] for c in clusts], dtype=np.int64)
        shower_ids = np.array([voxel_info[c[0],4] for c in clusts], dtype=np.int64)
        primary_ids = np.array([voxel_info[c[0],5] for c in clusts], dtype=np.int64)

        # drop out certain voxels from the image
        # treat each voxel independently, with a certain
        # efficiency to remain in the sample
        # smaller fragments will be affected more
        voxel_mask = [torch.rand(frag.shape[0]) <= self._dropout_eff for frag in voxels]

        # make the node (fragment) features
        N_frags = len(voxels)
        center_of_mass = torch.stack([torch.mean(frag[mask], dim = 0) for frag, mask in zip(voxels, voxel_mask)])
        covariance = torch.stack([torch.cov(frag[mask].T, correction = 1) for frag, mask in zip(voxels, voxel_mask)])
        eigh = torch.linalg.eigh(covariance)
        # L = eigh.eigenvalues
        V = eigh.eigenvectors
        principal_axis = V[:,:,2]
        N_voxels = torch.Tensor([frag[mask].shape[0] for frag, mask in zip(voxels, voxel_mask)])

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

        edge_features = torch.empty((0,19))
        for nodeA_ind, nodeB_ind in zip(edge_index[0], edge_index[1]):
            nodeA = voxels[nodeA_ind][voxel_mask[nodeA_ind]]
            nodeB = voxels[nodeB_ind][voxel_mask[nodeB_ind]]

            cdist = torch.cdist(nodeA, nodeB)
            closestA = nodeA[torch.max(cdist == torch.min(cdist), dim = 1).values][0]
            closestB = nodeB[torch.max(cdist == torch.min(cdist), dim = 0).values][0]
            dV = closestA - closestB
            dV_norm = torch.norm(dV)
            dV_unit = dV/dV_norm

            dV_outer = torch.outer(dV_unit, dV_unit).flatten()

            this_edge_feat = torch.cat((closestA,
                                        closestB,
                                        dV_unit,
                                        dV_norm.reshape(1),
                                        dV_outer,
                                        ))

            edge_features = torch.cat((edge_features, this_edge_feat.reshape((1, 19))))

        return GraphData(x = node_features,
                         edge_index = edge_index,
                         edge_attr = edge_features,
                         y = node_labels,
                         edge_label = edge_labels,
                         index = idx)

    def get(self,idx):
        return self[idx]


no_dropout = ShowerFeaturesFromVoxels('../data/if-graph-test.h5', 'UA')
frac_dropout = ShowerFeaturesFromVoxels('../data/if-graph-test.h5', 'UA', dropout_eff = 0.8)

nImages = 50
nThrows = 100
node_feature_ratio = []

Nbins = 100
# bins = torch.linspace(0, 2, Nbins+1)
CoM_bins = torch.linspace(-0.02, 0.02, Nbins+1)
cov_bins = torch.linspace(-1, 1, Nbins+1)
pa_bins = torch.linspace(-1, 1, Nbins+1)
N_vox_bins = torch.linspace(-1, 0.1, Nbins+1)

binArray = [CoM_bins, CoM_bins, CoM_bins,
            cov_bins, cov_bins, cov_bins,
            cov_bins, cov_bins, cov_bins,
            cov_bins, cov_bins, cov_bins,
            pa_bins, pa_bins, pa_bins,
            N_vox_bins,
            ]
binCenterArray = [0.5*(bins[1:] + bins[:-1])
                  for bins in binArray]

# image_ind = 0

feature_ratio_hists = torch.zeros((16, Nbins))
for image_ind in range(nImages):
# for image_ind in range(len(frac_dropout)):
    nominal_features = no_dropout[image_ind]['x']
    # print (nominal_features.shape)
    for i in tqdm.tqdm(range(nThrows)):
        try:
            this_throw = frac_dropout[image_ind]['x']
            frac_diff = (this_throw - nominal_features)/nominal_features
            for j in range(nominal_features.shape[1]):
                counts = torch.histogram(frac_diff[:,j],
                                         bins = binArray[j]).hist
                feature_ratio_hists[j] += counts
        except RuntimeError:
            continue

print (feature_ratio_hists)
# for i in range(nTrials):
#     print (frac_dropout[image_ind]['x']/nominal_features)

feat_labels = [r'Center-of-Mass x [(throw - nominal)/nominal]',
               r'Center-of-Mass y [(throw - nominal)/nominal]',
               r'Center-of-Mass z [(throw - nominal)/nominal]',
               r'Covariance x-x [(throw - nominal)/nominal]',
               r'Covariance x-y [(throw - nominal)/nominal]',
               r'Covariance x-z [(throw - nominal)/nominal]',
               r'Covariance y-x [(throw - nominal)/nominal]',
               r'Covariance y-y [(throw - nominal)/nominal]',
               r'Covariance y-z [(throw - nominal)/nominal]',
               r'Covariance z-x [(throw - nominal)/nominal]',
               r'Covariance z-y [(throw - nominal)/nominal]',
               r'Covariance z-z [(throw - nominal)/nominal]',
               r'Principal Axis x [(throw - nominal)/nominal]',
               r'Principal Axis y [(throw - nominal)/nominal]',
               r'Principal Axis z [(throw - nominal)/nominal]',
               r'N Voxels [(throw - nominal)/nominal]',
               ]

import scipy.stats as st
import scipy.optimize as opt
for feature_ind in range(nominal_features.shape[1]):
    binEdges = binArray[feature_ind]
    binWidth = binEdges[1] - binEdges[0]
    binCenters = binCenterArray[feature_ind]

    fig = plt.figure()
    
    norm = 1./binWidth/torch.sum(feature_ratio_hists[feature_ind])
    plt.stairs(norm*feature_ratio_hists[feature_ind], binArray[feature_ind], fill = True)
    
    print (binWidth)
    obs = (norm*feature_ratio_hists[feature_ind]).numpy()
    print (np.sum(binWidth.numpy()*obs))
    mean = np.sum(binWidth.numpy()*obs*binCenters.numpy())
    var = np.sum(binWidth.numpy()*obs*binCenters.numpy()**2) - mean**2
    std = np.sqrt(var)
    print ("mean", mean)
    print ("var", var)
    print ("std", std)
    
    def x_sq(args):
        loc, scale = args
        pred_prob = st.norm.pdf(binCenterArray[feature_ind],
                                loc = loc, scale = scale)
        return np.sum((pred_prob - obs)**2)
    # fit = opt.fmin_l_bfgs_b(x_sq,
    fit = opt.fmin(x_sq,
                   (mean, std),
                   # approx_grad = True,
                   )
    print (fit)
    # bf_loc, bf_width = fit[0]
    bf_loc, bf_width = fit
    
    x_plot_space = np.linspace(binArray[feature_ind][0], binArray[feature_ind][-1], 1000)
    prob = st.norm.pdf(x_plot_space, loc = bf_loc, scale = bf_width)
    plt.plot(x_plot_space, prob)
    plt.xlabel(feat_labels[feature_ind])
    plt.figtext(0.15, 0.75, "mean = "+str(round(mean, 2))+" \nstd. dev = %.2E"%std)
    plt.figtext(0.15, 0.6, "Best-fit Normal: \nloc = "+str(round(bf_loc, 2))+" \nscale = %.2E"%bf_width)
    plt.savefig(r'feature'+str(feature_ind)+'.png')
