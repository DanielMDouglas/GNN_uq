import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset

def plot_scatter3d(voxels, labels={}, edge_index=None, clust_labels=None, markersize=5, colorscale=None):
    
    import copy
    import plotly.graph_objects as go # Plotly graph objects
    from plotly.offline import iplot  # Interactive plots
    import plotly.express as px # For discrete color scale
    
    if colorscale is None:
        colorscale = px.colors.qualitative.Dark24

    # Initialize a graph with no labels
    blank = go.Scatter3d(x = voxels[:,0],
                         y = voxels[:,1],
                         z = voxels[:,2],
                         mode = 'markers', 
                         marker = dict(
                             size = 2, 
                             colorscale = colorscale
                         )
                        )
    
    # Loop over the set of labels, make a graph for each
    graphs = []
    for key, label in labels.items():
        graph = copy.copy(blank)
        graph['name'] = key
        graph['marker']['color'] = label
        graph['hovertext'] = [f'{l:0.0f}' for l in label]
        graphs.append(graph)
        
    if not len(graphs):
        graphs.append(blank)
        
    # Draw edges if requested
    if edge_index is not None:
        import scipy as sp
        edge_vertices = []
        clust_labels = np.unique(clust_labels, return_inverse=True)[1]
        for i, j in edge_index:
            vi, vj = voxels[clust_labels==i], voxels[clust_labels==j]
            d12 = sp.spatial.distance.cdist(vi, vj, 'euclidean')
            i1, i2 = np.unravel_index(np.argmin(d12), d12.shape)
            edge_vertices.append([vi[i1], vj[i2], [None, None, None]])

        edge_vertices = np.concatenate(edge_vertices)

        graphs.append(go.Scatter3d(x = edge_vertices[:,0], y = edge_vertices[:,1], z = edge_vertices[:,2],
                                   mode = 'lines',
                                   name = 'Predicted edges',
                                   line = dict(
                                        width = 2,
                                        color = 'Blue'
                                    ),
                                    hoverinfo = 'none'))

    iplot(graphs)

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

    def __del__(self):
        
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            
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

        # noise_lb, noise_ub = 0.02, 0.05
        # noise_lb, noise_ub = 0.05, 0.20
        # noise_lb, noise_ub = 0.05, 0.10
        noise_lb, noise_ub = 0.1, 0.3
        # noise_lb, noise_ub = 0.2, 0.5
        if self._mode in ['UA', 'blind']:
            node_noise = noise_lb + (noise_ub - noise_lb)*torch.rand(node_features.shape) # noise magnitude for UA/blind
            node_noise *= torch.abs(node_features)
            noisy_node_features = torch.normal(node_features, node_noise) # noisy features or UA/blind
            # print (node_features)
            # node_mean = torch.Tensor([]) # mean of node features
            # node_std = torch.Tensor([]) # mean of node features

            # noise_mean = 0.5*(noise_lb + noise_ub)*node_mean
            # noise_std = 0.5*(noise_lb + noise_ub)*node_mean
            # # print (torch.log(node_noise).shape)
            # # print (torch.mean(torch.log(node_noise), dim = 0),
            # #        torch.std(torch.log(node_noise), dim = 0),
            # #        )

            edge_noise = noise_lb + (noise_ub - noise_lb)*torch.rand(edge_features.shape)
            edge_noise *= torch.abs(edge_features)
            noisy_edge_features = torch.normal(edge_features, edge_noise)
            # print (edge_features - noisy_edge_features)

        elif self._mode in ['nonoise']:
            node_noise = torch.zeros_like(node_features.shape) # for no noise
            noisy_node_features = node_features # for no noise

            edge_noise = torch.zeros_like(edge_features.shape)
            noisy_edge_features = edge_features
            
        if self._mode in ['UA', 'nonoise']:
            node_uq_features = torch.cat((noisy_node_features, node_noise), dim = -1) # for UA/no noise
            edge_uq_features = torch.cat((noisy_edge_features, edge_noise), dim = -1)
        elif self._mode in ['blind']:
            node_uq_features = torch.cat((noisy_node_features, torch.zeros_like(node_noise)), dim = -1) # for blind
            edge_uq_features = torch.cat((noisy_edge_features, torch.zeros_like(edge_noise)), dim = -1)
        
        return GraphData(x = node_uq_features,
                         edge_index = edge_index,
                         edge_attr = edge_uq_features,
                         y = node_labels,
                         edge_label = edge_labels,
                         index = idx)

    def get(self,idx):
        return self[idx]

class ShowerFeaturesPared(GraphDataset):
    """
    class: an interface for shower fragment data files. This Dataset is designed to produce a batch of
           of node and edge feature data.
    """
    def __init__(self, file_path, mode, noise_interval = (0.2, 0.5)):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._mode = mode
        self._file_handle = None
        self._noise_interval = noise_interval
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['node_features'])

    def __del__(self):
        
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            
    def __len__(self):
        return self._entries

    def len(self):
        return len(self._entries)

    def __getitem__(self, idx):
        
        # Get the subset of node and edge features that correspond to the requested event ID
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)
            
        node_info = torch.tensor(self._file_handle['node_features'][idx].reshape(-1, 19), dtype=torch.float32)
        node_feature_selection = torch.Tensor([0, 1, 2,
                                               3, #4, 5,
                                               6, 7, #8,
                                               9, 10, 11,
                                               #12, 13, 14,
                                               15]).long()
        node_features = node_info[:,node_feature_selection]
        group_ids = node_info[:,-2].long()
        node_labels = node_info[:,-1].long()

        edge_info = torch.tensor(self._file_handle['edge_features'][idx].reshape(-1, 22),
                                 dtype=torch.float32)
        edge_feature_selection = torch.Tensor([0, 1, 2,
                                               3, 4, 5,
                                               # 6, 7, 8,
                                               # 9,
                                               # 10, 11, 12,
                                               # 13, 14, 15,
                                               # 16, 17, 18,
        ]).long()
        edge_features = edge_info[:,edge_feature_selection]
        edge_index = edge_info[:,-3:-1].long().t()
        edge_labels = edge_info[:,-1].long()

        # noise_lb, noise_ub = 0.02, 0.05
        # noise_lb, noise_ub = 0.05, 0.20
        # noise_lb, noise_ub = 0.05, 0.10
        # noise_lb, noise_ub = 0.1, 0.3
        # noise_lb, noise_ub = 0.2, 0.3
        # noise_lb, noise_ub = 0.2, 0.5
        noise_lb, noise_ub = self._noise_interval[0], self._noise_interval[1]
        # import time
        if self._mode in ['UA', 'blind']:
            # t1 = time.clock_gettime(0)
            node_noise = noise_lb + (noise_ub - noise_lb)*torch.rand(node_features.shape) # noise magnitude for UA/blind
            node_noise *= torch.abs(node_features)
            node_noise += 1.e-4
            # print (torch.all(node_noise >= 0))
            noisy_node_features = torch.normal(node_features, node_noise) # noisy features or UA/blind

            # t2 = time.clock_gettime(0)
            
            edge_noise = noise_lb + (noise_ub - noise_lb)*torch.rand(edge_features.shape)
            edge_noise *= torch.abs(edge_features)
            edge_noise += 1.e-4
            noisy_edge_features = torch.normal(edge_features, edge_noise)
            # t3 = time.clock_gettime(0)
            # print ("making noise", t3 - t2, t2 - t1)

        elif self._mode in ['nonoise']:
            noisy_node_features = node_features # for no noise
            noisy_edge_features = edge_features

        if self._mode in ['UA']:
            # ti = time.clock_gettime(0)
            node_uq_features = torch.cat((noisy_node_features,
                                          torch.log(node_noise)),
                                         dim = -1) # for UA/no noise
            edge_uq_features = torch.cat((noisy_edge_features,
                                          torch.log(edge_noise)),
                                         dim = -1)
            # tf = time.clock_gettime(0)
            # print ("packaging noise", tf - ti)

        elif self._mode in ['blind', 'nonoise']:
            # ti = time.clock_gettime(0)
            node_uq_features = torch.cat((noisy_node_features,
                                          torch.zeros_like(noisy_node_features)),
                                         dim = -1) # for blind
            edge_uq_features = torch.cat((noisy_edge_features,
                                          torch.zeros_like(noisy_edge_features)),
                                         dim = -1)
            # tf = time.clock_gettime(0)
            # print ("ignoring noise", tf - ti)

                    
        return GraphData(x = node_uq_features,
                         edge_index = edge_index,
                         edge_attr = edge_uq_features,
                         y = node_labels,
                         edge_label = edge_labels,
                         index = idx)

    def get(self,idx):
        return self[idx]

class ShowerFeaturesParedSDP(GraphDataset):
    """
    class: an interface for shower fragment data files. This Dataset is designed to produce a batch of
           of node and edge feature data.
    """
    def __init__(self, file_path, noise_interval = (0.2, 0.5)):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._file_handle = None
        self._noise_interval = noise_interval
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['node_features'])

    def __del__(self):
        
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            
    def __len__(self):
        return self._entries

    def len(self):
        return len(self._entries)

    def __getitem__(self, idx):
        
        # Get the subset of node and edge features that correspond to the requested event ID
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)
            
        node_info = torch.tensor(self._file_handle['node_features'][idx].reshape(-1, 19), dtype=torch.float32)
        node_feature_selection = torch.Tensor([0, 1, 2,
                                               3, #4, 5,
                                               6, 7, #8,
                                               9, 10, 11,
                                               #12, 13, 14,
                                               15]).long()
        node_features = node_info[:,node_feature_selection]
        group_ids = node_info[:,-2].long()
        node_labels = node_info[:,-1].long()

        edge_info = torch.tensor(self._file_handle['edge_features'][idx].reshape(-1, 22),
                                 dtype=torch.float32)
        edge_feature_selection = torch.Tensor([0, 1, 2,
                                               3, 4, 5,
                                               # 6, 7, 8,
                                               # 9,
                                               # 10, 11, 12,
                                               # 13, 14, 15,
                                               # 16, 17, 18,
        ]).long()
        edge_features = edge_info[:,edge_feature_selection]
        edge_index = edge_info[:,-3:-1].long().t()
        edge_labels = edge_info[:,-1].long()

        # noise_lb, noise_ub = 0.02, 0.05
        # noise_lb, noise_ub = 0.05, 0.20
        # noise_lb, noise_ub = 0.05, 0.10
        # noise_lb, noise_ub = 0.1, 0.3
        # noise_lb, noise_ub = 0.2, 0.3
        # noise_lb, noise_ub = 0.2, 0.5
        noise_lb, noise_ub = self._noise_interval[0], self._noise_interval[1]
        # import time

        # t1 = time.clock_gettime(0)
        node_noise = noise_lb + (noise_ub - noise_lb)*torch.rand(node_features.shape) # noise magnitude for UA/blind
        node_noise *= torch.abs(node_features)
        node_noise += 1.e-4
        # print (torch.all(node_noise >= 0))
        noisy_node_features = torch.normal(node_features, node_noise) # noisy features or UA/blind
        
        # t2 = time.clock_gettime(0)
        
        edge_noise = noise_lb + (noise_ub - noise_lb)*torch.rand(edge_features.shape)
        edge_noise *= torch.abs(edge_features)
        edge_noise += 1.e-4
        noisy_edge_features = torch.normal(edge_features, edge_noise)
        # t3 = time.clock_gettime(0)
        # print ("making noise", t3 - t2, t2 - t1)
        
        # ti = time.clock_gettime(0)
        node_uq_features = torch.cat((noisy_node_features,
                                      torch.log(node_noise)),
                                     dim = -1) # for UA/no noise
        edge_uq_features = torch.cat((noisy_edge_features,
                                      torch.log(edge_noise)),
                                     dim = -1)
        # tf = time.clock_gettime(0)
        # print ("packaging noise", tf - ti)

        mean_graph_data = GraphData(x = noisy_node_features,
                                    edge_index = edge_index,
                                    edge_attr = noisy_edge_features,
                                    y = node_labels,
                                    edge_label = edge_labels,
                                    index = idx)
        std_graph_data = GraphData(x = node_noise,
                                   edge_index = edge_index,
                                   edge_attr = edge_noise,
                                   y = node_labels,
                                   edge_label = edge_labels,
                                   index = idx)
        
        return mean_graph_data, std_graph_data
    
    def get(self,idx):
        return self[idx]

class ShowerFeaturesMeanonly(GraphDataset):
    """
    class: an interface for shower fragment data files. This Dataset is designed to produce a batch of
           of node and edge feature data.
    """
    def __init__(self, file_path, mode, noise_interval = (0.2, 0.5)):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._mode = mode
        self._file_handle = None
        self._noise_interval = noise_interval
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['node_features'])

    def __del__(self):
        
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            
    def __len__(self):
        return self._entries

    def len(self):
        return len(self._entries)

    def __getitem__(self, idx):
        
        # Get the subset of node and edge features that correspond to the requested event ID
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)
            
        node_info = torch.tensor(self._file_handle['node_features'][idx].reshape(-1, 19), dtype=torch.float32)
        node_feature_selection = torch.Tensor([0, 1, 2,
                                               3, #4, 5,
                                               6, 7, #8,
                                               9, 10, 11,
                                               #12, 13, 14,
                                               15]).long()
        node_features = node_info[:,node_feature_selection]
        group_ids = node_info[:,-2].long()
        node_labels = node_info[:,-1].long()

        edge_info = torch.tensor(self._file_handle['edge_features'][idx].reshape(-1, 22),
                                 dtype=torch.float32)
        edge_feature_selection = torch.Tensor([0, 1, 2,
                                               3, 4, 5,
                                               # 6, 7, 8,
                                               # 9,
                                               # 10, 11, 12,
                                               # 13, 14, 15,
                                               # 16, 17, 18,
        ]).long()
        edge_features = edge_info[:,edge_feature_selection]
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
