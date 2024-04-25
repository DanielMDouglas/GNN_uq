import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class ShowerVoxels(Dataset):
    """
    class: an interface to access shower fragment voxels. This Dataset is designed to produce a shower voxels
    """
    def __init__(self, file_path):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._file_handle = None
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['voxels'])
        
    def __del__(self):
        
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __len__(self):
        
        return self._entries

    def __getitem__(self, idx):
            
        # Get the subset of voxels that correspond to the requested entry
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


def collate_voxels(batch):
    
    import itertools
    
    return dict(
        voxels = list(itertools.chain.from_iterable([b['voxels'] for b in batch])),
        frag_ids = np.concatenate([b['frag_ids'] for b in batch]),
        shower_ids = np.concatenate([b['shower_ids'] for b in batch]),
        primary_ids = np.concatenate([b['primary_ids'] for b in batch]),
        batch = np.concatenate([i*np.ones(len(b['frag_ids']), dtype=np.int64) for i, b in enumerate(batch)])
        )
