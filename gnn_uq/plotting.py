import torch
import torch.nn as nn

import numpy as np
import os
from tqdm import tqdm
import h5py

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from voxelUtils import *

# from torch_geometric.loader import DataLoader as GraphDataLoader

import matplotlib.pyplot as plt

SLACred = '#8C1515'
SLACgrey = '#53565A'
SLACblue = '#007C92'
SLACteal = '#279989'
SLACgreen = '#8BC751'
SLACyellow = '#FEDD5C'
SLACorange = '#E04F39'
SLACpurple = '#53284F'
SLAClavender = '#765E99'
SLACbrown = '#5F574F'

SLACcolors = [SLACred,
              SLACblue,
              SLACteal,
              SLACgreen,
              SLACyellow,
              SLACgrey,
              SLACorange,
              SLACpurple,
              SLAClavender,
              SLACbrown,
]

def plot_angle(voxels, colors, azimuth):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.view_init(elev = 30,
                 azim = azimuth)
    
    for fragment, color in zip(voxels, colors):
        ax.scatter(fragment[:,0],
                   fragment[:,1],
                   fragment[:,2],
                   color = color,
                   )

    # plt.savefig('frag.png')

def make_rotation_series(voxels, colors):
    angleSpace = np.linspace(0, 360, 145)[:-1]
    for i, angle in enumerate(angleSpace):
        plot_angle(voxels, colors, angle)
        plt.savefig('plot_frames/'+str(i)+'.png')
        plt.clf()

def main(args):
    data = ShowerVoxels(args.data)

    event_ind = 6

    print (data[event_ind].keys())
    print (data[event_ind]['frag_ids'])
    print (data[event_ind]['shower_ids'])
    print (data[event_ind]['primary_ids'])

    voxels = data[event_ind]['voxels']
    frag_ids = data[event_ind]['frag_ids']
    shower_ids = data[event_ind]['shower_ids']
    primary_ids = data[event_ind]['primary_ids']

    # make three plot series with different color schemes:

    # # by fragments
    # colors = SLACcolors*3

    # primary and secondary
    colors = [SLACred if p_id else SLACblue for p_id in primary_ids]

    # # colored by shower
    # colors = []
    # for shower_id in shower_ids:
    #     for i, unq_shower_id in enumerate(np.unique(shower_ids)):
    #         if shower_id == unq_shower_id:
    #             colors.append(SLACcolors[i])
            
    make_rotation_series(voxels, colors)

    plt.show()
    
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str,
                        default = '/home/dan/studies/GNN_uq/data/if-graph-test.h5',
                        help = "input data (hdf5)")

    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    
    args = parser.parse_args()

    main(args)

