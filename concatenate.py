#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:26:02 2026

@author: jlee629
"""




import matplotlib.pyplot as plt
import numpy as np
# if not hasattr(np, 'bool8'):
#     np.bool8 = np.bool_ # Or np.bool
import os
import h5py
import logging
from pathlib import Path
# import seaborn as sns
# import umap
import random
# from statsmodels.multivariate.pca import PCA
# import keypoint_moseq as kpms
from matplotlib.widgets import Slider
# from mpl_toolkits.mplot3d import Axes3D


logger = logging.getLogger(__name__)


# %%

def marmopose_loader(filepath,track_name):
    """Load keypoints from sleap-anipose hdf5 files."""
    with h5py.File(filepath, "r") as f:
        coords = f[track_name][()]
        if "point_scores" in f.keys():
            confs = f["point_scores"][()]
        else:
            confs = np.ones_like(coords[..., 0])
        bodyparts = ["bodypart{}".format(i) for i in range(coords.shape[1])]
        # if coords.shape[1] == 1:
        coordinates = {track_name: coords}
        confidences = {track_name: confs}
        # else:
        #     coordinates = {
        #         f"{name}_track{i}": coords[:, i] for i in range(coords.shape[1])
        #     }
        #     confidences = {
        #         f"{name}_track{i}": confs[:, i] for i in range(coords.shape[1])
        #     }
    return coordinates, confidences, bodyparts

def load_points_3d_h5(file_path: Path) -> np.ndarray:
    """
    Load 3D points from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file.

    Returns:
        Array of 3D points, sorted by track name.
            - Shape: (n_tracks, n_frames, n_bodyparts, 3)
            - Final channel: (x, y, z)
    """
    all_points_3d = []
    with h5py.File(file_path, 'r') as f:
        keys = sorted(list(f.keys()))
        for name in keys:
            points = f[name][:]
            all_points_3d.append(points)
            
    all_points_3d = np.array(all_points_3d)
    
    logger.info(f'Loaded 3D points from {file_path} with order: {keys}')
    return all_points_3d

def save_points_3d_h5(points: np.ndarray, name: str, file_path: Path) -> None:
    """
    Saves 3D points for a track to an HDF5 file.

    Args:
        points: The 3D points to save. Shape of (n_frames, n_bodyparts, 3), final channel (x, y, z).
        name: The name of the track.
        file_path: The path to the HDF5 file.
    """
    with h5py.File(file_path, 'a') as f:
        if name in f:
            del f[name]
            logger.info(f'Overwriting existing {name} in {file_path}')
        f.create_dataset(name, data=points)

    logger.info(f'Saving 3D points for {name} in {file_path}')


def init_appendable_h5(config) -> None:
    """
    Initializes the HDF5 file with extendable datasets for cameras and tracks.

    Args:
        config: The configuration object.
    """
    n_tracks = 2
    n_bodyparts = len(config['bodyparts'])


    points_3d_path = Path(config['path']) / 'original_combined.h5'
    with h5py.File(points_3d_path, 'w') as f:
        for track_idx in range(n_tracks):
            track_name = f'track{track_idx+1}'
            f.create_dataset(track_name,
                             shape=(0, n_bodyparts, 3),
                             maxshape=(None, n_bodyparts, 3),
                             chunks=(1, n_bodyparts, 3),
                             dtype='float32')


# %%

bodyparts_real = ['head', 'leftear', 'rightear', 'neck', 
             'leftelbow', 'rightelbow', 'lefthand', 'righthand', 
             'spinemid', 'tailbase', 'leftknee', 'rightknee', 
             'leftfoot', 'rightfoot', 'tailmid', 'tailend']

project_dir = "/home/jlee629/kpmoseq"

config = {}
config['path'] = project_dir
config['bodyparts'] = bodyparts_real

init_appendable_h5(config)




# %%


fname1 = 'june_v2/original_new.h5'

fpath1 = os.path.join(project_dir,fname1)
coordinates, confidences, bodyparts =marmopose_loader(fpath1,'track1')


D1 = coordinates['track1']

fname2 = 'test_9/original_new.h5'

fpath2 = os.path.join(project_dir,fname2)
coordinates2, confidences, bodyparts =marmopose_loader(fpath2,'track1')

D2 = coordinates2['track1']

save_points_3d_h5(np.concatenate((D1,D2),axis = 0),'track1',Path(project_dir) / 'original_combined.h5')










