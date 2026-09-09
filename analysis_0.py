#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:31:25 2026

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


# %% helper functions

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


def init_appendable_h5(config) -> None:
    """
    Initializes the HDF5 file with extendable datasets for cameras and tracks.

    Args:
        config: The configuration object.
    """
    n_tracks = 2
    n_bodyparts = len(config['bodyparts'])


    points_3d_path = Path(config['path']) / 'original_new.h5'
    with h5py.File(points_3d_path, 'w') as f:
        for track_idx in range(n_tracks):
            track_name = f'track{track_idx+1}'
            f.create_dataset(track_name,
                             shape=(0, n_bodyparts, 3),
                             maxshape=(None, n_bodyparts, 3),
                             chunks=(1, n_bodyparts, 3),
                             dtype='float32')

def load_results(project_dir=None, model_name=None, path=None):
    """Load the results from a modeled dataset.

    The results path can be specified directly via `path`. Otherwise it is
    assumed to be `{project_dir}/{model_name}/results.h5`.

    Parameters
    ----------
    project_dir: str, default=None
    model_name: str, default=None
    path: str, default=None

    Returns
    -------
    results: dict
        See :py:func:`keypoint_moseq.fitting.apply_model`
    """
    path = _get_path(project_dir, model_name, path, "results.h5")
    return load_hdf5(path)


def load_hdf5(filepath, datapath=None):
    """Load a dict of pytrees from an hdf5 file.

    Parameters
    ----------
    filepath: str
        Path of the hdf5 file to load.

    datapath: str, default=None
        Path within the hdf5 file to load the data from. If None, the data is
        loaded from the root of the hdf5 file.

    Returns
    -------
    save_dict: dict
        Dictionary where the values are pytrees, i.e. recursive collections of
        tuples, lists, dicts, and numpy arrays.
    """
    with h5py.File(filepath, "r") as f:
        if datapath is None:
            return {k: _loadtree_hdf5(f[k]) for k in f}
        else:
            return _loadtree_hdf5(f[datapath])

def _loadtree_hdf5(leaf):
    """Recursively load a pytree from an h5 file group."""
    if isinstance(leaf, h5py.Dataset):
        data = np.array(leaf[()])
        if h5py.check_dtype(vlen=data.dtype) == str:
            data = np.array([item.decode("utf-8") for item in data])
        elif data.dtype.kind == "S":
            data = data.item().decode("utf-8")
        elif data.shape == ():
            data = data.item()
        return data
    else:
        leaf_type = leaf.attrs["type"]
        values = map(_loadtree_hdf5, leaf.values())
        if leaf_type == "dict":
            return dict(zip(leaf.keys(), values))
        elif leaf_type == "list":
            return list(values)
        elif leaf_type == "tuple":
            return tuple(values)
        else:
            raise ValueError(f"Unrecognized type {leaf_type}")



def _get_path(project_dir, model_name, path, filename, pathname_for_error_msg="path"):
    # if path is None:
    #     assert project_dir is not None and model_name is not None, fill(
    #         f"`model_name` and `project_dir` are required if `{pathname_for_error_msg}` is None."
    #     )
    path = os.path.join(project_dir, model_name, filename)
    return path

# %% Init

project_dir = "/home/jlee629/kpmoseq/june_v4"
sleap_file = "original.h5"  # any .slp or .h5 file with predictions for a single video
# video_dir = os.path.join(project_dir, 'videos_raw')
# fpath = os.path.join(project_dir,sleap_file)



bodyparts_real = ['head', 'leftear', 'rightear', 'neck', 
             'leftelbow', 'rightelbow', 'lefthand', 'righthand', 
             'spinemid', 'tailbase', 'leftknee', 'rightknee', 
             'leftfoot', 'rightfoot', 'tailmid', 'tailend']
skeleton = [
    ['head','leftear'],
    ['head','rightear'],
    ['leftear','neck'],
    ['rightear','neck'],
    ['neck','spinemid'],
    ['spinemid','tailbase'],
    ['neck','leftelbow'],
    ['neck','rightelbow'],
    ['leftelbow','lefthand'],
    ['rightelbow','righthand'],
    ['tailbase','leftknee'],
    ['tailbase','rightknee'],
    ['leftknee','leftfoot'],
    ['rightknee','rightfoot'],
    ['tailbase','tailmid'],
    ['tailmid','tailend'],]

skeleton2 = [
    [0,1],
    [0,2],
    [1,3],
    [2,3],
    [3,8],
    [8,9],
    [3,4],
    [3,5],
    [4,6],
    [5,7],
    [9,10],
    [9,11],
    [10,12],
    [11,13],
    [9,14],
    [14,15],]
    

sk_color = [
    'red',
    'red',
    'red',
    'red',
    'green',
    'green',
    'blue',
    'blue',
    'blue',
    'blue',
    'black',
    'black',
    'black',
    'black',
    'black',
    'black',]


config = {}
config['path'] = project_dir
config['bodyparts'] = bodyparts_real

# init_appendable_h5(config)


# %% outlier removal and median smoothing

def pre_prc_opt(name,pdir,fname0,fname1):
    fpath0 = os.path.join(pdir,fname0)
    fpath1 = os.path.join(pdir,fname1)
    data_original, time_nan,cent3 = pre_prc1(name,fpath0)

    coordinates, confidences, bodyparts =marmopose_loader(fpath1,name)

    data = np.zeros_like(coordinates[name])
    tempdata = np.zeros_like(coordinates[name])
    tempdata[:] = coordinates[name][:]
    nan_ind = {}
    for b in np.arange(len(bodyparts_real)):
        nan_ind[b] = np.where(time_nan == 0)[0]        
        tempdata[nan_ind[b],b,:] = 4000

    # median window smoothing
    window = 3
    data = data[:-window+1,:,:]
    
    for b in np.arange(len(bodyparts_real)):
        for ax in np.arange(3):
            swindow = np.lib.stride_tricks.sliding_window_view(tempdata[:,b,ax], (window,))
            data[:,b,ax] = np.nanmedian(swindow,axis = 1)
            
    # removing outliers back to nan
    for b in np.arange(len(bodyparts_real)):
        data[(data[:,b,0]>3000),b,:] = np.nan
    
    data3, cent2 = pre_prc2(data)
    a = np.empty((1,16,3))
    a[:] = np.nan   
    data4 = np.append(np.append(a,data3,axis = 0),a,axis = 0)
    
    


    
    return data4, time_nan, cent2
    
    
def pre_prc1(name,fpath):
    # name = 'track1' 
    coordinates, confidences, bodyparts =marmopose_loader(fpath,name)
    
    tempdata = np.zeros_like(coordinates[name])
    tempdata[:] = coordinates[name][:]
    nan_ind = {}
    # tracking nan values
    for b in np.arange(len(bodyparts_real)):
        
        # if nan is nested between two values, fill that value with mean
        # only if, 2 values before and after are available
        # only if 1 or 2 consequtive nan values are nested
        for nb in np.where(np.isnan(coordinates[name][:,b,0]))[0]:
            if nb >3 and nb < (np.size(tempdata,axis = 0)-3):
                if not np.isnan(tempdata[nb-2:nb,b,0]).any() and not np.isnan(tempdata[nb+1:nb+3,b,0]).any():
                    tempdata[nb,b,:] = np.nanmean(tempdata[nb-2:nb+3,b,:],axis = 0)
        
    
    
    # Changing nan values to 4000
    for b in np.arange(len(bodyparts_real)):
        nan_ind[b] = np.where(np.isnan(tempdata[:,b,0]))[0]        
        tempdata[nan_ind[b],b,:] = 4000
        
    
    # median window smoothing
    window = 3
    data = np.zeros_like(tempdata)

    data = data[:-window+1,:,:]
    
    for b in np.arange(len(bodyparts_real)):
        for ax in np.arange(3):
            swindow = np.lib.stride_tricks.sliding_window_view(tempdata[:,b,ax], (window,))
            data[:,b,ax] = np.nanmedian(swindow,axis = 1)
            
    # removing outliers back to nan
    for b in np.arange(len(bodyparts_real)):
        data[(data[:,b,0]>3000),b,:] = np.nan
    
    
    # time segmenting
    window2 = 3
    temp = np.zeros(np.size(data,axis = 0))
    for t in np.arange(np.size(data,axis = 0)):
        if t < window2 or t> (np.size(data,axis = 0)-window2):  
            if np.sum(np.isnan(data[t,:14,0])) < 8: # discount for tail-mid and tail-end
                temp[t] = 1
        else:
            if np.sum(np.isnan(data[t-window2+1:t+1,:14,0]).sum(axis = 1) < 8) == window2 or np.sum(np.isnan(data[t:t+window,:14,0]).sum(axis = 1) < 8) == window2:
                temp[t] = 1
    
    
    data2 = data*temp[:,np.newaxis,np.newaxis]
    
    
    data2[data2==0] = np.nan
    # filling both edges with nan
    
    data3, cent = pre_prc2(data2)
    a = np.empty((1,16,3))
    a[:] = np.nan   
    data4 = np.append(np.append(a,data3,axis = 0),a,axis = 0)
    
    temp2 = np.append(np.append([0],temp,axis = 0),[0],axis = 0)
    
    return data4,temp2,cent

#### outlier removal 2 with distance from spinemid/centroid
def pre_prc2(data):
    nb_sk = np.size(data,axis =1)
    nb_fr = np.size(data,axis =0)
    data2 = data[:,:,:]
    tempdata = np.zeros_like(data)
    tempdata2 = np.zeros((nb_fr,nb_sk))
    cent = np.zeros((nb_fr,3))
    
    for t in np.arange(nb_fr):
        cent[t,:] = np.nanmedian(data[t,:,:],axis = 0)
        
        # if not np.isnan(data[t,8,:]).any():
        #     tempdata[t,:,:] = data[t,:,:] - data[t,8,:]
        # elif not np.isnan(data[t,4,:]).any() and not np.isnan(data[t,9,:]).any():
        #     tempdata[t,:,:] = data[t,:,:] - (data[t,4,:]+data[t,9,:])/2
        if not np.isnan(cent[t,:]).any():
            tempdata[t,:,:] = data[t,:,:] - cent[t,:]
        else: 
            tempdata[t,:,:] = np.nan
            data2[t,:,:] = np.nan
        
        for sk in np.arange(nb_sk):
            tempdata2[t,sk] = np.linalg.norm(tempdata[t,sk,:])
    
    # for each keypoint, check if distance between keypoint and centroid is within norm
    for sk in np.arange(nb_sk):
        # if not sk == 8:
        # sns.histplot(data = tempdata2[:,sk], binwidth = 2)
        if not sk == 8:
            thresh = np.nanpercentile(tempdata2[:,sk],95)
            data2[np.where(tempdata2[:,sk]>thresh)[0],sk,:] = np.nan
    

    # cent = np.append(np.append(cent[0:2,:],cent, axis = 0),cent[-3:-1,:],axis =0)

    window2 = 5
    cent2 = np.zeros_like(cent)
    cent2 = cent2[:-window2+1,:]
    
    for ax in np.arange(3):
        swindow = np.lib.stride_tricks.sliding_window_view(cent[:,ax], (window2,))
        cent2[:,ax] = np.nanmedian(swindow,axis = 1)

    cent2 = np.append(np.append(cent2[0:2,:],cent2, axis = 0),cent2[-3:-1,:],axis =0)
    
    
    return data2,cent2


    
        
        

# %%

file_dir = '/media/jlee629/D/MarmoPose/projects'
project_name = 'pair-test-june26_v2'
fpath = os.path.join(file_dir,project_name,'points_3d')


# fpath_ori = os.path.join(fpath,'original.h5')
# fpath_opt = os.path.join(fpath,'optimized.h5')

D_track1, time1, cent1 = pre_prc_opt('track1',fpath,'original.h5','optimized.h5')
D_track2, time2, cent2 = pre_prc_opt('track2',fpath,'original.h5','optimized.h5')

D_all = np.concatenate((D_track1,D_track2),axis = 0)

# C_all = np.concatenate((C_1,C_2),axis = 0)
save_points_3d_h5(D_all,'track1',Path(project_dir) / 'original_new.h5')


# D_track_opt, time_opt = pre_prc_opt('track1',project_dir,'original1.h5','optimized1.h5')



# fpath1 = os.path.join(project_dir,'original1.h5')
# fpath2 = os.path.join(project_dir,'original3.h5')
# # fpath3 = os.path.join(project_dir,'optimized3.h5')


# D_track1,time1,cent1 = pre_prc1('track1', fpath1)
# D_track2,time2,cent2 = pre_prc1('track2', fpath1)

# D_track3,time3,cent3 = pre_prc1('track1', fpath2)
# D_track4,time4,cent4 = pre_prc1('track2', fpath2)

# # # D_track1_opt,time5 = pre_prc1('track1', fpath3)


D_track1, time1, cent1 = pre_prc_opt('track1',fpath,'original.h5','optimized.h5')
D_track2, time2, cent2 = pre_prc_opt('track2',fpath,'original.h5','optimized.h5')

D_track5, time5, cent5 = pre_prc_opt('track1',project_dir,'original2.h5','optimized2.h5')
D_track6, time6, cent6 = pre_prc_opt('track2',project_dir,'original2.h5','optimized2.h5')

D_track3, time3, cent3 = pre_prc_opt('track1',project_dir,'original3.h5','optimized3.h5')
D_track4, time4, cent4 = pre_prc_opt('track2',project_dir,'original3.h5','optimized3.h5')





# %% 
# name = 'track1'
# coord, confidences, bodyparts =marmopose_loader(fpath3,name)


# D_track1_opt2 = D_track1_opt*time3[:,np.newaxis,np.newaxis] 

# D_track1_opt2[D_track1_opt2 == 0]  = np.nan

for sc in [8]:
    fig, axs = plt.subplots(3, 1)
    # sc = 3
    for i in [0,1,2]:
        # axs[i].plot(coord[name][:,sc,i],color = 'blue')
        # axs[i].plot(D_track_opt[:,sc,i],'green')
        axs[i].plot(D_track1[:,sc,i],'red')
        axs[i].plot(cent1[:,i],'blue')
        # axs[i].plot(time3*800,'black',marker='.', linestyle='None')
        axs[i].set_ylim([-300,700])
    
    plt.show




# sns.histplot(data = data3[:,sc,:], binwidth = 200)


# D_track1 = D_track1[:14150,:,:]
# D_track2 = D_track2[:14150,:,:]



# %% save to h5

D_1 = np.concatenate((D_track1,D_track3,D_track5),axis = 0)
D_2 = np.concatenate((D_track2,D_track4,D_track6),axis = 0)



C_1 = np.concatenate((cent1,cent3,cent5),axis = 0)
C_2 = np.concatenate((cent2,cent4,cent6),axis = 0)
# C_3 = np.append(cent5,cent6,axis = 0)

init_appendable_h5(config)

D_all = np.concatenate((D_1,D_2),axis = 0)

C_all = np.concatenate((C_1,C_2),axis = 0)
# save_points_3d_h5(D_all,'track1',Path(project_dir) / 'original_new.h5')
save_points_3d_h5(np.concatenate((D_track1,D_track2),axis = 0),'track1',Path(project_dir) / 'original_new.h5')



# indices for animals and videos
ani_ind = {}
ani_ind['not_nan'] = np.where(~(np.isnan(C_all).all(axis = 1)))[0]
ani_ind['nan'] = np.where((np.isnan(C_all).all(axis = 1)))[0]

ani_ind['animal_id'] = np.hstack((np.zeros((1,(np.size(D_1,axis = 0)))),
                                  np.ones((1,(np.size(D_2,axis = 0))))))[0]
ani_ind['vid_id'] = np.hstack((np.zeros((1,(np.size(D_track1,axis = 0)))),
                               np.ones((1,(np.size(D_track3,axis = 0)))),
                               2*np.ones((1,(np.size(D_track5,axis = 0))))))[0]
ani_ind['vid_id'] = np.hstack((ani_ind['vid_id'],ani_ind['vid_id']))


# %% load from h5

project_dir2 = "/home/jlee629/kpmoseq/june_v4"
fname = 'original_new.h5'
fpath1 = os.path.join(project_dir2,fname)
coordinates, confidences, bodyparts =marmopose_loader(fpath1,'track1')

D_all = coordinates['track1']
D_all2, C_all2 = pre_prc2(D_all)


# %% indices for animals and videos 

video_dict = {}
video_dict[0] = {}
video_dict[0]['name'] = 'date'
video_dict[0]['length'] = 37500

video_dict[1] = {}
video_dict[1]['name'] = 'date'
video_dict[1]['length'] = 60000

ani_ind = {}
ani_ind['not_nan'] = np.where(~(np.isnan(C_all2).all(axis = 1)))[0]
ani_ind['nan'] = np.where((np.isnan(C_all2).all(axis = 1)))[0]

ani_ind['animal_id'] = []
ani_ind['vid_id'] = []
for vid in np.arange(len(video_dict)):
    temp = np.hstack((np.zeros((1,video_dict[vid]['length'])),
                                      np.ones((1,video_dict[vid]['length']))))[0]
    
    ani_ind['animal_id'] = np.hstack((ani_ind['animal_id'],temp))
    ani_ind['vid_id'] = np.hstack((ani_ind['vid_id'],np.ones((1,len(temp)))[0]*vid))
    
    
                     

    
# %% load keypoint_moseq syllables

project_dir2 = "/home/jlee629/kpmoseq/june_v4"
model_name = '2026_07_17-11_58_04'

results = load_results(project_dir2, model_name)


# fig, axs = plt.subplots(4, 1)
#     # sc = 3
# for i in [0,1,2]:
#         # axs[i].plot(coord[name][:,sc,i],color = 'blue')
#         # axs[i].plot(D_track_opt[:,sc,i],'green')
#         axs[i].plot(results['track1']['centroid'][:,i],'red')
#         axs[i].plot(C_3[:,i],'blue')
#         # axs[i].plot(time3*800,'black',marker='.', linestyle='None')
#         axs[i].set_ylim([-300,700])
# axs[3].plot(results['track1']['syllable'][:],marker='.', linestyle='None')
    
# plt.show

# %% derivative of centroid position

C_all = C_all2
# C_k = results['track1']['centroid']
S = np.zeros((len(C_all),1))
for t in np.arange(1,len(C_all)-1):
    S[t] = np.linalg.norm(C_all[t+1,:]-C_all[t-1,:])/2




# %% transforming coordinates normalizing pose to center and orientation
# we take neck and tailbase(sometimes tailbase invisible)

def transform_coord(data3):
    T = np.zeros((np.size(data3,axis=0),14*3+1))
    for t in np.arange(np.size(data3,axis=0)):
        if not np.isnan(data3[t,3,:]).any() and not np.isnan(data3[t,9,:]).any():
            l_axis = np.linalg.norm((data3[t,3,:]-data3[t,9,:]))
            u = (data3[t,3,:]-data3[t,9,:])/l_axis
            r = data3[t,0:14,:]-data3[t,9,:] # not taking into account tail # remove head and ear as well
            s = (r @ u)/l_axis
            d = np.linalg.norm(r - np.outer(s,u), axis = 1)/l_axis
            z = (data3[t,0:14,2]-data3[t,9,2])/np.abs(data3[t,3,2]-data3[t,9,2])
            
            T[t,:-1] = np.hstack((s,d,z))
            T[t,-1] = S[t,0]
        else:
            T[t,:] = np.nan
    return T

# T1 = transform_coord(D_1)
# T2 = transform_coord(D_2)

# T  = np.vstack((T1,T2))

T = transform_coord(D_all2)
    
# T = np.hstack((T3,S))




# %% UMAP

ani_ind2 = {}
ani_ind2['not_nan'] = np.where(~(np.isnan(T).all(axis = 1)))[0]
ani_ind2['nan'] = np.where((np.isnan(T).all(axis = 1)))[0]
# ani_ind2['animal_id'] = np.hstack((np.ones((1,(np.size(D_1,axis = 0)))),np.ones((1,(np.size(D_2,axis = 0))))*2))[0]

T_new = T[ani_ind2['not_nan']]


# T_new2 = np.where(np.isnan(T_new),10000,T_new)

reducer = umap.UMAP(n_neighbors=75,n_components=3,min_dist = 0.1)

embedding = reducer.fit_transform(T_new,force_all_finite="allow-nan")
embedding.shape

# from umap.parametric_umap import ParametricUMAP
# embedder = ParametricUMAP()
# embedding = embedder.fit_transform(T_new2)


fig = plt.figure(figsize=(8, 6))
# Add 3D axes
# ax = fig.add_subplot(111, projection='3d')
plt.scatter(embedding[:, 0],embedding[:, 1])
# ax.scatter(embedding[:, 0],embedding[:, 1],embedding[:, 2])

# %% segment data into animal and video
vid_ind = ['vid_1','vid_2','vid_3']
animal_ind = ['animal_1','animal_2']

# emb2 = np.zeros((len(ani_ind['animal_id']),3))
emb2 = np.empty((len(ani_ind['animal_id']),3)) * np.nan
# emb2[ani_ind2['not_nan'],:] = embedding[:]


data_all = {}
for j in [0,1]:
    data_all[vid_ind[j]] = {}
    for i in [0,1]:
        data_all[vid_ind[j]][animal_ind[i]] = {}
        
        idx = ((ani_ind['animal_id'] == i) & (ani_ind['vid_id']==j))
        
        data_all[vid_ind[j]][animal_ind[i]]['keypoints'] = D_all[idx,:,:]
        data_all[vid_ind[j]][animal_ind[i]]['centroid'] = results['track1']['centroid'][idx,:]
        data_all[vid_ind[j]][animal_ind[i]]['syllables'] = results['track1']['syllable'][idx]
        data_all[vid_ind[j]][animal_ind[i]]['speed'] = S[idx]
        data_all[vid_ind[j]][animal_ind[i]]['UMAP'] = emb2[idx] 



# %% plot and view data (no Umap)


import matplotlib.gridspec as gridspec

vidx = 'vid_1'
anid  ='animal_2'
anid2 = 'animal_1'
# centering data around spine mid
# add "if spinemid is nan, then ...
newdata2 = data_all[vidx][anid2]['keypoints']
newdata = data_all[vidx][anid]['keypoints']
emb3 = data_all[vidx][anid]['UMAP']

xtime = np.arange(len(newdata))*0.04
# for t in np.arange(np.size(newdata,axis = 0)):
#     for sk in np.arange(len(skeleton2)):
#         newdata[t,sk,:] = D_3[t,sk,:] # - D_1[t,8,:]

# newdata = newdata[ani_ind['not_nan'],:,:]

# --- 2. Set up the Figure and 3D Axes ---
fig = plt.figure(figsize=(16, 12))
# Add 3D axes

gs = gridspec.GridSpec(4, 3, figure=fig)

ax = fig.add_subplot(gs[:, 0:2], projection='3d')

axs = {}

axs[0] = fig.add_subplot(gs[0:2,2], projection='3d')

for f in np.arange(2,4):
    axs[f] = fig.add_subplot(gs[f, 2])  # row f, column 2



fig.subplots_adjust(bottom=0.25) # Adjust subplot to make room for the slider
t = 0

ax_l = 450

ax.set_xlim([-150, ax_l])
ax.set_ylim([-150, ax_l])
ax.set_zlim([-300, 500])

# for f in np.arange(3):
#     axs[f].plot(data_all[vidx]['animal_2']['centroid'][:,f])
#     axs[f].set_xlim([-50,50])
    

# sc = axs[0].scatter(emb2[:, 0],emb2[:, 1],c = 'b',alpha = 0.01)

# red_dot, = axs[0].plot(emb3[0, 0], emb3[0, 1], 'ro', markersize=10, zorder=5)


axs[2].plot(xtime,np.convolve(data_all[vidx][anid]['speed'][:,0],10,mode = 'same')*0.04)
axs[2].set_xlim([-50*.04,50*.04])
axs[2].set_ylabel('velocity cm/s')
axs[2].set_ylim([0,150])
axs[3].plot(xtime,data_all[vidx][anid]['syllables'][:])
axs[3].set_xlim([-50*.04,50*.04])
axs[3].set_ylim([0,25])
axs[2].set_title('time (s)')
axs[3].set_ylabel('syllable identity')


red_line = axs[3].axvline(x=t, ymin=0, ymax=1,color = 'red')
lines = {}
lines2 = {}

# plot skeleton
# for sk,skc in zip(skeleton2,sk_color):
for sk in np.arange(len(skeleton2)):
    lines[sk] = ax.plot3D(newdata[t,skeleton2[sk],0], newdata[t,skeleton2[sk],1], -newdata[t,skeleton2[sk],2], sk_color[sk])
    lines2[sk] = ax.plot3D(newdata2[t,skeleton2[sk],0], newdata2[t,skeleton2[sk],1], -newdata2[t,skeleton2[sk],2], sk_color[sk],alpha = 0.2)
  

# ---2.1 Plot centroids and syllables

samp_t = random.sample(range(0,len(newdata)),5000)
axs[0].scatter3D(data_all[vidx][anid]['centroid'][samp_t,0],
             data_all[vidx][anid]['centroid'][samp_t,1],
             data_all[vidx][anid]['centroid'][samp_t,2]) 

axs[0].scatter3D(data_all[vidx][anid2]['centroid'][samp_t,0],
             data_all[vidx][anid2]['centroid'][samp_t,1],
             data_all[vidx][anid2]['centroid'][samp_t,2],alpha = 0.2) 

# red_scatter, = ax.plot3D(data_all[vidx][anid]['centroid'][0,0],
#                           data_all[vidx][anid]['centroid'][0,1],
#                           data_all[vidx][anid]['centroid'][0,2],'ro', markersize=10, zorder=5)
 

# --- 3. Create the Slider Widget ---
ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03]) # [left, bottom, width, height]
time_slider = Slider(
    ax=ax_slider,
    label='Time Step',
    valmin=0,
    valmax=np.size(newdata,axis = 0) - 1,   
    valinit=0,
    valstep=1.0 # Ensures the slider snaps to integer time steps
)


# Create an array of 5 blue pixels
c_array = np.tile(np.array([0.0, 0.0, 1.0, 0.01], dtype=np.float32), (np.size(newdata, axis=0), 1))

# --- 4. Define the Update Function and Connect to Slider ---
def update(val):
    """Callback function to update the plot data based on slider value."""
    t = int(time_slider.val)
    # Update the data of the 3D plot artist
    for sk in np.arange(len(skeleton2)):
        lines[sk][0].set_data_3d(newdata[t,skeleton2[sk],0], newdata[t,skeleton2[sk],1], -newdata[t,skeleton2[sk],2])
        lines2[sk][0].set_data_3d(newdata2[t,skeleton2[sk],0], newdata2[t,skeleton2[sk],1], -newdata2[t,skeleton2[sk],2])

        # lines[sk].set_color(sk_color[sk])
    
    # update centroid
    # red_scatter.set_data_3d([data_all[vidx][anid]['centroid'][t,0]],
    #                         [data_all[vidx][anid]['centroid'][t,1]],
    #                         [data_all[vidx][anid]['centroid'][t,2]])
    
    # update subplot panels    
    for f in np.arange(2,4):
        axs[f].set_xlim([(-50+t)*.04,(50+t)*.04])
        
    red_line.set_xdata([t*0.04])
    # Update the scatter
    # red_dot.set_data([emb3[t, 0]], [emb3[t, 1]])
    
    # update xlabel for syllable names 
    if data_all[vidx][anid]['syllables'][t] == 4:
        axs[3].set_xlabel('walking')
    else:
        axs[3].set_xlabel(str(data_all[vidx][anid]['syllables'][t]))
    # Redraw the figure
    
    
    fig.canvas.draw_idle()
    
# Register the update function with the slider's on_changed event
time_slider.on_changed(update)

def on_key(event):
    step = 10 if event.key in ('up', 'down') else 1
    if event.key in ('right', 'up'):
        time_slider.set_val(min(time_slider.val + step, time_slider.valmax))
    elif event.key in ('left', 'down'):
        time_slider.set_val(max(time_slider.val - step, time_slider.valmin))

fig.canvas.mpl_connect('key_press_event', on_key)


# --- 5. Display the Plot ---
plt.show()


# %%

# # %% plot and view data 


# # making new_embedding based on animal and session
# # emb2 = np.zeros((len(ani_ind['animal_id']),2))
# # emb2[ani_ind['not_nan'],:] = embedding[:,0:2]

# # emb2 = emb2[np.where(ani_ind['animal_id'] ==2)[0],:]
# emb2 = embedding[:,[0,1]]

# # centering data around spine mid
# # add "if spinemid is nan, then ...
# newdata = np.zeros_like(D_3)
# for t in np.arange(np.size(newdata,axis = 0)):
#     for sk in np.arange(len(skeleton2)):
#         newdata[t,sk,:] = D_3[t,sk,:] # - D_1[t,8,:]

# newdata = newdata[ani_ind['not_nan'],:,:]

# # --- 2. Set up the Figure and 3D Axes ---
# fig = plt.figure(figsize=(8, 6))
# # Add 3D axes
# ax = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122)
# fig.subplots_adjust(bottom=0.25) # Adjust subplot to make room for the slider
# t = 0

# ax_l = 500

# ax.set_xlim([-ax_l, ax_l])
# ax.set_ylim([-ax_l, ax_l])
# ax.set_zlim([-ax_l, ax_l])

# lines = {}
# # for sk,skc in zip(skeleton2,sk_color):
# for sk in np.arange(len(skeleton2)):
#     lines[sk] = ax.plot3D(newdata[t,skeleton2[sk],0], newdata[t,skeleton2[sk],1], -newdata[t,skeleton2[sk],2], sk_color[sk])
    
    
    
# sc = ax2.scatter(emb2[:, 0],emb2[:, 1],c = 'b',alpha = 0.01)
# red_dot = ax2.plot(emb2[0, 0], emb2[0, 1], 'ro', markersize=10, zorder=5)
# # create figure for scatter. 
    

# # --- 3. Create the Slider Widget ---
# ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03]) # [left, bottom, width, height]
# time_slider = Slider(
#     ax=ax_slider,
#     label='Time Step',
#     valmin=0,
#     valmax=np.size(newdata,axis = 0) - 1,
#     valinit=0,
#     valstep=1.0 # Ensures the slider snaps to integer time steps
# )


# # Create an array of 5 blue pixels
# c_array = np.tile(np.array([0.0, 0.0, 1.0, 0.01], dtype=np.float32), (np.size(newdata, axis=0), 1))

# # --- 4. Define the Update Function and Connect to Slider ---
# def update(val):
#     """Callback function to update the plot data based on slider value."""
#     t = int(time_slider.val)
#     # Update the data of the 3D plot artist
#     for sk in np.arange(len(skeleton2)):
#         lines[sk][0].set_data_3d(newdata[t,skeleton2[sk],0], newdata[t,skeleton2[sk],1], -newdata[t,skeleton2[sk],2])
#         # lines[sk].set_color(sk_color[sk])
        
#     # Update the scatter
#     red_dot.set_data([emb2[t, 0]], [emb2[t, 1]])
#     # Redraw the figure
#     fig.canvas.draw_idle()

# # Register the update function with the slider's on_changed event
# time_slider.on_changed(update)

# def on_key(event):
#     step = 10 if event.key in ('up', 'down') else 1
#     if event.key in ('right', 'up'):
#         time_slider.set_val(min(time_slider.val + step, time_slider.valmax))
#     elif event.key in ('left', 'down'):
#         time_slider.set_val(max(time_slider.val - step, time_slider.valmin))

# fig.canvas.mpl_connect('key_press_event', on_key)


# # --- 5. Display the Plot ---
# plt.show()

# # # Plot the initial data (e.g., just the first point)
# # # The `plot` object must be an artist that can be updated later
# # # We use ax.plot() for a line/scatter plot. `lines` will be a list of artists.
# # lines = ax.plot(data[:1, 0], data[:1, 1], data[:1, 2], marker='o') 
# # line = lines[0] # Get the specific line artist to update

# # # Set initial axis limits
# # ax.set_xlim([min(x_data), max(x_data)])
# # ax.set_ylim([min(y_data), max(y_data)])
# # ax.set_zlim([min(z_data), max(z_data)])
# # ax.set_xlabel('X Coordinate')
# # ax.set_ylabel('Y Coordinate')
# # ax.set_zlabel('Z Coordinate')
# # ax.set_title('3D Plot with Time Slider')



# # %% 

# import matplotlib.animation as animation
# from matplotlib.animation import FFMpegWriter # or PillowWriter for .gif

# # %% plot and view data 
# emb2 = embedding[:,0:2]

# # centering data around spine mid
# newdata = np.zeros_like(D_3)
# for t in np.arange(np.size(newdata, axis=0)):
#     for sk in np.arange(len(skeleton2)):
#         newdata[t, sk, :] = D_3[t, sk, :]
# newdata = newdata[ani_ind['not_nan'], :, :]

# # --- 2. Set up the Figure and 3D Axes ---
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122)

# ax_l = 500
# ax.set_xlim([-ax_l, ax_l])
# ax.set_ylim([-ax_l, ax_l])
# ax.set_zlim([-ax_l, ax_l])

# lines = {}
# for sk in np.arange(len(skeleton2)):
#     lines[sk] = ax.plot3D(
#         newdata[0, skeleton2[sk], 0],
#         newdata[0, skeleton2[sk], 1],
#         -newdata[0, skeleton2[sk], 2],
#         sk_color[sk]
#     )

# sc = ax2.scatter(embedding[:, 0], embedding[:, 1], c='b', alpha=0.01)
# red_dot, = ax2.plot(emb2[0, 0], emb2[0, 1], 'ro', markersize=10, zorder=5)

# # Add a time label (optional but useful in videos)
# time_text = ax.text2D(0.05, 0.95, 'Frame: 0', transform=ax.transAxes)

# # --- 3. Define the animation update function ---
# def animate(t):
#     """Update function called for each frame."""
#     # Update 3D skeleton lines
#     for sk in np.arange(len(skeleton2)):
#         lines[sk][0].set_data_3d(
#             newdata[t, skeleton2[sk], 0],
#             newdata[t, skeleton2[sk], 1],
#             -newdata[t, skeleton2[sk], 2]
#         )
    
#     # Update the red dot in the embedding scatter
#     red_dot.set_data([emb2[t, 0]], [emb2[t, 1]])
    
#     # Update frame label
#     time_text.set_text(f'Frame: {t}')
    
#     # Return all updated artists
#     return [lines[sk][0] for sk in np.arange(len(skeleton2))] + [red_dot, time_text]

# # --- 4. Create the FuncAnimation object ---
# n_frames = np.size(newdata, axis=0)

# ani = animation.FuncAnimation(
#     fig,
#     animate,
#     frames=n_frames,
#     interval=50,       # milliseconds between frames (~20 fps)
#     blit=False,        # blit=False is safer with 3D axes
#     repeat=False
# )

# # --- 5. Save the video ---
# # Option A: Save as .mp4 (requires ffmpeg installed)
# writer_mp4 = FFMpegWriter(fps=20, metadata=dict(title='Skeleton Animation'), bitrate=1800)
# ani.save('skeleton_animation.mp4', writer=writer_mp4)
# print("Saved skeleton_animation.mp4")

# # Option B: Save as .gif (requires Pillow installed: pip install Pillow)
# # writer_gif = animation.PillowWriter(fps=20)
# # ani.save('skeleton_animation.gif', writer=writer_gif)
# # print("Saved skeleton_animation.gif")

# plt.show()


# # %% #### Old tests with UMAP ####


# # %% PCA with NIPALS
# T_new2 = np.where(np.isnan(T_new),10000,T_new)


# # pc = PCA(T_new2, method='nipals')



# # %%


# fig = plt.figure(figsize=(8, 6))
# # Add 3D axes
# # ax = fig.add_subplot(111, projection='3d')
# plt.scatter(embedding[:, 0],embedding[:, 1])
# # ax.scatter(embedding[:, 0],embedding[:, 1],embedding[:, 2])
# # plt.gca().set_aspect('equal', 'datalim')
# # plt.title('UMAP projection of the Penguin dataset', fontsize=24);



