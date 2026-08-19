#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:13:30 2026

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
import umap
import random
# from statsmodels.multivariate.pca import PCA
# import keypoint_moseq as kpms
from matplotlib.widgets import Slider
# from mpl_toolkits.mplot3d import Axes3D

import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import pandas as pd
from datashader.mpl_ext import dsshow



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
project_dir = "/home/jlee629/kpmoseq/projects/feb_may"
fname = 'combined.h5'



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

# %% load from h5


fpath1 = os.path.join(project_dir,fname)
coordinates, confidences, bodyparts =marmopose_loader(fpath1,'track_all')


index = np.load(os.path.join(project_dir,'combined_index.npy'), allow_pickle=True)
combined_arr = coordinates['track_all']

# Get track1 data from session 0
entry = index[0]
start, end = entry['track1']['start'], entry['track1']['end']
track1_data = combined_arr[start:end+1]

# # Print summary
# for entry in index:
#     print(entry['session_dir'])
#     print(f"  track1: {entry['track1']['start']} -> {entry['track1']['end']}")
#     print(f"  track2: {entry['track2']['start']} -> {entry['track2']['end']}")

coordinates, _, _ =marmopose_loader(os.path.join(project_dir,'combined_centroid.h5'),'centroid_all')
cent_all = coordinates['centroid_all']

# %% derivative of centroid position

# C_k = results['track1']['centroid']
S = np.zeros((len(cent_all),1))
for t in np.arange(1,len(cent_all)-1):
    S[t] = np.linalg.norm(cent_all[t+1,:]-cent_all[t-1,:])/2


# %% load keypoint syllables

model_name = '2026_08_19-10_33_58'

results = load_results(project_dir, model_name)
syllables_raw = np.concatenate([results[k]['syllable'] for k in sorted(results.keys())])




# %% transforming coordinates normalizing pose to center and orientation
# we take neck and tailbase(sometimes tailbase invisible)

ani_ind2 = {}
ani_ind2['nan'] = []
ani_ind2['not-nan'] = []


nan_mask = np.isnan(combined_arr).any(axis=(1, 2))

ani_ind2 = {
    'nan':     np.where(nan_mask)[0],
    'not-nan': np.where(~nan_mask)[0]
}
        
#%%


def transform_coord(data,speed):
    data3 = data[ani_ind2['not-nan'],:,:]
    S = speed[ani_ind2['not-nan'],:]
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

T = transform_coord(combined_arr,S)

# %%

reducer = umap.UMAP(n_neighbors=50,n_components=3,min_dist = 0.1)

embedding = reducer.fit_transform(T,force_all_finite="allow-nan")
embedding.shape



# %%
# Align syllables to not-nan frames used in transform_coord
# T has shape (len(not-nan frames), n_features)
not_nan_idx = ani_ind2['not-nan']
syllables_aligned = syllables_raw[not_nan_idx]

# Remove any remaining NaN rows from T (frames where neck/tailbase was NaN)
valid_mask = ~np.isnan(T).any(axis=1)
embedding_valid = embedding[valid_mask]
syllables_valid = syllables_aligned[valid_mask].astype(int)

# Build dataframe for datashader
n_syllables = int(syllables_valid.max()) + 1

MAX_SYLLABLE = 25
syllables_capped = np.where(syllables_valid <= MAX_SYLLABLE, 
                             syllables_valid.astype(str), 
                             'other')

categories = [str(i) for i in range(MAX_SYLLABLE + 1)] + ['other']

df = pd.DataFrame({
    'x': embedding_valid[:, 0],
    'y': embedding_valid[:, 1],
    'z': embedding_valid[:, 2],
    'syllable': pd.Categorical(syllables_capped, categories=categories)
})


color_key = {str(i): cc.glasbey[i] for i in range(MAX_SYLLABLE + 1)}
color_key['other'] = '#aaaaaa'


# --- 2D projections colored by syllable ---
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# projections = [('x', 'y', 'XY'), ('x', 'z', 'XZ'), ('y', 'z', 'YZ')]

# for ax, (dim1, dim2, title) in zip(axes, projections):
#     dsshow(
#         df,
#         ds.Point(dim1, dim2),
#         ds.count_cat('syllable'),
#         color_key=color_key,
#         ax=ax,
#         aspect='auto'
#     )
#     ax.set_title(f'UMAP {title} — colored by syllable')
#     ax.set_xlabel(dim1)
#     ax.set_ylabel(dim2)

# plt.tight_layout()
# plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
projections = [('x', 'y', 'XY'), ('x', 'z', 'XZ'), ('y', 'z', 'YZ')]



# import matplotlib.image as mpimg
# from io import BytesIO
# from PIL import Image
from matplotlib.patches import Patch



for ax, (dim1, dim2, title) in zip(axes, projections):
    canvas = ds.Canvas(plot_width=800, plot_height=800)
    agg = canvas.points(df, dim1, dim2, ds.count_cat('syllable'))
    img = tf.spread(tf.shade(agg, color_key=color_key), px=1)
    
    # Convert datashader image to matplotlib
    pil_img = img.to_pil()
    ax.imshow(pil_img, origin='upper', aspect='auto',
              extent=[df[dim1].min(), df[dim1].max(),
                      df[dim2].min(), df[dim2].max()])
    
    legend_handles = [Patch(color=color_key[str(i)], label=f'syllable {i}') 
                  for i in range(MAX_SYLLABLE + 1)]
    legend_handles.append(Patch(color='#aaaaaa', label='other'))
    
ax.legend(handles=legend_handles, 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              fontsize=6,
              ncol=2)    
ax.set_title(f'UMAP {title} — colored by syllable')
ax.set_xlabel(dim1)
ax.set_ylabel(dim2)




plt.tight_layout()
plt.show()

# %% plot and view data (no Umap)


import matplotlib.gridspec as gridspec

vidx = 'vid_1'
anid  ='animal_2'
anid2 = 'animal_1'
# centering data around spine mid
# add "if spinemid is nan, then ...

entry = index[1]
anid  = 'track1'
anid2 = 'track2'



start, end = entry[anid2]['start'], entry[anid2]['end']
newdata2 = combined_arr[start:end+1]
cent2 = cent_all[start:end+1]

start, end = entry[anid]['start'], entry[anid]['end']
newdata = combined_arr[start:end+1]
speed = S[start:end+1]
syllab = syllables_raw[start:end+1]
cent = cent_all[start:end+1]
# newdata2 = data_all[vidx][anid2]['keypoints']
# newdata = data_all[vidx][anid]['keypoints']
# emb3 = data_all[vidx][anid]['UMAP']

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


axs[2].plot(xtime,np.convolve(speed[:,0],10,mode = 'same')*0.04)
axs[2].set_xlim([-50*.04,50*.04])
axs[2].set_ylabel('velocity cm/s')
axs[2].set_ylim([0,150])
axs[3].plot(xtime,syllab)
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
axs[0].scatter3D(cent[samp_t,0],
             cent[samp_t,1],
             cent[samp_t,2]) 

axs[0].scatter3D(cent2[samp_t,0],
             cent2[samp_t,1],
             cent2[samp_t,2],alpha = 0.2) 

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
    if syllab[t] == 3:
        axs[3].set_xlabel('walking')
    else:
        axs[3].set_xlabel(str(syllab[t]))
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
