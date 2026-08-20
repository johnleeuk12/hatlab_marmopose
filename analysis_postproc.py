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
import seaborn as sns
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

def get_syllable_instances(syllables_raw, combined_arr, fps=25):
    nan_mask = np.isnan(combined_arr).any(axis=(1, 2))
    
    # Boundaries occur where syllable changes OR where NaN status changes
    syl_change = np.diff(syllables_raw) != 0
    nan_change = np.diff(nan_mask.astype(int)) != 0
    boundaries = np.where(syl_change | nan_change)[0] + 1
    
    # Segment start/end indices
    starts = np.concatenate(([0], boundaries))
    ends   = np.concatenate((boundaries, [len(syllables_raw)]))
    
    rows = []
    for s, e in zip(starts, ends):
        # Skip NaN segments
        if nan_mask[s]:
            continue
        duration_frames = e - s
        rows.append({
            'syllable':        int(syllables_raw[s]),
            'start_frame':     s,
            'duration_frames': duration_frames,
        })
    
    return pd.DataFrame(rows)



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
syllables_org = np.concatenate([results[k]['syllable'] for k in sorted(results.keys())])

# compute_df: get syllable instance frequencies and durations

instances_df = get_syllable_instances(syllables_org, combined_arr, fps=25)

thresh = 0.1

hist, _ = np.histogram(instances_df['syllable'],np.arange(100),density = True )

syllables_raw = syllables_org.copy() 
for s in np.unique(syllables_org):
    if hist[s] < 1*1e-3: # 1% frequency threshold, subject to change
        syllables_raw[np.where(syllables_org == s)] = 99
    

comp_df = get_syllable_instances(syllables_raw, combined_arr, fps=25)

# sns.histplot(data = instances_df['syllable'],stat = 'percent')


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


# %%
# =============================================================================
# 4D syllable centroids: UMAP (x,y,z) + normalized velocity
# =============================================================================

# Extract velocity for valid frames (last column of T, already speed-aligned)
velocity_valid = T[valid_mask, -1].reshape(-1, 1)

# Normalize velocity to [0,1] so it's on comparable scale to UMAP dims
v_min, v_max = np.nanmin(velocity_valid), np.nanmax(velocity_valid)
velocity_norm = (velocity_valid - v_min) / (v_max - v_min + 1e-8)

# Build 4D embedding: (n_valid_frames, 4)
embedding_4d = np.concatenate([embedding_valid, velocity_norm*2], axis=1)

# Compute per-syllable centroids and spread in 4D space
syllable_centroids = {}
syllable_spread    = {}

for syl in range(n_syllables):
    mask = syllables_valid == syl
    if mask.sum() == 0:
        continue
    pts = embedding_4d[mask]
    syllable_centroids[syl] = np.nanmean(pts, axis=0)  # shape (4,)
    syllable_spread[syl]    = np.nanstd(pts,  axis=0)  # shape (4,)

# Build centroid matrix: shape (n_syllables, 4)
syl_ids         = sorted(syllable_centroids.keys())
centroid_matrix = np.stack([syllable_centroids[s] for s in syl_ids], axis=0)

# Pairwise Euclidean distances between syllable centroids
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.cluster.hierarchy import leaves_list

centroid_dist = cdist(centroid_matrix, centroid_matrix, metric='euclidean')

print(f"Syllable centroids computed for {len(syl_ids)} syllables in 4D space")
print(f"Centroid matrix shape: {centroid_matrix.shape}")

# =============================================================================
# Cluster spread summary
# =============================================================================
print("\nSyllable spread (std dev) per dimension [UMAP-x, UMAP-y, UMAP-z, velocity]:")
print(f"{'Syl':>5} {'n_frames':>10} {'std_x':>8} {'std_y':>8} {'std_z':>8} {'std_v':>8}")
for syl in syl_ids:
    mask     = syllables_valid == syl
    n_frames = mask.sum()
    std      = syllable_spread[syl]
    print(f"{syl:>5} {n_frames:>10} {std[0]:>8.3f} {std[1]:>8.3f} {std[2]:>8.3f} {std[3]:>8.3f}")

# =============================================================================
# Hierarchical clustering
# =============================================================================
# Condense distance matrix to 1D for linkage
from scipy.spatial.distance import squareform
dist_condensed = squareform(centroid_dist, checks=False)
Z = linkage(dist_condensed, method='ward')

# Optimal leaf ordering for cleaner dendrogram
ordered_leaves = leaves_list(Z)
syl_labels     = [str(syl_ids[i]) for i in range(len(syl_ids))]

# =============================================================================
# Figure: distance heatmap + dendrogram + spread
# =============================================================================
fig_sim, axes_sim = plt.subplots(
    2, 2,
    figsize=(16, 14),
    gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 2]}
)
fig_sim.suptitle('Syllable similarity analysis (4D: UMAP + velocity)', fontsize=13)

# --- Top left: dendrogram ---
ax_dend = axes_sim[0, 0]
dendrogram(
    Z,
    labels=syl_labels,
    ax=ax_dend,
    color_threshold=0.7 * max(Z[:, 2]),
    leaf_font_size=9,
    above_threshold_color='grey'
)
ax_dend.set_title('Hierarchical clustering of syllable centroids (Ward)', fontsize=10)
ax_dend.set_ylabel('Distance', fontsize=9)
ax_dend.set_xlabel('Syllable', fontsize=9)
ax_dend.spines['top'].set_visible(False)
ax_dend.spines['right'].set_visible(False)

# --- Top right: cluster spread heatmap (std per dimension) ---
ax_spread = axes_sim[0, 1]
spread_matrix = np.stack([syllable_spread[s] for s in syl_ids], axis=0)  # (n_syl, 4)
im_spread = ax_spread.imshow(spread_matrix, aspect='auto', cmap='YlOrRd')
ax_spread.set_xticks(range(4))
ax_spread.set_xticklabels(['UMAP-x', 'UMAP-y', 'UMAP-z', 'velocity'], fontsize=8, rotation=30)
ax_spread.set_yticks(range(len(syl_ids)))
ax_spread.set_yticklabels(syl_labels, fontsize=8)
ax_spread.set_title('Cluster spread (std dev)', fontsize=10)
plt.colorbar(im_spread, ax=ax_spread, fraction=0.046, pad=0.04)

# --- Bottom left: distance heatmap (reordered by dendrogram) ---
ax_heat = axes_sim[1, 0]
dist_reordered = centroid_dist[np.ix_(ordered_leaves, ordered_leaves)]
labels_reordered = [syl_labels[i] for i in ordered_leaves]

im_heat = ax_heat.imshow(dist_reordered, aspect='auto', cmap='viridis_r')
ax_heat.set_xticks(range(len(syl_ids)))
ax_heat.set_xticklabels(labels_reordered, fontsize=8, rotation=90)
ax_heat.set_yticks(range(len(syl_ids)))
ax_heat.set_yticklabels(labels_reordered, fontsize=8)
ax_heat.set_title('Pairwise centroid distance (reordered by clustering)', fontsize=10)
plt.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)

# Annotate heatmap cells with distance values
for i in range(len(syl_ids)):
    for j in range(len(syl_ids)):
        val = dist_reordered[i, j]
        ax_heat.text(j, i, f'{val:.1f}',
                     ha='center', va='center',
                     fontsize=6,
                     color='white' if val < dist_reordered.max() * 0.5 else 'black')

# --- Bottom right: n_frames per syllable bar chart ---
ax_bar = axes_sim[1, 1]
n_frames_per_syl = [np.sum(syllables_valid == s) for s in syl_ids]
bars = ax_bar.barh(range(len(syl_ids)), n_frames_per_syl,
                   color=[cc.glasbey[i] for i in range(len(syl_ids))],
                   edgecolor='none')
ax_bar.set_yticks(range(len(syl_ids)))
ax_bar.set_yticklabels(syl_labels, fontsize=8)
ax_bar.set_xlabel('n frames', fontsize=9)
ax_bar.set_title('Frames per syllable', fontsize=10)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# =============================================================================
# Print closest syllable pairs (most similar)
# =============================================================================
print("\nClosest syllable pairs by centroid distance:")
print(f"{'Syl A':>6} {'Syl B':>6} {'Distance':>10}")

# Get upper triangle indices, sorted by distance
n = len(syl_ids)
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((syl_ids[i], syl_ids[j], centroid_dist[i, j]))

pairs_sorted = sorted(pairs, key=lambda x: x[2])
for syl_a, syl_b, dist in pairs_sorted[:10]:
    print(f"{syl_a:>6} {syl_b:>6} {dist:>10.3f}")


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


# %% plot and view data
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path

# =============================================================================
# Config
# =============================================================================
SESSION_IDX  = 1        # which session from index to view
N_CAMS       = 4
VIDEO_SUFFIX = '2'      # bak-{cam}-{VIDEO_SUFFIX}.mp4
FPS          = 25       # frames per second
DT           = 1 / FPS  # seconds per frame
WINDOW_HALF  = 50       # half-window for velocity/syllable time plots (frames)
AX_LIM       = 450      # 3D pose axis limit
BUFFER_HALF  = 150      # frames to buffer on each side of current position

# =============================================================================
# Load session data
# =============================================================================
entry = index[SESSION_IDX]
anid  = 'track1'
anid2 = 'track2'

start,  end  = entry[anid]['start'],  entry[anid]['end']
start2, end2 = entry[anid2]['start'], entry[anid2]['end']

cent     = cent_all[start:end+1]
cent2    = cent_all[start2:end2+1]

newdata_old  = combined_arr[start:end+1]
newdata2 = combined_arr[start2:end2+1]

for sk in np.arange(16):
    newdata[:,sk,:] = newdata_old[:,sk,:]-cent[:,:]
# fig, axs = plt.subplots(4, 1)
#     # sc = 3
# for i in [0,1,2]:
#         # axs[i].plot(coord[name][:,sc,i],color = 'blue')
#         # axs[i].plot(D_track_opt[:,sc,i],'green')
#         axs[i].plot(newdata[:,8,i],'red')
#         axs[i].plot(cent[:,i],'blue')
#         # axs[i].plot(time3*800,'black',marker='.', linestyle='None')
#         # axs[i].set_ylim([-300,700])
    
# plt.show

speed    = S[start:end+1]
syllab   = syllables_raw[start:end+1]
xtime    = np.arange(len(newdata)) * DT

# =============================================================================
# Video capture setup — single combined video with rolling buffer
# =============================================================================
def open_combined_capture(session_dir):
    video_path = Path(session_dir).parent / 'videos_labeled_2d' / 'vid_combined.mp4'
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    return cap


class FrameBuffer:
    def __init__(self, cap, half=BUFFER_HALF):
        self.cap    = cap
        self.half   = half
        self.center = -1
        self.buffer = {}  # frame_num -> np.ndarray

    def _load_range(self, start, end):
        """Sequentially read frames from start to end into buffer."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for fn in range(start, end + 1):
            ret, frame = self.cap.read()
            if ret: 
                self.buffer[fn] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                self.buffer[fn] = np.zeros((100, 100, 3), dtype=np.uint8)

    def get(self, frame_num):
        """Return frame, refilling buffer if frame_num is outside current window."""
        if frame_num not in self.buffer:
            self.buffer.clear()
            start = max(0, frame_num - self.half)
            end   = frame_num + self.half
            self._load_range(start, end)
            self.center = frame_num
        return self.buffer[frame_num]


cap = open_combined_capture(entry['session_dir'])
buf = FrameBuffer(cap, half=BUFFER_HALF)

# Pre-load initial buffer
buf.get(0)

# =============================================================================
# Figure layout
# =============================================================================
#  Col 0 (70%): video — full height left side
#  Col 1 (30%): 4 stacked panels — 3D pose, centroid, velocity, syllable
# =============================================================================
fig = plt.figure(figsize=(22, 12))
gs  = gridspec.GridSpec(4, 2, figure=fig,
                        width_ratios=[4, 1],
                        height_ratios=[1, 1, 0.7, 0.7],
                        hspace=0.15, wspace=0.1)

# Left: video — spans all 4 rows
ax_vid = fig.add_subplot(gs[:, 0])
ax_vid.axis('off')
ax_vid.set_title('Camera grid', fontsize=9)

# # Right row 0: 3D pose for non centered
# ax_pose = fig.add_subplot(gs[0:2, 1], projection='3d')
# ax_pose.set_xlim([-200, 450])
# ax_pose.set_ylim([-200, 450])
# ax_pose.set_zlim([-500, 300])
# ax_pose.set_title('3D pose', fontsize=9)

# Right row 0: 3D pose for centered
ax_pose = fig.add_subplot(gs[0:2, 1], projection='3d')
ax_pose.set_xlim([-200, 200])
ax_pose.set_ylim([-200, 200])
ax_pose.set_zlim([-200, 200])
ax_pose.set_title('3D pose', fontsize=9)


# Right row 1: centroid scatter
ax_cent = fig.add_subplot(gs[0:2, 1], projection='3d')
samp_t  = random.sample(range(0, len(newdata)),  min(5000, len(newdata)))
samp_t2 = random.sample(range(0, len(newdata2)), min(5000, len(newdata2)))
ax_cent.scatter3D(cent[samp_t, 0],   cent[samp_t, 1],   cent[samp_t, 2],
                  s=1, alpha=0.3, label=anid)
ax_cent.scatter3D(cent2[samp_t2, 0], cent2[samp_t2, 1], cent2[samp_t2, 2],
                  s=1, alpha=0.1, label=anid2)
ax_cent.set_title('centroid', fontsize=9)
ax_cent.set_xlim([-200, 550])
ax_cent.set_ylim([-200, 550])
ax_cent.set_zlim([-200, 650])
ax_cent.legend(fontsize=7)
ax_cent.set_visible(False)  # hidden by default

# Right row 2: velocity
ax_vel = fig.add_subplot(gs[2, 1])
ax_vel.plot(xtime, np.convolve(speed[:, 0], 10, mode='same') * DT, lw=0.8)
ax_vel.set_ylabel('velocity (cm/s)', fontsize=8)
ax_vel.set_ylim([0, 50])
ax_vel.set_title('time (s)', fontsize=8)

# Right row 3: syllable
ax_syl = fig.add_subplot(gs[3, 1])
ax_syl.plot(xtime, syllab, lw=0.8)
ax_syl.set_ylabel('syllable', fontsize=8)
ax_syl.set_ylim([0, 25])

fig.subplots_adjust(bottom=0.12, left=0.02, right=0.97, top=0.95)

#==============================================================================
#Toggle button

# Toggle buttons for pose / centroid
ax_btn_pose = fig.add_axes([0.75, 0.91, 0.1, 0.03])
ax_btn_cent = fig.add_axes([0.85, 0.91, 0.1, 0.03])

from matplotlib.widgets import Button
btn_pose = Button(ax_btn_pose, 'Pose',     color='steelblue', hovercolor='lightblue')
btn_cent = Button(ax_btn_cent, 'Centroid', color='lightgrey', hovercolor='lightblue')

def show_pose(event):
    ax_pose.set_visible(True)
    ax_cent.set_visible(False)
    btn_pose.color = 'steelblue'
    btn_cent.color = 'lightgrey'
    fig.canvas.draw_idle()

def show_cent(event):
    ax_pose.set_visible(False)
    ax_cent.set_visible(True)
    btn_pose.color = 'lightgrey'
    btn_cent.color = 'steelblue'
    fig.canvas.draw_idle()

btn_pose.on_clicked(show_pose)
btn_cent.on_clicked(show_cent)

# =============================================================================
# Initial frame render
# =============================================================================
t0 = 0

# Video
init_frame = buf.get(t0)
vid_im = ax_vid.imshow(init_frame, aspect='auto')

# Skeleton lines
lines  = {}
lines2 = {}
for sk in range(len(skeleton2)):
    lines[sk]  = ax_pose.plot3D(
        newdata[t0, skeleton2[sk], 0],
        newdata[t0, skeleton2[sk], 1],
        -newdata[t0, skeleton2[sk], 2],
        sk_color[sk]
    )
    lines2[sk] = ax_pose.plot3D(
        newdata2[t0, skeleton2[sk], 0],
        newdata2[t0, skeleton2[sk], 1],
        -newdata2[t0, skeleton2[sk], 2],
        sk_color[sk], alpha=0.2
    )

# Red vertical time lines
red_line_vel = ax_vel.axvline(x=t0 * DT, color='red', lw=1)
red_line_syl = ax_syl.axvline(x=t0 * DT, color='red', lw=1)

# Initial time window
ax_vel.set_xlim([-WINDOW_HALF * DT, WINDOW_HALF * DT])
ax_syl.set_xlim([-WINDOW_HALF * DT, WINDOW_HALF * DT])

# =============================================================================
# Slider
# =============================================================================
ax_slider = fig.add_axes([0.05, 0.04, 0.9, 0.02])
time_slider = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=0,
    valmax=len(newdata) - 1,
    valinit=0,
    valstep=1.0
)

# =============================================================================
# Update function
# =============================================================================
def update(val):
    t = int(time_slider.val)

    # --- Update video ---
    frame = buf.get(t)
    vid_im.set_data(frame)

    # --- Update 3D skeleton ---
    for sk in range(len(skeleton2)):
        lines[sk][0].set_data_3d(
            newdata[t, skeleton2[sk], 0],
            newdata[t, skeleton2[sk], 1],
            -newdata[t, skeleton2[sk], 2]
        )
        lines2[sk][0].set_data_3d(
            newdata2[t, skeleton2[sk], 0],
            newdata2[t, skeleton2[sk], 1],
            -newdata2[t, skeleton2[sk], 2]
        )

    # --- Update time window ---
    t_sec = t * DT
    ax_vel.set_xlim([(t - WINDOW_HALF) * DT, (t + WINDOW_HALF) * DT])
    ax_syl.set_xlim([(t - WINDOW_HALF) * DT, (t + WINDOW_HALF) * DT])
    red_line_vel.set_xdata([t_sec])
    red_line_syl.set_xdata([t_sec])

    # --- Update syllable label ---
    syl_id = int(syllab[t]) if not np.isnan(syllab[t]) else -1
    label_map = {3: 'walking'}  # add known syllable->behavior mappings here
    ax_syl.set_xlabel(label_map.get(syl_id, str(syl_id)), fontsize=8)

    fig.canvas.draw_idle()


time_slider.on_changed(update)

# =============================================================================
# Keyboard navigation
# =============================================================================
def on_key(event):
    step = 10 if event.key in ('up', 'down') else 1
    if event.key in ('right', 'up'):
        time_slider.set_val(min(time_slider.val + step, time_slider.valmax))
    elif event.key in ('left', 'down'):
        time_slider.set_val(max(time_slider.val - step, time_slider.valmin))

fig.canvas.mpl_connect('key_press_event', on_key)

# =============================================================================
# Cleanup on close
# =============================================================================
def on_close(event):
    cap.release()

fig.canvas.mpl_connect('close_event', on_close)

# =============================================================================
# Show
# =============================================================================
plt.show()