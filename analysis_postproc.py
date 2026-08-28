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
import networkx as nx
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

def get_syllable_instances(syllables_raw, combined_arr, fps=25, min_duration=3):
    syl = np.asarray(syllables_raw)
    nan_mask = np.isnan(combined_arr).any(axis=(1, 2))
    n = len(syl)

    cols = ['syllable', 'start_frame', 'duration_frames']
    if n == 0:
        return pd.DataFrame(columns=cols)

    # --- Initial segmentation: split on label change or NaN-status change ---
    change = (np.diff(syl) != 0) | (np.diff(nan_mask.astype(int)) != 0)
    bounds = np.where(change)[0] + 1
    starts = np.concatenate(([0], bounds))
    ends   = np.concatenate((bounds, [n]))

    runs = [{'start': int(s), 'end': int(e), 'label': int(syl[s]),
             'is_nan': bool(nan_mask[s]), 'dropped': False}
            for s, e in zip(starts, ends)]

    # --- Absorb short runs, left to right so relabelling cascades ---
    n_prev = n_next = n_drop = 0
    for i, r in enumerate(runs):
        if r['is_nan'] or (r['end'] - r['start']) >= min_duration:
            continue

        prev = runs[i - 1] if i > 0 else None
        nxt  = runs[i + 1] if i + 1 < len(runs) else None

        prev_ok = prev is not None and not prev['is_nan'] and not prev['dropped']
        next_ok = nxt  is not None and not nxt['is_nan']

        if prev_ok:
            r['label'] = prev['label']
            n_prev += 1
        elif next_ok:
            r['label'] = nxt['label']
            n_next += 1
        else:
            r['dropped'] = True
            n_drop += 1

    # --- Emit, merging runs that now share a label and stay frame-adjacent ---
    rows = []
    for r in runs:
        if r['is_nan'] or r['dropped']:
            continue
        if rows and rows[-1]['syllable'] == r['label'] \
                and rows[-1]['start_frame'] + rows[-1]['duration_frames'] == r['start']:
            rows[-1]['duration_frames'] += r['end'] - r['start']
        else:
            rows.append({'syllable':        r['label'],
                         'start_frame':     r['start'],
                         'duration_frames': r['end'] - r['start']})

    out = pd.DataFrame(rows, columns=cols)
    out.attrs['n_absorbed_prev'] = n_prev
    out.attrs['n_absorbed_next'] = n_next
    out.attrs['n_dropped']       = n_drop
    # print(f'  short runs (<{min_duration} frames): '
    #       f'{n_prev} absorbed into preceding, '
    #       f'{n_next} into following, {n_drop} dropped')
    return out

def get_transition_matrix(inst_df, n_states=100, normalize='bigram'):
    syl = inst_df['syllable'].values
    
    trans_mat = np.zeros((n_states, n_states), dtype=float)
    np.add.at(trans_mat, (syl[:-1], syl[1:]), 1)

    if normalize == 'bigram':
        total = trans_mat.sum()
        if total > 0:
            trans_mat /= total
    elif normalize == 'rows':
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_mat /= row_sums

    return trans_mat


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
# entry = index[0]
# start, end = entry['track1']['start'], entry['track1']['end']
# track1_data = combined_arr[start:end+1]

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

model_name = '2026_08_17-15_01_57'

# model_name = '2026_08_19-10_33_58'

results = load_results(project_dir, model_name)
syllables_org = np.concatenate([results[k]['syllable'] for k in sorted(results.keys())])

# compute_df: get syllable instance frequencies and durations

instances_df = get_syllable_instances(syllables_org, combined_arr, fps=25)

thresh = 0.1

hist, _ = np.histogram(instances_df['syllable'],np.arange(100),density = True )

syllables_raw = syllables_org.copy() 
for s in np.unique(syllables_org):
    if hist[s] < 2*1e-3: # 1% frequency threshold, subject to change
        syllables_raw[np.where(syllables_org == s)] = 99



comp_df = get_syllable_instances(syllables_raw, combined_arr, fps=25)
trans_combined = get_transition_matrix(comp_df, normalize='bigram')
comp_df.to_pickle(os.path.join(project_dir, 'comp_df.pkl'))
# sns.histplot(data = instances_df['syllable'],stat = 'percent')
# sns.histplot(data = comp_df['duration_frames'],binwidth = 1)


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

reducer = umap.UMAP(n_neighbors=25,n_components=3,min_dist = 0.1)

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

from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import matplotlib.gridspec as gridspec

# --- Exclude syllable 99 (low-frequency catch-all) ---
syll_include = sorted([s for s in np.unique(syllables_valid) if s != 99])
n_syl        = len(syll_include)
syl_labels   = [str(s) for s in syll_include]

# Color map: consistent color per syllable across all plots
syl_colors = {s: cc.glasbey[i] for i, s in enumerate(syll_include)}

# --- Extract velocity for valid frames (last column of T) ---
velocity_valid = T[valid_mask, -1].reshape(-1, 1)
v_min, v_max   = np.nanmin(velocity_valid), np.nanmax(velocity_valid)
velocity_norm  = (velocity_valid - v_min) / (v_max - v_min + 1e-8)

# --- Build 4D embedding: (n_valid_frames, 4) ---
embedding_4d = np.concatenate([embedding_valid, velocity_norm * 2], axis=1)

# --- Per-syllable centroids and spread ---
syllable_centroids = {}
syllable_spread    = {}

for syl in syll_include:
    mask = syllables_valid == syl
    if mask.sum() == 0:
        continue
    pts = embedding_4d[mask]
    syllable_centroids[syl] = np.nanmean(pts, axis=0)  # (4,)
    syllable_spread[syl]    = np.nanstd(pts,  axis=0)  # (4,)

centroid_matrix = np.stack([syllable_centroids[s] for s in syll_include], axis=0)
centroid_dist   = cdist(centroid_matrix, centroid_matrix, metric='euclidean')

# --- Hierarchical clustering ---
Z              = linkage(squareform(centroid_dist, checks=False), method='ward')
ordered_leaves = leaves_list(Z)
labels_ordered = [syl_labels[i] for i in ordered_leaves]
dist_reordered = centroid_dist[np.ix_(ordered_leaves, ordered_leaves)]
syll_ordered   = [syll_include[i] for i in ordered_leaves]

# --- Real syllable instance frequency from comp_df (excludes 99) ---
freq_counts  = comp_df[comp_df['syllable'] != 99]['syllable'].value_counts()
freq_total   = freq_counts.sum()
freq_dict    = {s: freq_counts.get(s, 0) / freq_total for s in syll_include}

# --- Transition matrix from comp_df ---
N_STATES = 100

def get_transition_matrix(inst_df, n_states=N_STATES, normalize='bigram'):
    syl       = inst_df['syllable'].values
    trans_mat = np.zeros((n_states, n_states), dtype=float)
    np.add.at(trans_mat, (syl[:-1], syl[1:]), 1)
    if normalize == 'bigram':
        total = trans_mat.sum()
        if total > 0:
            trans_mat /= total
    elif normalize == 'rows':
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_mat /= row_sums
    return trans_mat

trans_combined = get_transition_matrix(comp_df[comp_df['syllable'] != 99], normalize='bigram')

# Subset transition matrix to syll_include only
syl_idx        = np.array(syll_include)
trans_sub      = trans_combined[np.ix_(syl_idx, syl_idx)]
trans_reordered = trans_sub[np.ix_(ordered_leaves, ordered_leaves)]

# # Per-session transition matrices
# trans_per_session = {}
# for i, entry in enumerate(index):
#     sess_start = entry['track1']['start']
#     sess_end   = entry['track2']['end']
#     sess_df    = comp_df[
#         (comp_df['start_frame'] >= sess_start) &
#         (comp_df['start_frame'] <= sess_end)  &
#         (comp_df['syllable']    != 99)
#     ].reset_index(drop=True)
#     trans_per_session[f'session{i}'] = get_transition_matrix(sess_df, normalize='bigram')
#     print(f"session{i}: {len(sess_df)} instances")

# Cluster transition matrix independently
# Use row+col profiles as fingerprint for each syllable
trans_sym      = (trans_sub + trans_sub.T) / 2
Z_trans        = linkage(squareform(cdist(trans_sym, trans_sym, metric='euclidean'), 
                                    checks=False), method='ward')
ordered_leaves2 = leaves_list(Z_trans)
labels_ordered2 = [syl_labels[i] for i in ordered_leaves2]
trans_reordered2 = trans_sub[np.ix_(ordered_leaves2, ordered_leaves2)]

# %%
# =============================================================================
# Figure 1: Dendrogram + Spread + Distance heatmap + Transition heatmap
# =============================================================================

fig1 = plt.figure(figsize=(24, 14))
fig1.suptitle('Syllable similarity analysis (4D: UMAP + velocity)', fontsize=13)

gs1 = gridspec.GridSpec(2, 3, figure=fig1,
                         width_ratios=[2.5, 2.5, 1.5],
                         height_ratios=[1, 2],
                         hspace=0.35, wspace=0.4)

# Row 0, Col 0: Dendrogram (unchanged)
ax_dend = fig1.add_subplot(gs1[0, 0])
dendrogram(
    Z,
    labels=syl_labels,
    ax=ax_dend,
    color_threshold=0.7 * max(Z[:, 2]),
    leaf_font_size=9,
    above_threshold_color='grey'
)
ax_dend.set_title('Hierarchical clustering (Ward)', fontsize=10)
ax_dend.set_ylabel('Distance', fontsize=9)
ax_dend.set_xlabel('Syllable', fontsize=9)
ax_dend.spines['top'].set_visible(False)
ax_dend.spines['right'].set_visible(False)
 
# --- Row 0, Col 1-2: Syllable frequency bar chart ---
ax_freq = fig1.add_subplot(gs1[0, 1:])
freq_vals    = [freq_dict[s] for s in syll_include]
freq_colors  = [syl_colors[s] for s in syll_include]
ax_freq.bar(range(n_syl), freq_vals, color=freq_colors, edgecolor='none')
ax_freq.set_xticks(range(n_syl))
ax_freq.set_xticklabels(syl_labels, fontsize=8, rotation=90)
ax_freq.set_ylabel('Instance frequency', fontsize=9)
ax_freq.set_title('Syllable frequency (instance-based, excl. 99)', fontsize=10)
ax_freq.spines['top'].set_visible(False)
ax_freq.spines['right'].set_visible(False)

# Row 1, Col 0: Distance heatmap — reordered by centroid clustering
ax_heat = fig1.add_subplot(gs1[1, 0])
im_heat = ax_heat.imshow(dist_reordered, aspect='equal', cmap='viridis_r')
ax_heat.set_xticks(range(n_syl))
ax_heat.set_xticklabels(labels_ordered, fontsize=7, rotation=90)
ax_heat.set_yticks(range(n_syl))
ax_heat.set_yticklabels(labels_ordered, fontsize=7)
ax_heat.set_title('Centroid distance\n(ordered by centroid clustering)', fontsize=10)
plt.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)

# Row 1, Col 1: Transition heatmap — reordered by transition clustering
ax_trans = fig1.add_subplot(gs1[1, 1])
im_trans = ax_trans.imshow(trans_reordered2, aspect='equal', cmap='hot_r',
                            vmin=0, vmax=np.percentile(trans_reordered2[trans_reordered2 > 0], 95))
ax_trans.set_xticks(range(n_syl))
ax_trans.set_xticklabels(labels_ordered2, fontsize=7, rotation=90)
ax_trans.set_yticks(range(n_syl))
ax_trans.set_yticklabels(labels_ordered2, fontsize=7)
ax_trans.set_title('Transition matrix\n(ordered by transition clustering)', fontsize=10)
ax_trans.set_xlabel('to', fontsize=8)
ax_trans.set_ylabel('from', fontsize=8)
plt.colorbar(im_trans, ax=ax_trans, fraction=0.046, pad=0.04)
    
# --- Row 1, Col 2: Cluster spread heatmap ---
ax_spread = fig1.add_subplot(gs1[1, 2])
spread_matrix = np.stack([syllable_spread[s] for s in syll_include], axis=0)
im_spread = ax_spread.imshow(spread_matrix, aspect='auto', cmap='YlOrRd')
ax_spread.set_xticks(range(4))
ax_spread.set_xticklabels(['UMAP-x', 'UMAP-y', 'UMAP-z', 'velocity'], fontsize=8, rotation=30)
ax_spread.set_yticks(range(n_syl))
ax_spread.set_yticklabels(syl_labels, fontsize=8)
ax_spread.set_title('Cluster spread (std dev)', fontsize=10)
plt.colorbar(im_spread, ax=ax_spread, fraction=0.046, pad=0.04)

plt.show()


# %% 
# =============================================================================
# Syllable transition graph
# =============================================================================

EDGE_THRESH  = 0.002
NODE_SCALING = 2000
LAYOUT       = 'circular'  # or 'spring'

# Build graph directly from subsetted transition matrix
trans_plot = trans_sub * 100
G          = nx.from_numpy_array(trans_plot)

# Relabel nodes from 0..n_syl to actual syllable numbers
mapping = {i: syll_include[i] for i in range(n_syl)}
G       = nx.relabel_nodes(G, mapping)

# Remove weak edges
weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < EDGE_THRESH * 100]
G.remove_edges_from(weak_edges)

# Layout
pos = nx.circular_layout(G) if LAYOUT == 'circular' else nx.spring_layout(G, seed=42)

# Node sizes proportional to frequency
node_sizes = [freq_dict[s] * NODE_SCALING + 1000 for s in G.nodes()]

widths = nx.get_edge_attributes(G, 'weight')

fig_graph, ax_graph = plt.subplots(figsize=(12, 12))
ax_graph.axis('off')
ax_graph.set_title('Syllable transition graph', fontsize=11)

nx.draw_networkx_nodes(G, pos, ax=ax_graph,
                       node_size=node_sizes,
                       node_color='white',
                       edgecolors='red')

nx.draw_networkx_edges(G, pos, ax=ax_graph,
                       edgelist=widths.keys(),
                       width=list(widths.values()),
                       edge_color='black',
                       alpha=0.6)

nx.draw_networkx_labels(G, pos, ax=ax_graph,
                        font_color='black', font_size=9)

plt.tight_layout()
plt.show()
# %%
# =============================================================================
# Syllable context sequences
#
# Finds the most frequent sequences of syllables immediately preceding or
# following a target syllable (e.g. locomotion syllables 3 and 6), and plots
# the top-k as horizontal histograms.
#
# Notes
# -----
# * Pass the FULL comp_df, including syllable 99. Dropping 99 rows beforehand
#   would splice together instances that were not actually adjacent, turning
#   3 -> 99 -> 7 into a spurious 3 -> 7. Windows containing 99 are excluded
#   here instead, via the `exclude` argument.
# * Windows that straddle a discontinuity are dropped. Consecutive rows of
#   comp_df are not necessarily adjacent in time: get_syllable_instances drops
#   NaN segments, so the row before a NaN gap and the row after it sit next to
#   each other in the frame while being separated in the recording. Contiguity
#   is tested with start_frame[i+1] == start_frame[i] + duration_frames[i],
#   which also handles session and track boundaries for free.
# * get_syllable_instances merges consecutive identical labels into one
#   instance, so no window can contain an immediate self-repeat.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def _instance_groups(comp_df):
    """
    Split instances into contiguous runs.

    Returns
    -------
    syl   : (n,) int array of syllable labels
    group : (n,) int array; equal values mean uninterrupted succession
    """
    syl   = comp_df['syllable'].values.astype(int)
    start = comp_df['start_frame'].values.astype(np.int64)
    dur   = comp_df['duration_frames'].values.astype(np.int64)

    if len(syl) < 2:
        return syl, np.zeros(len(syl), dtype=int)

    contiguous = start[1:] == start[:-1] + dur[:-1]
    group = np.concatenate([[0], np.cumsum(~contiguous)])
    return syl, group


def get_context_sequences(comp_df, target, n=2, direction='before',
                          exclude=(99,), top_k=10):
    """
    Count the n-syllable sequences that precede or follow a target syllable.

    Parameters
    ----------
    comp_df   : DataFrame with syllable, start_frame, duration_frames
    target    : int, the syllable whose context is examined
    n         : int, sequence length (2 or 3 is usually readable)
    direction : 'before' or 'after'
    exclude   : iterable of syllables; any window containing one is discarded
    top_k     : int, number of sequences returned

    Returns
    -------
    DataFrame with columns sequence, label, count, frequency, expected, ratio
        frequency : count / n_windows
        expected  : count predicted if syllables were drawn independently
        ratio     : frequency / expected_frequency, so >1 means the sequence
                    occurs more often than its constituent syllables alone
                    would predict
    """
    if direction not in ('before', 'after'):
        raise ValueError("direction must be 'before' or 'after'")

    syl, group = _instance_groups(comp_df)
    exclude = set(exclude)
    n_inst = len(syl)

    # Marginal instance probabilities for the independence null
    counts = Counter(s for s in syl if s not in exclude)
    total  = sum(counts.values())
    p_marg = {s: c / total for s, c in counts.items()} if total else {}

    windows = []
    for i in np.flatnonzero(syl == target):
        if direction == 'before':
            lo, hi = i - n, i
        else:
            lo, hi = i + 1, i + 1 + n

        if lo < 0 or hi > n_inst:
            continue
        # Whole window plus the target must lie in one contiguous run
        if group[lo] != group[i] or group[hi - 1] != group[i]:
            continue

        win = tuple(syl[lo:hi])
        if exclude & set(win):
            continue
        windows.append(win)

    if not windows:
        return pd.DataFrame(columns=['sequence', 'label', 'count',
                                     'frequency', 'expected', 'ratio'])

    tally = Counter(windows)
    n_win = len(windows)

    rows = []
    for win, c in tally.most_common(top_k):
        p_exp = np.prod([p_marg.get(s, 0.0) for s in win])
        label = (' > '.join(map(str, win)) + f' > [{target}]'
                 if direction == 'before'
                 else f'[{target}] > ' + ' > '.join(map(str, win)))
        rows.append({
            'sequence':  win,
            'label':     label,
            'count':     c,
            'frequency': c / n_win,
            'expected':  p_exp * n_win,
            'ratio':     (c / n_win) / p_exp if p_exp > 0 else np.nan,
        })

    out = pd.DataFrame(rows)
    out.attrs['n_windows'] = n_win
    out.attrs['n_target']  = int((syl == target).sum())
    return out


def plot_context_sequences(comp_df, targets=(3, 6), n=2, top_k=10,
                           exclude=(99,), show_expected=True):
    """
    Grid of horizontal histograms: one row per target, columns before / after.

    When show_expected is True, a hollow marker shows the count predicted under
    independence, so a tall bar that merely reflects two common syllables is
    distinguishable from a genuinely stereotyped sequence.
    """
    targets = list(targets)
    fig, axes = plt.subplots(len(targets), 2,
                             figsize=(15, 3.2 * len(targets) + 1),
                             squeeze=False)

    results = {}
    for r, target in enumerate(targets):
        for c, direction in enumerate(['before', 'after']):
            ax = axes[r][c]
            df = get_context_sequences(comp_df, target, n=n,
                                       direction=direction,
                                       exclude=exclude, top_k=top_k)
            results[(target, direction)] = df

            if df.empty:
                ax.text(0.5, 0.5, f'no windows for syllable {target}',
                        ha='center', va='center', fontsize=9)
                ax.set_axis_off()
                continue

            y = np.arange(len(df))[::-1]        # most frequent at the top
            ax.barh(y, df['count'], color='steelblue', edgecolor='none')

            if show_expected:
                ax.scatter(df['expected'], y, s=28, facecolors='none',
                           edgecolors='crimson', linewidths=1.2, zorder=3,
                           label='expected if independent')
                ax.legend(fontsize=7, loc='lower right')

            ax.set_yticks(y)
            ax.set_yticklabels(df['label'], fontsize=8, family='monospace')
            ax.set_xlabel('instance count', fontsize=9)
            ax.set_title(
                f'{n}-syllable sequences {direction} syllable {target}   '
                f'(n={df.attrs["n_windows"]} of '
                f'{df.attrs["n_target"]} occurrences)',
                fontsize=9
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.show()
    return results


# =============================================================================
# Usage
# =============================================================================
ctx = plot_context_sequences(comp_df, targets=(3,4), n=2, top_k=10)

9# ranked by enrichment rather than raw count
df = ctx[(3, 'before')]
print(df.sort_values('ratio', ascending=False)[
          ['label', 'count', 'expected', 'ratio']].to_string(index=False))

# single direction, longer context
df = get_context_sequences(comp_df, target=3, n=3, direction='after')

print(df.sort_values('ratio', ascending=False)[
          ['label', 'count', 'expected', 'ratio']].to_string(index=False))



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
newdata = newdata_old.copy()


for sk in np.arange(16):
    newdata[:,sk,:] = newdata_old[:,sk,:] #-cent[:,:]
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
ax_pose = fig.add_subplot(gs[0:2, 1], projection='3d')
ax_pose.set_xlim([-200, 450])
ax_pose.set_ylim([-200, 450])
ax_pose.set_zlim([-200, 600])
ax_pose.set_title('3D pose', fontsize=9)

# Right row 0: 3D pose for centered
# ax_pose = fig.add_subplot(gs[0:2, 1], projection='3d')
# ax_pose.set_xlim([-200, 200])
# ax_pose.set_ylim([-200, 200])
# ax_pose.set_zlim([-200, 200])
# ax_pose.set_title('3D pose', fontsize=9)


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
        newdata[t0, skeleton2[sk], 2],
        sk_color[sk]
    )
    lines2[sk] = ax_pose.plot3D(
        newdata2[t0, skeleton2[sk], 0],
        newdata2[t0, skeleton2[sk], 1],
        newdata2[t0, skeleton2[sk], 2],
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
            newdata[t, skeleton2[sk], 2]
        )
        lines2[sk][0].set_data_3d(
            newdata2[t, skeleton2[sk], 0],
            newdata2[t, skeleton2[sk], 1],
            newdata2[t, skeleton2[sk], 2]
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


# np.save('/home/jlee629/kpmoseq/projects/feb_may/combined_eg.npy',combined_arr[0:50000,:,:],allow_pickle = True)
comp_df.to_csv(os.path.join(project_dir, 'comp_df_v2.csv'), index=False)


# %% posture angle testing (temp code)


# ax   = combined_arr[:, 8, :] - combined_arr[:, 9, :]          # trunk vector
# run  = np.linalg.norm(ax[:, [0, 1]], axis=1)   # horizontal component (x,y)
# elev = np.degrees(np.arctan2(ax[:, 2]*-1, run))

# sns.histplot(elev)


# L = np.linalg.norm(combined_arr[:, 8, :] - combined_arr[:, 9, :], axis=1)
# good = np.abs(L - np.nanmedian(L)) / np.nanmedian(L) < 0.25

# sns.histplot(elev[good])

Z_SIGN = -1

# quality filter
L    = np.linalg.norm(combined_arr[:, 8, :] - combined_arr[:, 9, :], axis=1)
good = np.abs(L - np.nanmedian(L)) / np.nanmedian(L) < 0.25

# trunk elevation, folded
v    = combined_arr[:, 8, :] - combined_arr[:, 9, :]
elev = np.abs(np.degrees(np.arctan2(v[:, 2] * Z_SIGN,
                                    np.linalg.norm(v[:, [0, 1]], axis=1))))


v1 = combined_arr[:, 9, :] - combined_arr[:, 8, :]
v2 = combined_arr[:, 3, :] - combined_arr[:, 8, :]
n1 = np.linalg.norm(v1, axis=1)
n2 = np.linalg.norm(v2, axis=1)
with np.errstate(invalid='ignore', divide='ignore'):
    cos = np.einsum('ij,ij->i', v1, v2) / (n1 * n2)
cos = np.clip(cos, -1.0, 1.0)
flexion = np.degrees(np.arccos(cos))

sns.histplot(flexion[good])

#%%

# frames belonging to each syllable
d = comp_df[comp_df.syllable != 99]
sel = {}
for s, g in d.groupby('syllable'):
    i = np.concatenate([np.arange(r.start_frame, r.start_frame + r.duration_frames)
                        for r in g.itertuples(index=False)])
    e = elev[i[good[i] & np.isfinite(elev[i])]]
    if len(e) >= 30:
        sel[s] = e

order = sorted(sel, key=lambda s: np.median(sel[s]))
ncol  = 5
nrow  = int(np.ceil(len(order) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2*ncol, 2.4*nrow), sharex=True)
for ax, s in zip(axes.ravel(), order):
    sns.histplot(sel[s], bins=np.arange(0, 91, 5), ax=ax, color='steelblue')
    ax.axvline(np.median(sel[s]), color='firebrick', ls='--', lw=1)
    ax.set_title(f'syl {s}  n={len(sel[s])}  med={np.median(sel[s]):.0f}°', fontsize=9)
    ax.set_xlabel(''); ax.set_ylabel('')
for ax in axes.ravel()[len(order):]:
    ax.set_axis_off()
fig.supxlabel('trunk elevation (deg)')
fig.tight_layout()