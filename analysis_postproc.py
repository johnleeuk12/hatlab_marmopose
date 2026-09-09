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
# import umap
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

# import marmo as mm

logger = logging.getLogger(__name__)

# %% helper functions

import marmo.io as io
from marmo.instances import get_syllable_instances
import marmo.transitions as transitions


# %% Init
# project_dir = "/home/jlee629/kpmoseq/projects/feb_may"
# project_dir = 'D:/KeypointMoseq/projects/feb_may'
project_dir = '/media/jlee629/D/KeypointMoseq/projects/feb_may'
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
coordinates, confidences, bodyparts =io.marmopose_loader(fpath1,'track_all')


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

coordinates, _, _ =io.marmopose_loader(os.path.join(project_dir,'combined_centroid.h5'),'centroid_all')
cent_all = coordinates['centroid_all']

# %% derivative of centroid position

# C_k = results['track1']['centroid']
S = np.zeros((len(cent_all),1))
for t in np.arange(1,len(cent_all)-1):
    S[t] = np.linalg.norm(cent_all[t+1,:]-cent_all[t-1,:])/2




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


# %% load keypoint syllables

model_name = '2026_09_09-12_29_41'

# model_name = '2026_08_19-10_33_58'

results = io.load_results(project_dir, model_name)
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
trans_combined = transitions.get_transition_matrix(comp_df, normalize='bigram')
comp_df.to_pickle(os.path.join(project_dir, 'comp_df.pkl'))
comp_df.to_csv(os.path.join(project_dir, 'comp_df.csv'), index=False)

# sns.histplot(data = instances_df['syllable'],stat = 'percent')
# sns.histplot(data = comp_df['duration_frames'],binwidth = 1)


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

# %% Additional analyses start here

import marmo.dispersion as dis
import marmo.posture as ps
import marmo.clustering as mcl

# d = comp_df[~comp_df.syllable.isin((99, 199))]
# order = d.groupby('syllable').duration_frames.median().sort_values().index

# fig, ax = plt.subplots(figsize=(12, 5))
# ax.boxplot([d.loc[d.syllable == s, 'duration_frames'] for s in order],
#            labels=[str(s) for s in order], showfliers=False)
# ax.set_xlabel('syllable'); ax.set_ylabel('duration (frames)')
# ax.set_title('Syllable duration, ordered by median')
# plt.tight_layout()



res = dis.keypoint_dispersion(combined_arr, comp_df, bodyparts_real, exclude_syllables=(99,12))
dis.plot_keypoint_dispersion(res,font_size_multiplier=1.5)
#
# print(res['summary'].to_string())
# print(res['ego'].round(3).to_string())
#
# # fair cross-syllable comparison: fixed 25-frame window from onset
res_w = dis.keypoint_dispersion(combined_arr, comp_df, bodyparts_real,
                            window_frames=5, exclude_syllables=(99,12))

dis.plot_keypoint_dispersion(res_w,font_size_multiplier=1.5)


ang = ps.frame_angles(combined_arr)
I   = ps.instance_posture(combined_arr, comp_df, ang)
STATIONARY = [0,1,2,6,7,8,9,14,15,16,17,18,19,20,21,22,23]
# A. per-syllable bimodality
tr = ps.find_syllable_troughs(I, syllables=STATIONARY, min_inst=20)
ps.plot_syllable_troughs(I, tr)
#
# ps.plot_by_syllable(I,'flexion', syllables = STATIONARY)





# %%
feats = mcl.instance_pose_features(combined_arr, comp_df, STATIONARY)
emb   = mcl.fit_umap(feats, n_components=3,n_neighbors= 25)
cl, ct, Zi = mcl.cluster_instances(feats, k=6)
#
# mcl.plot_umap_projections(emb, feats)
# mcl.plot_umap_projections(emb, feats, labels=cl, label_name='cluster')
mcl.plot_umap_3d(emb, feats, labels=cl, label_name='cluster')


comp_df_post = mcl.relabel_from_clusters(comp_df, feats, cl, stationary=STATIONARY,
                                     n_frames=len(combined_arr))
# save_comp_df_post(comp_df_post, os.path.join(project_dir,'comp_df_post.csv'))


ang = ps.frame_angles(combined_arr)
I   = ps.instance_posture(combined_arr, comp_df_post, ang)
# STATIONARY = [0,1,2,5,9,10,11,12,14,16,17,23,24,25,26]
# A. per-syllable bimodality
tr = ps.find_syllable_troughs(I, syllables=[101,102,103,104,105,106], min_inst=20)
ps.plot_syllable_troughs(I, tr)
#

# comp_df_post.to_csv(os.path.join(project_dir, 'comp_df_post_2.csv'), index=False)



# %%
# =============================================================================
# Usage -- drop-in for the Figure 1 section of analysis_postproc.py
# =============================================================================
# from marmo.clustering import fit_umap

 
from marmo.transitions import get_transition_matrix
from marmo.plotfigs import (instance_velocity, build_4d_space,
                            syllable_centroids, plot_figure1,
                            plot_transition_graph)
 
SOURCE_DF = comp_df_post          # or comp_df
EXCLUDE   = (99, 199)
MIN_COUNT = 5                     # cells below this are masked, not zeroed
syls      = sorted(s for s in SOURCE_DF.syllable.unique() if s not in EXCLUDE)
NO_EXTREM = ('tailmid', 'tailend', 'lefthand', 'righthand', 'leftfoot', 'rightfoot')
# --- instance-median feature space ---
feats = mcl.instance_pose_features(combined_arr, SOURCE_DF, syls,use_quality_filter=False)
# emb   = mcl.fit_umap(feats, n_components=3)
# cl, ct, Zi = mcl.cluster_instances(feats, k=4)
# mcl.plot_umap_projections(emb, feats)


vel   = instance_velocity(combined_arr, feats['meta'])
 
emb4d, emb3       = build_4d_space(feats['X'], vel, n_neighbors=15)
syl_include, C, S = syllable_centroids(emb4d, feats['labels'], min_inst=3)
 
# --- transitions ---
seq = SOURCE_DF[~SOURCE_DF.syllable.isin(EXCLUDE)]
 
trans_prob = get_transition_matrix(seq, normalize='bigram')
 
trans_enrich = get_transition_matrix(seq, normalize='enrichment',
                                     min_count=MIN_COUNT, drop_self=True)
 
# --- figures ---
fig, Z, D = plot_figure1(syl_include, C, S, feats['labels'], trans_prob)
 
fig2, G = plot_transition_graph(syl_include, feats['labels'], trans_prob,
                                top_n=50)

# %%

from marmo.geometry import ego_frame, segment_quality
from marmo.config import TAIL, SPINE, MIN_FRAMES, TRUNK_THR
 
 
def why_dropped(pts, comp_df, syllables,
                min_frames=MIN_FRAMES, trunk_thr=TRUNK_THR,
                use_quality_filter=True):
    """
    Per-syllable instance counts surviving each stage.
 
    Columns
    -------
    n_total     instances with this label in comp_df
    in_range    ...ending inside the array (only bites if pts is an excerpt)
    ego_ok      ...with at least min_frames where the egocentric frame exists,
                i.e. tailbase/spinemid/neck present
    qual_ok     ...also passing the trunk-length quality filter
    lost_to     the stage that removed the rest
 
    A syllable reaching qual_ok = 0 is absent from feats['labels'].
    """
    T = len(pts)
    ego = ego_frame(pts)
    ok_ego = np.isfinite(ego[:, SPINE, 0])
    if use_quality_filter:
        ok_qual = ok_ego & (segment_quality(pts, TAIL, SPINE) < trunk_thr)
    else:
        ok_qual = ok_ego
    rows = []
    for s in sorted(syllables):
        g = comp_df[comp_df.syllable == s]
        gi = g[(g.start_frame + g.duration_frames) <= T]
        n_ego = n_qual = 0
        for r in gi.itertuples(index=False):
            sl = slice(r.start_frame, r.start_frame + r.duration_frames)
            if ok_ego[sl].sum() >= min_frames:
                n_ego += 1
            if ok_qual[sl].sum() >= min_frames:
                n_qual += 1
        lost = ('-' if n_qual else
                'out of range' if len(gi) == 0 else
                'no egocentric frame' if n_ego == 0 else
                'quality filter')
        rows.append({'syllable': s, 'n_total': len(g), 'in_range': len(gi),
                     'ego_ok': n_ego, 'qual_ok': n_qual,
                     'dur_med': g.duration_frames.median(), 'lost_to': lost})
    t = pd.DataFrame(rows).set_index('syllable')
    dropped = t.index[t.qual_ok == 0].tolist()
    print(t.to_string())
    print(f'\nabsent from feats: {dropped}')
    if dropped:
        print('  median duration of dropped syllables: '
              f'{t.loc[dropped, "dur_med"].median():.0f} frames  '
              f'(kept: {t.loc[t.qual_ok > 0, "dur_med"].median():.0f})')
        print('  short syllables fail most easily: min_frames is an ABSOLUTE count,')
        print('  so a 4-frame instance needs 3 of 4 frames to pass while a 40-frame')
        print('  one needs 3 of 40.')
    return t

why_dropped(combined_arr, comp_df_post, syls)

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
                           exclude=(99,199), show_expected=True):
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
ctx = plot_context_sequences(comp_df_post, targets=(3,4), n=2, top_k=10)

# ranked by enrichment rather than raw count
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
# comp_df.to_csv(os.path.join(project_dir, 'comp_df_v2.csv'), index=False)


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


# # %% # %%
# # =============================================================================
# # 4D syllable centroids: UMAP (x,y,z) + normalized velocity
# # =============================================================================

# from scipy.spatial.distance import cdist, squareform
# from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
# import matplotlib.gridspec as gridspec

# # --- Exclude syllable 99 (low-frequency catch-all) ---
# syll_include = sorted([s for s in np.unique(syllables_valid) if s != 99])
# n_syl        = len(syll_include)
# syl_labels   = [str(s) for s in syll_include]

# # Color map: consistent color per syllable across all plots
# syl_colors = {s: cc.glasbey[i] for i, s in enumerate(syll_include)}

# # --- Extract velocity for valid frames (last column of T) ---
# velocity_valid = T[valid_mask, -1].reshape(-1, 1)
# v_min, v_max   = np.nanmin(velocity_valid), np.nanmax(velocity_valid)
# velocity_norm  = (velocity_valid - v_min) / (v_max - v_min + 1e-8)

# # --- Build 4D embedding: (n_valid_frames, 4) ---
# embedding_4d = np.concatenate([embedding_valid, velocity_norm * 2], axis=1)

# # --- Per-syllable centroids and spread ---
# syllable_centroids = {}
# syllable_spread    = {}

# for syl in syll_include:
#     mask = syllables_valid == syl
#     if mask.sum() == 0:
#         continue
#     pts = embedding_4d[mask]
#     syllable_centroids[syl] = np.nanmean(pts, axis=0)  # (4,)
#     syllable_spread[syl]    = np.nanstd(pts,  axis=0)  # (4,)

# centroid_matrix = np.stack([syllable_centroids[s] for s in syll_include], axis=0)
# centroid_dist   = cdist(centroid_matrix, centroid_matrix, metric='euclidean')

# # --- Hierarchical clustering ---
# Z              = linkage(squareform(centroid_dist, checks=False), method='ward')
# ordered_leaves = leaves_list(Z)
# labels_ordered = [syl_labels[i] for i in ordered_leaves]
# dist_reordered = centroid_dist[np.ix_(ordered_leaves, ordered_leaves)]
# syll_ordered   = [syll_include[i] for i in ordered_leaves]

# # --- Real syllable instance frequency from comp_df (excludes 99) ---
# freq_counts  = comp_df[comp_df['syllable'] != 99]['syllable'].value_counts()
# freq_total   = freq_counts.sum()
# freq_dict    = {s: freq_counts.get(s, 0) / freq_total for s in syll_include}

# # --- Transition matrix from comp_df ---
# N_STATES = 100

# def get_transition_matrix(inst_df, n_states=N_STATES, normalize='bigram'):
#     syl       = inst_df['syllable'].values
#     trans_mat = np.zeros((n_states, n_states), dtype=float)
#     np.add.at(trans_mat, (syl[:-1], syl[1:]), 1)
#     if normalize == 'bigram':
#         total = trans_mat.sum()
#         if total > 0:
#             trans_mat /= total
#     elif normalize == 'rows':
#         row_sums = trans_mat.sum(axis=1, keepdims=True)
#         row_sums[row_sums == 0] = 1
#         trans_mat /= row_sums
#     return trans_mat

# trans_combined = get_transition_matrix(comp_df[comp_df['syllable'] != 99], normalize='bigram')

# # Subset transition matrix to syll_include only
# syl_idx        = np.array(syll_include)
# trans_sub      = trans_combined[np.ix_(syl_idx, syl_idx)]
# trans_reordered = trans_sub[np.ix_(ordered_leaves, ordered_leaves)]

# # # Per-session transition matrices
# # trans_per_session = {}
# # for i, entry in enumerate(index):
# #     sess_start = entry['track1']['start']
# #     sess_end   = entry['track2']['end']
# #     sess_df    = comp_df[
# #         (comp_df['start_frame'] >= sess_start) &
# #         (comp_df['start_frame'] <= sess_end)  &
# #         (comp_df['syllable']    != 99)
# #     ].reset_index(drop=True)
# #     trans_per_session[f'session{i}'] = get_transition_matrix(sess_df, normalize='bigram')
# #     print(f"session{i}: {len(sess_df)} instances")

# # Cluster transition matrix independently
# # Use row+col profiles as fingerprint for each syllable
# trans_sym      = (trans_sub + trans_sub.T) / 2
# Z_trans        = linkage(squareform(cdist(trans_sym, trans_sym, metric='euclidean'), 
#                                     checks=False), method='ward')
# ordered_leaves2 = leaves_list(Z_trans)
# labels_ordered2 = [syl_labels[i] for i in ordered_leaves2]
# trans_reordered2 = trans_sub[np.ix_(ordered_leaves2, ordered_leaves2)]

# # %%
# # =============================================================================
# # Figure 1: Dendrogram + Spread + Distance heatmap + Transition heatmap
# # =============================================================================

# fig1 = plt.figure(figsize=(24, 14))
# fig1.suptitle('Syllable similarity analysis (4D: UMAP + velocity)', fontsize=13)

# gs1 = gridspec.GridSpec(2, 3, figure=fig1,
#                          width_ratios=[2.5, 2.5, 1.5],
#                          height_ratios=[1, 2],
#                          hspace=0.35, wspace=0.4)

# # Row 0, Col 0: Dendrogram (unchanged)
# ax_dend = fig1.add_subplot(gs1[0, 0])
# dendrogram(
#     Z,
#     labels=syl_labels,
#     ax=ax_dend,
#     color_threshold=0.7 * max(Z[:, 2]),
#     leaf_font_size=9,
#     above_threshold_color='grey'
# )
# ax_dend.set_title('Hierarchical clustering (Ward)', fontsize=10)
# ax_dend.set_ylabel('Distance', fontsize=9)
# ax_dend.set_xlabel('Syllable', fontsize=9)
# ax_dend.spines['top'].set_visible(False)
# ax_dend.spines['right'].set_visible(False)
 
# # --- Row 0, Col 1-2: Syllable frequency bar chart ---
# ax_freq = fig1.add_subplot(gs1[0, 1:])
# freq_vals    = [freq_dict[s] for s in syll_include]
# freq_colors  = [syl_colors[s] for s in syll_include]
# ax_freq.bar(range(n_syl), freq_vals, color=freq_colors, edgecolor='none')
# ax_freq.set_xticks(range(n_syl))
# ax_freq.set_xticklabels(syl_labels, fontsize=8, rotation=90)
# ax_freq.set_ylabel('Instance frequency', fontsize=9)
# ax_freq.set_title('Syllable frequency (instance-based, excl. 99)', fontsize=10)
# ax_freq.spines['top'].set_visible(False)
# ax_freq.spines['right'].set_visible(False)

# # Row 1, Col 0: Distance heatmap — reordered by centroid clustering
# ax_heat = fig1.add_subplot(gs1[1, 0])
# im_heat = ax_heat.imshow(dist_reordered, aspect='equal', cmap='viridis_r')
# ax_heat.set_xticks(range(n_syl))
# ax_heat.set_xticklabels(labels_ordered, fontsize=7, rotation=90)
# ax_heat.set_yticks(range(n_syl))
# ax_heat.set_yticklabels(labels_ordered, fontsize=7)
# ax_heat.set_title('Centroid distance\n(ordered by centroid clustering)', fontsize=10)
# plt.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)

# # Row 1, Col 1: Transition heatmap — reordered by transition clustering
# ax_trans = fig1.add_subplot(gs1[1, 1])
# im_trans = ax_trans.imshow(trans_reordered2, aspect='equal', cmap='hot_r',
#                             vmin=0, vmax=np.percentile(trans_reordered2[trans_reordered2 > 0], 95))
# ax_trans.set_xticks(range(n_syl))
# ax_trans.set_xticklabels(labels_ordered2, fontsize=7, rotation=90)
# ax_trans.set_yticks(range(n_syl))
# ax_trans.set_yticklabels(labels_ordered2, fontsize=7)
# ax_trans.set_title('Transition matrix\n(ordered by transition clustering)', fontsize=10)
# ax_trans.set_xlabel('to', fontsize=8)
# ax_trans.set_ylabel('from', fontsize=8)
# plt.colorbar(im_trans, ax=ax_trans, fraction=0.046, pad=0.04)
    
# # --- Row 1, Col 2: Cluster spread heatmap ---
# ax_spread = fig1.add_subplot(gs1[1, 2])
# spread_matrix = np.stack([syllable_spread[s] for s in syll_include], axis=0)
# im_spread = ax_spread.imshow(spread_matrix, aspect='auto', cmap='YlOrRd')
# ax_spread.set_xticks(range(4))
# ax_spread.set_xticklabels(['UMAP-x', 'UMAP-y', 'UMAP-z', 'velocity'], fontsize=8, rotation=30)
# ax_spread.set_yticks(range(n_syl))
# ax_spread.set_yticklabels(syl_labels, fontsize=8)
# ax_spread.set_title('Cluster spread (std dev)', fontsize=10)
# plt.colorbar(im_spread, ax=ax_spread, fraction=0.046, pad=0.04)

# plt.show()