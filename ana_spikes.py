#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:23:58 2026

@author: jlee629
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# if not hasattr(np, 'bool8'):
#     np.bool8 = np.bool_ # Or np.bool
import os

# %%

data_dir = '/media/jlee629/D/Neural Data/Phy'
session_name = 'TS_05_11'
folder_name = 'kilosort4'

fpath1 = os.path.join(data_dir,session_name,folder_name)



fs = 30000.0

spike_times    = np.load(os.path.join(fpath1,'spike_times.npy')).ravel()      # samples
spike_clusters = np.load(os.path.join(fpath1,'spike_clusters.npy')).ravel()

g = pd.read_csv(os.path.join(fpath1,'cluster_group.tsv'), sep='\t')
g.columns = [c.strip().lower() for c in g.columns]
labels = dict(zip(g['cluster_id'].astype(int),
                  g['group'].fillna('unsorted').str.strip().str.lower()))

type_map = {'good': 'single', 'mua': 'multi'}
total_s = spike_times.max() / fs

units = {}
for cid, lab in labels.items():
    if lab not in type_map:
        continue                                   # drops noise and unsorted
    st = spike_times[spike_clusters == cid]
    units[cid] = {
        'spike_times': st,
        'spike_times_s': st / fs,
        'type': type_map[lab],
        'n_spikes': st.size,
        'rate_hz': st.size / total_s,
    }
    
    
# %%

bin_s = 0.2                                  # 40 ms = one video frame at 25 fps
n_bins = int(np.ceil(total_s / bin_s))
edges  = np.arange(n_bins + 1) * bin_s         # seconds

unit_ids = sorted(units)                       # row order — keep this alongside the matrix
# counts = np.zeros((len(unit_ids), n_bins), dtype=np.int32)

# for i, cid in enumerate(unit_ids):
#     counts[i] = np.histogram(units[cid]['spike_times_s'], bins=edges)[0]

keep = np.isin(spike_clusters, unit_ids)
row  = np.searchsorted(unit_ids, spike_clusters[keep])      # unit_ids must be sorted
col  = np.floor((spike_times[keep] / fs) / bin_s).astype(np.int64)
np.clip(col, 0, n_bins - 1, out=col)                        # last spike lands in-range

counts = np.zeros((len(unit_ids), n_bins), dtype=np.int32)
np.add.at(counts, (row, col), 1)


rates = counts / bin_s                         # Hz
unit_types = np.array([units[cid]['type'] for cid in unit_ids])    



    
# %%

from rastermap import Rastermap
from scipy.stats import zscore
# spks is neurons by time

# fit rastermap

spks = zscore(rates,axis = 1)
model = Rastermap(n_clusters=None, # None turns off clustering and sorts single neurons 
                  n_PCs=10, # use fewer PCs than neurons
                  locality=0.5, # some locality in sorting (this is a value from 0-1)
                  time_lag_window=10, # use future timepoints to compute correlation
                  grid_upsample=0, # 0 turns off upsampling since we're using single neurons
                ).fit(spks)
y = model.embedding # neurons x 1
isort = model.isort



# visualize binning over neurons
X_embedding = model.X_embedding

# plot
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")