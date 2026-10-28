# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:26:30 2026

@author: jlee629
"""

# %% Testing the model fidelity: Label stability across Gibbs iterations.
    
prj_name ='/media/jlee629/D/KeypointMoseq/projects/feb_may/2026_09_11-14_24_57'

from sklearn.metrics import adjusted_rand_score
with h5py.File(f'{prj_name}/checkpoint.h5','r') as f:
    its = sorted(int(i) for i in f['model_snapshots'])[-10:]
    zs  = [f[f'model_snapshots/{i}/states/z'][()].ravel() for i in its]
print([round(adjusted_rand_score(a, b), 3) for a, b in zip(zs, zs[1:])])
