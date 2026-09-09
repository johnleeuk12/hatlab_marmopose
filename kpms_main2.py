# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import keypoint_moseq as kpms
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py


# %%
project_dir = "/home/jlee629/kpmoseq/june_v4"
kpms.generate_config(project_dir)

# config = kpms.load_config(project_dir)

config = lambda: kpms.load_config(project_dir)

sleap_file = "original_new.h5"  # any .slp or .h5 file with predictions for a single video
kpms.setup_project(project_dir, sleap_file=sleap_file)


bodyparts = ['head', 'leftear', 'rightear', 'neck', 
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
    

video_dir = os.path.join(project_dir, 'videos_raw')

kpms.setup_project(
    project_dir,
    video_dir=video_dir,
    bodyparts=bodyparts,
    skeleton=skeleton,overwrite=True)

kpms.update_config(
    project_dir,
    anterior_bodyparts = ['neck'],
    posterior_bodyparts = ['tailbase'],
    use_bodyparts = ['head', 'leftear', 'rightear','neck', 
                 'leftelbow', 'rightelbow', 'lefthand', 'righthand', 
                 'spinemid', 'tailbase', 'leftknee', 'rightknee', 
                 'leftfoot', 'rightfoot'],
    fps = 25)


fpath = os.path.join(project_dir, 'original_new.h5')

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




# %%

coordinates, confidences, bodyparts =marmopose_loader(fpath,"track1")

kpms.update_config(project_dir, outlier_scale_factor=2.0)

coordinates, confidences = kpms.outlier_removal(
    coordinates,
    confidences,
    project_dir,
    overwrite=True,
    **config()
)

data, metadata = kpms.format_data(coordinates, confidences, **config())



# kpms.noise_calibration(project_dir, coordinates, confidences, **config())


pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)


kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())




# %% model
kpms.update_config(
    project_dir,
    sigmasq_loc=kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config()["fps"])
)

# initialize the model
model = kpms.init_model(data, pca=pca, **config())

num_ar_iters = 50

model = kpms.update_hypparams(model, kappa=1e7)


model, model_name = kpms.fit_model(
    model, data, metadata, project_dir, ar_only=True, num_iters=num_ar_iters
)



# %% fitting full model

model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=num_ar_iters
)

# modify kappa to maintain the desired syllable time-scale
model = kpms.update_hypparams(model, kappa=1e7)

# run fitting for an additional 500 iters
model = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    model_name,
    ar_only=False,
    start_iter=current_iter,
    num_iters=current_iter + 500,
    parallel_message_passing=False,
)[0]

# %% 

model_name = '2026_07_17-11_58_04'


# %%
kpms.reindex_syllables_in_checkpoint(project_dir, model_name);


# load the most recent model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

# extract and save results
# results = kpms.extract_results(model, metadata, project_dir, model_name)

# load saved results
results = kpms.load_results(project_dir, model_name)
# %%
kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config(),
                               min_frequency = 0.001,
                               sampling_options={"mode": "random", "n_neighbors": 10})


# %%

kpms.generate_grid_movies(
   results,
   project_dir,
   model_name,
   coordinates=coordinates,
   keypoints_only=True,
   keypoints_scale=1,
   use_dims=[0,1], # controls projection plane
   min_frequency = 0.001,
   **config());


# %%

kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config())

metric="cosine",
pre=0.167,
post=0.5,
min_frequency=0.005,
min_duration=3,
bodyparts=None,
use_bodyparts=None,
density_sample=False,
sampling_options={"n_neighbors": 50},
figsize=(6, 3),
fps=None,


distances, syllable_ixs = kpms.syllable_similarity(
        coordinates,
        results,
        metric,
        pre,
        post,
        min_frequency,
        min_duration,
        bodyparts,
        use_bodyparts,
        density_sample,
        sampling_options,
    )






