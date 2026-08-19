#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py

Preprocessing pipeline for 3D triangulated marmoset pose data.

Pipeline (per keypoint, per track):
    1. Load optimized.h5 (EKS-smoothed triangulated data)
    2. Remove isolated valid frames (1-2 valid frames surrounded by >=10 NaN frames on each side)
    3. Linear interpolation of short NaN gaps (<=2 consecutive NaN frames between valid frames)
    4. Centroid-based outlier removal (keypoints >95th percentile distance from centroid)
    5. Per-frame time segmentation (NaN entire frame if keypoints disagree across tracks)
    6. NaN padding at session boundaries
    7. Save per-session preprocessed file
    8. Concatenate all sessions and save combined file

Outputs:
    - Per-session: <session_dir>/preprocessed.h5  (track1, track2)
    - Combined:    <output_dir>/combined.h5        (track1, track2)

Author: jlee629
"""

import numpy as np
import h5py
import os
import logging
from pathlib import Path

import seaborn as sns


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


#%%
# =============================================================================
# Configuration
# =============================================================================

# Root output directory for the combined file
OUTPUT_DIR = "/home/jlee629/kpmoseq/projects/feb_may"

# List of session folders to process (each must contain optimized.h5)
SESSION_DIRS = [
    "/media/jlee629/D/MarmoPose/projects/pair-02_13_s1_v1/points_3d",
    "/media/jlee629/D/MarmoPose/projects/pair-02_13_s3_v1/points_3d",
    "/media/jlee629/D/MarmoPose/projects/pair-02_16_s1_v1/points_3d",
    "/media/jlee629/D/MarmoPose/projects/pair-05_06_s2_v2/points_3d",
    "/media/jlee629/D/MarmoPose/projects/pair-05_11_s1_v2/points_3d",
    # Add more session folders here
]

# SESSION_DIRS = [
#     "/media/jlee629/D/MarmoPose/projects/pair-02_13_s1_v1/points_3d"
# ]

BODYPARTS = [
    'head', 'leftear', 'rightear', 'neck',
    'leftelbow', 'rightelbow', 'lefthand', 'righthand',
    'spinemid', 'tailbase', 'leftknee', 'rightknee',
    'leftfoot', 'rightfoot', 'tailmid', 'tailend'
]

TRACKS = ['track1', 'track2']

# Preprocessing parameters
# input
OPTIMIZED_FNAME     = 'dae_smoothed.h5'

# output
PREPROCESSED_FNAME  = 'preprocessed.h5'
COMBINED_FNAME      = 'combined.h5'
COMBINED_INDEX_FNAME = 'combined_index.npy'
CENTROID_FNAME      = 'centroid.h5'
COMBINED_CENTROID_FNAME = 'combined_centroid.h5'


ISOLATED_VALID_MAX  = 2     # max consecutive valid frames to consider "isolated"
ISOLATED_NAN_MIN    = 10    # min NaN frames on each side to trigger removal
INTERP_MAX_GAP      = 2     # max consecutive NaN frames to interpolate
OUTLIER_PERCENTILE  = 99    # percentile threshold for centroid-distance outlier removal
N_BOUNDARY_FRAMES   = 5     # NaN frames inserted at session boundaries
SMOOTH_WINDOW       = 5     # median smoothing window size (frames), must be odd

#%%
# =============================================================================
# I/O helpers
# =============================================================================

def load_optimized_h5(session_dir: str, track: str) -> np.ndarray:
    """
    Load EKS-optimized 3D keypoints for a single track.

    Returns
    -------
    coords : np.ndarray, shape (n_frames, n_bodyparts, 3)
        NaN where keypoints are missing.
    """
    fpath = os.path.join(session_dir, OPTIMIZED_FNAME)
    with h5py.File(fpath, 'r') as f:
        if track not in f:
            raise KeyError(f"Track '{track}' not found in {fpath}. "
                           f"Available keys: {list(f.keys())}")
        coords = f[track][()].astype(np.float64)
    logger.info(f"Loaded {track} from {fpath}, shape {coords.shape}")
    return coords


def save_preprocessed_h5(data_dict: dict, out_path: Path) -> None:
    """
    Save preprocessed tracks to an HDF5 file.

    Parameters
    ----------
    data_dict : dict
        Keys are track names, values are np.ndarray (n_frames, n_bodyparts, 3).
    out_path : Path
    """
    with h5py.File(out_path, 'w') as f:
        for track, data in data_dict.items():
            f.create_dataset(track, data=data.astype(np.float32))
    logger.info(f"Saved preprocessed data to {out_path}")


def load_preprocessed_h5(fpath: Path) -> dict:
    """
    Load preprocessed tracks from an HDF5 file.

    Returns
    -------
    dict : track_name -> np.ndarray (n_frames, n_bodyparts, 3)
    """
    data = {}
    with h5py.File(fpath, 'r') as f:
        for key in sorted(f.keys()):
            data[key] = f[key][()].astype(np.float64)
    return data

# %%

# =============================================================================
# Step 2: Remove isolated valid frames
# =============================================================================

def remove_isolated_valid_frames(arr: np.ndarray,
                                  max_valid: int = ISOLATED_VALID_MAX,
                                  min_nan_surround: int = ISOLATED_NAN_MIN
                                  ) -> np.ndarray:
    """
    Per-keypoint: NaN out runs of valid (non-NaN) frames that are too short
    to be trusted, defined as <=max_valid consecutive valid frames surrounded
    by >=min_nan_surround NaN frames on each side.

    Parameters
    ----------
    arr : np.ndarray, shape (n_frames, n_bodyparts, 3)
    max_valid : int
    min_nan_surround : int

    Returns
    -------
    arr : np.ndarray, same shape, with isolated valid runs NaN'd out
    """
    arr = arr.copy()
    n_frames, n_bp, _ = arr.shape

    for b in range(n_bp):
        valid = ~np.isnan(arr[:, b, 0])  # True where frame is valid

        t = 0
        while t < n_frames:
            if valid[t]:
                # Find end of this valid run
                run_start = t
                while t < n_frames and valid[t]:
                    t += 1
                run_end = t  # exclusive

                run_len = run_end - run_start

                if run_len <= max_valid:
                    # Count NaN frames before the run
                    nan_before = 0
                    for k in range(run_start - 1, -1, -1):
                        if not valid[k]:
                            nan_before += 1
                        else:
                            break

                    # Count NaN frames after the run
                    nan_after = 0
                    for k in range(run_end, n_frames):
                        if not valid[k]:
                            nan_after += 1
                        else:
                            break

                    if nan_before >= min_nan_surround and nan_after >= min_nan_surround:
                        arr[run_start:run_end, b, :] = np.nan
            else:
                t += 1

    return arr


# =============================================================================
# Step 3: Linear interpolation of short NaN gaps
# =============================================================================

def interpolate_short_gaps(arr: np.ndarray,
                            max_gap: int = INTERP_MAX_GAP
                            ) -> np.ndarray:
    """
    Per-keypoint, per-axis: linearly interpolate NaN gaps of <=max_gap
    consecutive frames that are surrounded by valid frames on both sides.

    Parameters
    ----------
    arr : np.ndarray, shape (n_frames, n_bodyparts, 3)
    max_gap : int

    Returns
    -------
    arr : np.ndarray, same shape, with short gaps filled
    """
    arr = arr.copy()
    n_frames, n_bp, n_ax = arr.shape

    for b in range(n_bp):
        for ax in range(n_ax):
            signal = arr[:, b, ax]
            nan_mask = np.isnan(signal)

            t = 0
            while t < n_frames:
                if nan_mask[t]:
                    # Find end of this NaN run
                    gap_start = t
                    while t < n_frames and nan_mask[t]:
                        t += 1
                    gap_end = t  # exclusive

                    gap_len = gap_end - gap_start

                    # Only interpolate if gap is short enough and surrounded
                    # by valid frames on both sides
                    if gap_len <= max_gap and gap_start > 0 and gap_end < n_frames:
                        v_before = signal[gap_start - 1]
                        v_after  = signal[gap_end]

                        if not np.isnan(v_before) and not np.isnan(v_after):
                            # Linear interpolation
                            for i, idx in enumerate(range(gap_start, gap_end)):
                                alpha = (i + 1) / (gap_len + 1)
                                signal[idx] = v_before + alpha * (v_after - v_before)

                    arr[:, b, ax] = signal
                else:
                    t += 1

    return arr

def median_smooth(arr: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """
    Per-keypoint, per-axis: apply a sliding nanmedian filter.
    Edges are padded with NaN (no reflection or repetition).

    Parameters
    ----------
    arr : np.ndarray, shape (n_frames, n_bodyparts, 3)
    window : int, must be odd

    Returns
    -------
    arr : np.ndarray, same shape
    """
    arr = arr.copy()
    n_frames, n_bp, n_ax = arr.shape
    pad = window // 2

    for b in range(n_bp):
        for ax in range(n_ax):
            signal = arr[:, b, ax]
            swindow = np.lib.stride_tricks.sliding_window_view(signal, window)
            smoothed = np.nanmedian(swindow, axis=1)
            arr[pad:n_frames - pad, b, ax] = smoothed

    return arr


# =============================================================================
# Step 4: Centroid-based outlier removal
# =============================================================================

def remove_centroid_outliers(arr: np.ndarray,
                              percentile: float = OUTLIER_PERCENTILE
                              ) -> tuple:
    """
    Per-keypoint: NaN out frames where a keypoint's distance from the
    frame centroid exceeds the given percentile threshold.

    The centroid is computed per frame as the median across all keypoints.
    Spinemid (index 8) is excluded from outlier removal as it anchors the
    centroid itself.

    Also returns a smoothed centroid trajectory (window=5 median).

    Parameters
    ----------
    arr : np.ndarray, shape (n_frames, n_bodyparts, 3)
    percentile : float

    Returns
    -------
    arr_clean : np.ndarray, same shape
    centroid  : np.ndarray, shape (n_frames, 3), smoothed centroid trajectory
    """
    arr = arr.copy()
    n_frames, n_bp, _ = arr.shape

    # Compute per-frame centroid
    centroid_raw = np.full((n_frames, 3), np.nan)
    for t in range(n_frames):
        centroid_raw[t] = np.nanmedian(arr[t], axis=0)

    # Compute per-keypoint distance from centroid
    dist = np.full((n_frames, n_bp), np.nan)
    for t in range(n_frames):
        if not np.isnan(centroid_raw[t]).any():
            diff = arr[t] - centroid_raw[t]
            dist[t] = np.linalg.norm(diff, axis=1)
    
    # sns.histplot(data = dist[:,7], binwidth = 2)
    # NaN out outliers (skip spinemid = index 8)
    for b in range(n_bp):
        if b == 8:
            continue
        thresh = np.nanpercentile(dist[:, b], percentile)
        outlier_frames = np.where(dist[:, b] > thresh)[0]
        arr[outlier_frames, b, :] = np.nan

    # Smooth centroid with median window
    window = 5
    centroid_smooth = np.full((n_frames, 3), np.nan)
    for ax in range(3):
        swindow = np.lib.stride_tricks.sliding_window_view(centroid_raw[:, ax], window)
        smoothed = np.nanmedian(swindow, axis=1)
        # Pad edges to restore original length
        pad_before = window // 2
        pad_after  = n_frames - len(smoothed) - pad_before
        centroid_smooth[:, ax] = np.concatenate([
            np.full(pad_before, np.nan),
            smoothed,
            np.full(pad_after, np.nan)
        ])

    return arr, centroid_smooth


# =============================================================================
# Step 5: Per-frame time segmentation
# =============================================================================

def time_segmentation(arr: np.ndarray) -> np.ndarray:
    """
    Per-frame: if any keypoints are NaN but others are not (disagreement
    across keypoints within a frame), NaN the entire frame.

    Since optimized.h5 should have either all keypoints or none per frame,
    this step catches any disagreements introduced by per-keypoint processing.

    Parameters
    ----------
    arr : np.ndarray, shape (n_frames, n_bodyparts, 3)

    Returns
    -------
    arr : np.ndarray, same shape
    """
    arr = arr.copy()
    n_frames, n_bp, _ = arr.shape

    for t in range(n_frames):
        nan_counts = np.isnan(arr[t, :, 0])
        # if nan_counts.any() and not nan_counts.all():
        if np.sum(nan_counts[0:14]) > 6:
            arr[t, :, :] = np.nan

    n_removed = np.sum(
        [np.isnan(arr[t, 0, 0]) and not np.isnan(arr[t, :, 0]).all()
         for t in range(n_frames)]
    )
    logger.info(f"Time segmentation: {np.sum(np.isnan(arr[:, 0, 0]))} total NaN frames "
                f"after enforcement")
    return arr


# =============================================================================
# Step 6: NaN boundary padding
# =============================================================================

def add_boundary_padding(arr: np.ndarray,
                          n_frames: int = N_BOUNDARY_FRAMES
                          ) -> np.ndarray:
    """
    Prepend and append n_frames of NaN to signal a session boundary to MoSeq.

    Parameters
    ----------
    arr : np.ndarray, shape (T, n_bodyparts, 3)
    n_frames : int

    Returns
    -------
    np.ndarray, shape (T + 2*n_frames, n_bodyparts, 3)
    """
    if np.size(arr.shape) == 3:
        _, n_bp, n_ax = arr.shape
        pad = np.full((n_frames, n_bp, n_ax), np.nan)
        return np.concatenate([pad, arr, pad], axis=0)
    elif np.size(arr.shape) == 2:
        _, n_ax = arr.shape
        pad = np.full((n_frames, n_ax), np.nan)
        return np.concatenate([pad, arr, pad], axis=0)

# =============================================================================
# Full per-session preprocessing pipeline
# =============================================================================

def preprocess_session(session_dir: str) -> dict:
    """
    Run the full preprocessing pipeline on one session for all tracks.

    Parameters
    ----------
    session_dir : str
        Path to the session folder containing optimized.h5.

    Returns
    -------
    results : dict
        Keys are track names, values are dicts with:
            'data'     : np.ndarray (T_padded, n_bodyparts, 3)
            'centroid' : np.ndarray (T_padded, 3)
    """
    results = {}

    for track in TRACKS:
        logger.info(f"--- Processing {track} in {session_dir} ---")

        # Step 1: Load
        data = load_optimized_h5(session_dir, track)

        # Step 2: Remove isolated valid frames
        data = remove_isolated_valid_frames(data,
                                            max_valid=ISOLATED_VALID_MAX,
                                            min_nan_surround=ISOLATED_NAN_MIN)
        logger.info(f"  After isolated valid removal: "
                    f"{np.sum(np.isnan(data[:, 0, 0]))} NaN frames")

        # Step 3: Linear interpolation of short gaps
        data = interpolate_short_gaps(data, max_gap=INTERP_MAX_GAP)
        logger.info(f"  After interpolation: "
                    f"{np.sum(np.isnan(data[:, 0, 0]))} NaN frames")

        # Step 3.5: Median smoothing
        data = median_smooth(data, window=SMOOTH_WINDOW)

        # Step 4: Centroid-based outlier removal
        data, centroid = remove_centroid_outliers(data, percentile=OUTLIER_PERCENTILE)
        logger.info(f"  After outlier removal: "
                    f"{np.sum(np.isnan(data[:, 0, 0]))} NaN frames")

        # Step 5: Per-frame time segmentation
        data = time_segmentation(data)

        # Step 6: NaN boundary padding
        data     = add_boundary_padding(data,     n_frames=N_BOUNDARY_FRAMES)
        centroid = add_boundary_padding(centroid, n_frames=N_BOUNDARY_FRAMES)

        results[track] = {'data': data, 'centroid': centroid}
        logger.info(f"  Final shape: {data.shape}")

    return results


# %%

# =============================================================================
# Main: process all sessions, save per-session + combined
# =============================================================================

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

all_track_all       = []   # accumulates track_all across sessions for combined.h5
all_centroid_all    = []
session_index       = []   # accumulates index entries across sessions

cursor = 0           # tracks current position in combined track_all array

for session_dir in SESSION_DIRS:
    logger.info(f"========== Session: {session_dir} ==========")

    session_results = preprocess_session(session_dir)

    # Save per-session preprocessed.h5 (track1, track2 separately, unchanged)
    per_session_out = Path(session_dir) / PREPROCESSED_FNAME
    save_preprocessed_h5(
        {track: session_results[track]['data'] for track in TRACKS},
        per_session_out
    )
    # Save per-session centroid
    per_session_centroid_out = Path(session_dir) / CENTROID_FNAME
    save_preprocessed_h5(
        {track: session_results[track]['centroid'] for track in TRACKS},
        per_session_centroid_out
    )
    # Build track_all for this session: concatenate track1 and track2 along time axis
    # Each track is already padded with N_BOUNDARY_FRAMES on each side
    # An additional NaN boundary pad is inserted between track1 and track2
    t1 = session_results['track1']['data']
    t2 = session_results['track2']['data']
    sep = np.full((N_BOUNDARY_FRAMES, t1.shape[1], t1.shape[2]), np.nan)
    track_all_session = np.concatenate([t1, sep, t2], axis=0)
    
    
    c1 = session_results['track1']['centroid']
    c2 = session_results['track2']['centroid']
    sep_c = np.full((N_BOUNDARY_FRAMES, 3), np.nan)
    centroid_all_session = np.concatenate([c1, sep_c, c2], axis=0)
    all_centroid_all.append(centroid_all_session)
        
    
    # Record start/end indices for each track within the combined track_all array
    t1_start = cursor + N_BOUNDARY_FRAMES          # skip leading pad
    t1_end   = cursor + t1.shape[0] - N_BOUNDARY_FRAMES - 1  # skip trailing pad
    t2_start = cursor + t1.shape[0] + N_BOUNDARY_FRAMES + N_BOUNDARY_FRAMES  # skip sep + pad
    t2_end   = cursor + track_all_session.shape[0] - N_BOUNDARY_FRAMES - 1

    session_index.append({
        'session_dir': session_dir,
        'track1': {'start': t1_start, 'end': t1_end},
        'track2': {'start': t2_start, 'end': t2_end},
    })

    all_track_all.append(track_all_session)
    cursor += track_all_session.shape[0]

    logger.info(f"  track1 frames in combined: {t1_start} -> {t1_end}")
    logger.info(f"  track2 frames in combined: {t2_start} -> {t2_end}")

# Save combined.h5 with single track_all dataset
logger.info("========== Saving combined file ==========")
combined_arr = np.concatenate(all_track_all, axis=0)
save_preprocessed_h5({'track_all': combined_arr}, output_dir / COMBINED_FNAME)
logger.info(f"  track_all shape: {combined_arr.shape}")

# save combined centroid
combined_centroid_arr = np.concatenate(all_centroid_all, axis=0)
save_preprocessed_h5({'centroid_all': combined_centroid_arr}, output_dir / COMBINED_CENTROID_FNAME)
logger.info(f"  centroid_all shape: {combined_centroid_arr.shape}")


# Save index as .npy
# Structure: list of dicts, one per session
# Each dict: {'session_dir': str, 'track1': {'start': int, 'end': int}, 'track2': {...}}
np.save(output_dir / COMBINED_INDEX_FNAME, session_index)
logger.info(f"  Index saved to {output_dir / COMBINED_INDEX_FNAME}")
logger.info("Done.")





