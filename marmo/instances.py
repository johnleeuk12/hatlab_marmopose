"""
marmo.instances -- segmenting a frame-level syllable sequence into instances.
"""

import numpy as np
import pandas as pd



def get_syllable_instances(syllables_raw, combined_arr, fps=25, min_duration=3):
    """
    Segment a frame-level syllable sequence into instances.

    An instance is a run of one syllable label bounded by a label change or by
    a NaN frame, so no instance ever spans a session, track, or tracking gap.

    Runs shorter than `min_duration` frames are too brief to be real syllables
    and are absorbed into a neighbour:

      * into the PRECEDING label, if the run is contiguous with a valid
        preceding run (i.e. not separated from it by NaN);
      * otherwise into the FOLLOWING label, if there is a valid following run
        (this covers runs sitting immediately after a NaN gap, and runs at the
        very start of the array);
      * otherwise dropped, when the run is isolated between NaN stretches on
        both sides.

    Absorption is applied left to right so it cascades correctly: a chain of
    consecutive short runs all collapse into the label preceding the chain
    rather than each inheriting a label that is itself about to be replaced.
    After absorption, neighbouring runs that now share a label and are still
    frame-adjacent are merged into a single instance, so `duration_frames`
    reflects the merged extent.

    Parameters
    ----------
    syllables_raw : (T,) array of int syllable labels
    combined_arr  : (T, n_bodyparts, 3) coordinate array; a frame counts as
                    NaN if any keypoint coordinate is NaN
    fps           : int, accepted for signature compatibility
    min_duration  : int, runs with duration < min_duration are absorbed.
                    min_duration=2 absorbs single-frame runs only.

    Returns
    -------
    DataFrame with columns syllable, start_frame, duration_frames.
    `.attrs` carries n_absorbed_prev, n_absorbed_next, n_dropped.
    """
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
    print(f'  short runs (<{min_duration} frames): '
          f'{n_prev} absorbed into preceding, '
          f'{n_next} into following, {n_drop} dropped')
    return out