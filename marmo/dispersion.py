"""
marmo.dispersion -- per-keypoint movement magnitude by syllable, and the
resulting motion classification.

Two coordinate frames, because they answer different questions:

  world       raw coordinates. A flat, uniformly high profile across keypoints
              means the whole animal translated, which is what identifies
              locomotion. No egocentric measure can see this.
  egocentric  translation and body orientation removed, so only articulation
              remains -- "some parts move, others do not". World coordinates
              compress that contrast during locomotion, because the shared
              translation term is added to every keypoint equally.

IMPORTANT: dispersion is computed WITHIN each instance and then aggregated
across instances with a median. Pooling frames across instances would make
world dispersion measure where in the cage the animal was between bouts rather
than how much it moved during them. The world frame is fixed (cameras are
mounted to the cage), so different bouts genuinely happen at different
locations.

For the motion classification, the two axes are taken from the keypoints where
each is least contaminated: translation from the WORLD dispersion of trunk
keypoints (they barely articulate), articulation from the EGOCENTRIC dispersion
of distal keypoints. Mean-world versus mean-egocentric would be correlated by
construction, since world displacement is translation PLUS articulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from marmo.config import (TAIL, NECK, TRUNK_KEYPOINTS, DISTAL_KEYPOINTS,
                          DEFAULT_EXCLUDE_BODYPARTS, EXCLUDE_SYLLABLE)
from marmo.geometry import ego_frame, body_length, robust_dispersion


def keypoint_dispersion(combined_arr, comp_df, bodyparts,
                        origin_idx=TAIL, axis_idx=NECK,
                        min_frames_per_instance=5,
                        min_instances=3,
                        window_frames=None,
                        syllables=None,
                        exclude_syllables=(EXCLUDE_SYLLABLE,),
                        exclude_bodyparts=DEFAULT_EXCLUDE_BODYPARTS,
                        normalize_by_body_length=True):
    """
    Build (syllable x bodypart) dispersion tables in world and egocentric frames.

    Parameters
    ----------
    combined_arr : (T, K, 3) post-DAE coordinates
    comp_df      : instance table with syllable, start_frame, duration_frames
    bodyparts    : list of K names
    origin_idx, axis_idx : keypoint indices defining the body frame
                           (default 9=tailbase, 3=neck)
    min_frames_per_instance : a keypoint needs this many valid frames within an
                           instance to contribute
    min_instances : a cell needs this many contributing instances to be reported
    window_frames : if set, truncate every instance to this many frames from
                    onset and skip shorter ones. Use this when comparing across
                    syllables, because a longer instance gives a keypoint more
                    time to travel and so inflates its dispersion regardless of
                    how vigorously it moved.
    syllables : explicit list of syllables to INCLUDE. None means all present
                in comp_df. Use this when you already have a set in hand, e.g. a
                stationary list.
    exclude_syllables : labels to DROP. Accepts a single int or any iterable, so
                exclude_syllables=99 and exclude_syllables=(99, 7, 15) both work.
                Defaults to the low-frequency catch-all alone. Applied after
                `syllables`, so an explicit include list still has the catch-all
                removed unless you pass exclude_syllables=().

                Note this only drops syllables from the OUTPUT tables. It does
                not change the egocentric frame, the body-length scale, or the
                quality threshold, all of which are computed over the whole
                array -- so excluding a syllable never shifts the values
                reported for the ones you keep.
    exclude_bodyparts : names omitted from the output tables. Excluded at
                    output time rather than by slicing the array, so
                    origin_idx / axis_idx keep referring to the original
                    keypoint order.
    normalize_by_body_length : express both frames in body lengths rather than
                    mm, using a single median body length for the whole
                    recording. A constant is used rather than a per-frame
                    length so that noise in the neck/tailbase distance is not
                    injected into every keypoint.

    Returns
    -------
    dict with
      world, ego   : DataFrames, index=syllable, columns=bodyparts
      coverage     : DataFrame, fraction of instances that contributed per cell
      summary      : DataFrame, per-syllable n_instances / n_used / durations
    """
    K = combined_arr.shape[1]
    if len(bodyparts) != K:
        raise ValueError(f'bodyparts has {len(bodyparts)} names, '
                         f'combined_arr has {K} keypoints')

    exclude_bodyparts = set(exclude_bodyparts or ())
    keep = [(k, bp) for k, bp in enumerate(bodyparts) if bp not in exclude_bodyparts]
    keep_names = [bp for _, bp in keep]
    if exclude_bodyparts:
        print(f'excluding bodyparts: {sorted(exclude_bodyparts)}')

    scale = body_length(combined_arr, origin_idx, axis_idx) if normalize_by_body_length else 1.0
    unit = 'body lengths' if normalize_by_body_length else 'mm'
    print(f'body length = {scale:.1f}; dispersion reported in {unit}')

    print('building egocentric coordinates...')
    ego_all = ego_frame(combined_arr, origin_idx=origin_idx,
                        axis_from=origin_idx, axis_to=axis_idx)
    n_ego_bad = int(np.isnan(ego_all[:, origin_idx, 0]).sum())
    print(f'  {n_ego_bad} of {len(combined_arr)} frames had no usable body frame')

    df = comp_df
    if syllables is not None:
        df = df[df['syllable'].isin(list(syllables))]
    if exclude_syllables is not None:
        drop = ([exclude_syllables] if np.isscalar(exclude_syllables)
                else list(exclude_syllables))
        if drop:
            df = df[~df['syllable'].isin(drop)]
    syllables = sorted(df['syllable'].unique())
    if not syllables:
        raise ValueError('no syllables left after filtering')

    world_rows, ego_rows, cov_rows, summary = {}, {}, {}, []

    for syl in syllables:
        inst = df[df['syllable'] == syl]

        w_acc = [[] for _ in range(K)]
        e_acc = [[] for _ in range(K)]
        n_used = 0

        for r in inst.itertuples(index=False):
            s = int(r.start_frame)
            dur = int(r.duration_frames)
            if window_frames is not None:
                if dur < window_frames:
                    continue
                dur = window_frames
            e = s + dur

            Pw = combined_arr[s:e]
            Pe = ego_all[s:e]
            n_used += 1

            for k in range(K):
                dw = robust_dispersion(Pw[:, k, :], min_frames_per_instance)
                de = robust_dispersion(Pe[:, k, :], min_frames_per_instance)
                if np.isfinite(dw):
                    w_acc[k].append(dw)
                if np.isfinite(de):
                    e_acc[k].append(de)

        # Aggregate across instances with a median; gate on contributing count
        w_row, e_row, c_row = {}, {}, {}
        for k, bp in keep:
            nw = len(w_acc[k])
            ne = len(e_acc[k])
            w_row[bp] = np.median(w_acc[k]) / scale if nw >= min_instances else np.nan
            e_row[bp] = np.median(e_acc[k]) / scale if ne >= min_instances else np.nan
            c_row[bp] = nw / n_used if n_used else 0.0

        world_rows[syl], ego_rows[syl], cov_rows[syl] = w_row, e_row, c_row
        summary.append({
            'syllable':      syl,
            'n_instances':   len(inst),
            'n_used':        n_used,
            'median_dur_fr': float(inst['duration_frames'].median()),
            'mean_coverage': float(np.mean(list(c_row.values()))),
        })

    out = {
        'world':    pd.DataFrame(world_rows).T.reindex(columns=keep_names),
        'ego':      pd.DataFrame(ego_rows).T.reindex(columns=keep_names),
        'coverage': pd.DataFrame(cov_rows).T.reindex(columns=keep_names),
        'summary':  pd.DataFrame(summary).set_index('syllable'),
    }
    for key in ('world', 'ego', 'coverage'):
        out[key].index.name = 'syllable'
    out['unit'] = unit
    return out


def plot_keypoint_dispersion(res, coverage_thresh=0.6,
                             hide_in_ego=('neck', 'tailbase'),
                             hide_in_world=(),
                             figsize=(18, 10),font_size_multiplier = 1):
    """
    World and egocentric heatmaps (keypoints on y, syllables on x) plus a
    world-vs-egocentric scatter.

    Cells whose coverage falls below coverage_thresh are hatched rather than
    read as confident measurements, so a low value that merely reflects a
    rarely-tracked keypoint does not masquerade as stillness.

    hide_in_ego defaults to neck and tailbase because they are structurally
    ~0 in the body frame: tailbase is the origin and neck defines the axis.
    They are NOT hidden from the world panel, where both carry real signal and
    are in fact good proxies for whole-body translation. They are also excluded
    from the egocentric mean in the scatter, since averaging in structural
    zeros would drag it down artificially.

    The scatter gives the joint readout:
        high world / high ego -> locomotion with limb cycling
        high world / low  ego -> translation without articulation
        low  world / high ego -> stationary manipulation (grooming, feeding)
        low  world / low  ego -> rest
    """
    import matplotlib.gridspec as gridspec

    W, E, C = res['world'], res['ego'], res['coverage']
    unit = res.get('unit', '')
    syls = list(W.index)

    w_rows = [b for b in W.columns if b not in set(hide_in_world)]
    e_rows = [b for b in E.columns if b not in set(hide_in_ego)]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2.6, 1],
                           hspace=0.35, wspace=0.25)

    for row, (M, rows_kept, title) in enumerate((
            (W, w_rows, 'World frame'),
            (E, e_rows, 'Egocentric frame'))):
        ax = fig.add_subplot(gs[row, 0])
        sub = M[rows_kept]
        vals = sub.values.T                       # (keypoint, syllable)
        cov  = C[rows_kept].values.T

        im = ax.imshow(vals, aspect='auto', cmap='magma')
        ax.set_xticks(range(len(syls)))
        ax.set_xticklabels(syls, fontsize=8*font_size_multiplier)
        ax.set_yticks(range(len(rows_kept)))
        ax.set_yticklabels(rows_kept, fontsize=8*font_size_multiplier)
        ax.set_xlabel('syllable', fontsize=9*font_size_multiplier)
        ax.set_title(f'{title} - dispersion ({unit})', fontsize=10*font_size_multiplier)
        plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02)

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if cov[i, j] < coverage_thresh or not np.isfinite(vals[i, j]):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               fill=False, hatch='///',
                                               edgecolor='white', lw=0,
                                               alpha=.55))

    # Quadrant scatter, one point per syllable
    ax = fig.add_subplot(gs[:, 1])
    wm = W[w_rows].mean(axis=1)
    em = E[e_rows].mean(axis=1)
    ax.scatter(wm, em, s=60, c='steelblue', edgecolors='k',
               linewidths=.6, zorder=3)
    for syl in syls:
        ax.annotate(str(syl), (wm[syl], em[syl]), fontsize=8*font_size_multiplier,
                    xytext=(4, 3), textcoords='offset points')
    ax.axvline(np.nanmedian(wm), color='grey', lw=.8, ls='--')
    ax.axhline(np.nanmedian(em), color='grey', lw=.8, ls='--')
    ax.axvline(np.percentile(wm,75), color='red', lw=.8, ls='--')
    ax.axhline(np.percentile(em,75), color='red', lw=.8, ls='--')
    ax.set_xlabel(f'mean world dispersion ({unit})', fontsize=9*font_size_multiplier)
    ax.set_ylabel(f'mean egocentric dispersion ({unit})', fontsize=9*font_size_multiplier)
    ax.set_title('Translation vs articulation\n(dashed = median split)',
                 fontsize=10*font_size_multiplier)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()
    return fig


def classify_syllable_motion(res,
                             trunk_keypoints=TRUNK_KEYPOINTS,
                             distal_keypoints=DISTAL_KEYPOINTS,
                             trunk_agg='mean',
                             distal_agg='max',
                             method='noise',
                             n_noise=3.0,
                             trans_thresh=None,
                             artic_thresh=None,
                             verbose=True):
    """
    Reduce the dispersion tables to two axes and sort syllables into categories.

    Parameters
    ----------
    res : output of keypoint_dispersion()
    trunk_keypoints  : names combined for the translation axis (from res['world'])
    distal_keypoints : names combined for the articulation axis (from res['ego'])
    trunk_agg  : 'mean' | 'max' | 'q75'. Mean is right here -- trunk keypoints
                 should agree, so averaging reduces noise.
    distal_agg : 'mean' | 'max' | 'q75'. Default 'max', i.e. "does ANY distal
                 keypoint move a lot". Mean dilutes localized articulation: a
                 head-only scan or a one-handed reach gets divided by the number
                 of static keypoints in the set and can fall below threshold.
                 'q75' is a compromise if a single noisy keypoint is a concern.
    method : 'noise'  -> threshold = n_noise x the minimum observed value on
                         that axis, i.e. a multiple of the noise floor
             'median' -> median split on each axis
             'manual' -> use trans_thresh / artic_thresh
    n_noise : multiplier for method='noise'
    trans_thresh, artic_thresh : absolute thresholds in body lengths

    Returns
    -------
    DataFrame indexed by syllable with columns
        translation, articulation, category
    and .attrs['thresholds'] = (trans_thresh, artic_thresh)

    Categories
    ----------
        stationary    low translation,  low articulation  -> rest, holding still
        in_place      low translation,  high articulation -> grooming, feeding,
                                                             scanning, reaching
        transported   high translation, low articulation  -> rigid translation:
                                                             gliding, sliding,
                                                             being carried
        locomotion    high translation, high articulation -> walking, running,
                                                             climbing, jumping
    """
    W, E = res['world'], res['ego']

    tk = [k for k in trunk_keypoints  if k in W.columns]
    dk = [k for k in distal_keypoints if k in E.columns]
    if not tk:
        raise ValueError(f'none of {trunk_keypoints} in res["world"]')
    if not dk:
        raise ValueError(f'none of {distal_keypoints} in res["ego"]')
    if verbose:
        print(f'translation  = {trunk_agg} of world[{tk}]')
        print(f'articulation = {distal_agg} of ego[{dk}]')

    def _agg(frame, how):
        if how == 'mean':
            return frame.mean(axis=1, skipna=True)
        if how == 'max':
            return frame.max(axis=1, skipna=True)
        if how == 'q75':
            return frame.quantile(0.75, axis=1)
        raise ValueError("agg must be 'mean', 'max' or 'q75'")

    trans = _agg(W[tk], trunk_agg)
    artic = _agg(E[dk], distal_agg)

    df = pd.DataFrame({'translation': trans, 'articulation': artic}).dropna()

    # --- Thresholds ---
    if method == 'noise':
        t_thr = n_noise * df['translation'].min()
        a_thr = n_noise * df['articulation'].min()
    elif method == 'median':
        t_thr = df['translation'].median()
        a_thr = df['articulation'].median()
    elif method == 'manual':
        if trans_thresh is None or artic_thresh is None:
            raise ValueError("method='manual' needs trans_thresh and artic_thresh")
        t_thr, a_thr = trans_thresh, artic_thresh
    else:
        raise ValueError("method must be 'noise', 'median' or 'manual'")

    hi_t = df['translation'] > t_thr
    hi_a = df['articulation'] > a_thr
    df['category'] = np.select(
        [~hi_t & ~hi_a, ~hi_t & hi_a, hi_t & ~hi_a, hi_t & hi_a],
        ['stationary', 'in_place', 'transported', 'locomotion'],
        default='unknown')

    df.attrs['thresholds'] = (float(t_thr), float(a_thr))

    if verbose:
        print(f'\nthresholds ({method}): translation > {t_thr:.4f}, '
              f'articulation > {a_thr:.4f} body lengths')
        print(f'noise floor: translation {df["translation"].min():.4f}, '
              f'articulation {df["articulation"].min():.4f}')
        print('\n' + df['category'].value_counts().to_string())
        print()
        for cat in ('locomotion', 'transported', 'in_place', 'stationary'):
            sub = df[df['category'] == cat]
            if len(sub):
                print(f'{cat:>12}: {sorted(sub.index.tolist())}')

    return df.sort_values(['category', 'translation'])


def plot_motion_classes(cls, figsize=(9, 8), annotate=True):
    """
    Translation vs articulation scatter with quadrant boundaries, plus marginal
    histograms. Check the marginals for a natural gap -- if one axis is clearly
    bimodal, the gap is a better threshold than any multiple of the noise floor,
    and can be passed back via method='manual'.
    """
    t_thr, a_thr = cls.attrs['thresholds']
    colors = {'stationary': '#4477aa', 'in_place': '#228833',
              'transported': '#ccbb44', 'locomotion': '#ee6677',
              'unknown': 'grey'}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=.05, wspace=.05)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    for cat, grp in cls.groupby('category'):
        ax.scatter(grp['translation'], grp['articulation'], s=70,
                   color=colors.get(cat, 'grey'), edgecolors='k',
                   linewidths=.6, label=cat, zorder=3)
    if annotate:
        for syl, r in cls.iterrows():
            ax.annotate(str(syl), (r['translation'], r['articulation']),
                        fontsize=8, xytext=(5, 4), textcoords='offset points')

    ax.axvline(t_thr, color='grey', ls='--', lw=.9)
    ax.axhline(a_thr, color='grey', ls='--', lw=.9)
    ax.set_xlabel('translation  (world dispersion, trunk; body lengths)', fontsize=9)
    ax.set_ylabel('articulation  (egocentric dispersion, distal; body lengths)',
                  fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    nb = max(6, min(20, len(cls) // 2))
    axt.hist(cls['translation'], bins=nb, color='steelblue', alpha=.75)
    axt.axvline(t_thr, color='grey', ls='--', lw=.9)
    axt.set_ylabel('n', fontsize=8)
    axt.tick_params(labelbottom=False, labelsize=7)
    axt.set_title('Syllable motion classes\n'
                  'check marginals for a natural gap', fontsize=10)
    for s in ('top', 'right'):
        axt.spines[s].set_visible(False)

    axr.hist(cls['articulation'], bins=nb, orientation='horizontal',
             color='seagreen', alpha=.75)
    axr.axhline(a_thr, color='grey', ls='--', lw=.9)
    axr.set_xlabel('n', fontsize=8)
    axr.tick_params(labelleft=False, labelsize=7)
    for s in ('top', 'right'):
        axr.spines[s].set_visible(False)

    plt.show()
    return fig