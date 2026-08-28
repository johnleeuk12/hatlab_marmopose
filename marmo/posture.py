"""
marmo.posture -- trunk elevation and spinal flexion, per instance and per
syllable, plus per-syllable bimodality testing and the postural split.

Design notes, all driven by what the real data showed:

  QUALITY FILTER. Elevation is gated on the tailbase->spinemid segment;
  flexion additionally on spinemid->neck, since it is the angle AT spinemid.
  See geometry.quality_mask.

  FOLDED ELEVATION. Head-down is 4.8% of frames unfiltered but only 1.3%
  filtered, so folding with abs() and excluding are near-equivalent.

  TRUNK SEGMENT FOR ELEVATION. Measuring elevation on tailbase->spinemid makes
  it orthogonal to flexion by construction: flexion is the angle AT spinemid
  and cannot change the direction of the segment leading into it. Using
  tailbase->neck instead couples them -- a curled animal lying flat reads as
  ~55 deg elevation because the raised neck tilts the vector.

  PER-INSTANCE MEDIANS. Pooling frames weights long instances more heavily, and
  elevation genuinely changes within locomotion instances.

  TRANSITIONAL ON RANGE, NOT MAD. Real within-instance elevation MAD has a
  median of 0.62 deg, so a MAD threshold sits inside the noise. Range is
  interpretable: median 4.7 deg, p90 22 deg.

  PER-SYLLABLE BIMODALITY. Pooling syllables destroys it, because the high mode
  is shared across syllables (54-78 deg, median 68) while the low mode is
  syllable-specific (10-58 deg). Superimposing them stacks the high modes and
  smears the low ones into a continuum.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from marmo.config import (TAIL, SPINE, NECK, Z_SIGN, TRUNK_THR, FOLD_ABS,
                          TRANS_RANGE, MIN_FRAMES, MIN_INSTANCES,
                          EXCLUDE_SYLLABLE)
from marmo.geometry import segment_length, segment_quality


def frame_angles(pts, z_sign=Z_SIGN, trunk_thr=TRUNK_THR, fold=FOLD_ABS,
                 verbose=True):
    """
    Per-frame elevation and flexion with their own quality masks.

    elevation : angle of tailbase->spinemid above horizontal, via
                arctan2(rise, horizontal run). Signed then optionally folded.
                Measured on the TRUNK segment so it is orthogonal to flexion by
                construction: flexion is the angle AT spinemid and cannot change
                the direction of the segment leading into it.
    flexion   : angle at spinemid between tailbase and neck. 180 = extended.

    Returns dict with elev, flex, ok_elev, ok_flex, q_trunk, q_upper
    """
    L1 = segment_length(pts, TAIL, SPINE)
    L2 = segment_length(pts, SPINE, NECK)
    m1, m2 = np.nanmedian(L1), np.nanmedian(L2)
    q1 = segment_quality(pts, TAIL, SPINE)
    q2 = segment_quality(pts, SPINE, NECK)

    v = pts[:, SPINE, :] - pts[:, TAIL, :]
    elev = np.degrees(np.arctan2(v[:, 2] * z_sign,
                                 np.linalg.norm(v[:, [0, 1]], axis=1)))
    if fold:
        elev = np.abs(elev)

    u1 = pts[:, TAIL, :] - pts[:, SPINE, :]
    u2 = pts[:, NECK, :] - pts[:, SPINE, :]
    n1, n2 = np.linalg.norm(u1, axis=1), np.linalg.norm(u2, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = np.einsum('ij,ij->i', u1, u2) / (n1 * n2)
    flex = np.degrees(np.arccos(np.clip(cos, -1, 1)))

    ok_elev = np.isfinite(elev) & (q1 < trunk_thr)
    ok_flex = np.isfinite(flex) & (q1 < trunk_thr) & (q2 < trunk_thr)

    if verbose:
        tr = np.isfinite(L1)
        print(f'trunk median {m1:.1f} mm, upper median {m2:.1f} mm')
        print(f'elevation usable: {ok_elev.sum()} ({ok_elev.sum()/tr.sum():.0%} of trackable)')
        print(f'flexion   usable: {ok_flex.sum()} ({ok_flex.sum()/tr.sum():.0%} of trackable)')

    return {'elev': elev, 'flex': flex, 'ok_elev': ok_elev, 'ok_flex': ok_flex,
            'q_trunk': q1, 'q_upper': q2}


def instance_posture(pts, comp_df, ang=None, min_frames=MIN_FRAMES,
                     trans_range=TRANS_RANGE, exclude=EXCLUDE_SYLLABLE, verbose=True):
    """
    One row per syllable instance.

    Columns: syllable, start_frame, duration, n_elev, n_flex, frac_elev,
             elevation, elev_range, flexion, flex_range, transitional

    transitional marks instances whose elevation spans more than trans_range
    degrees. Their median describes neither endpoint of a postural change, so
    binning them would blur both groups.
    """
    if ang is None:
        ang = frame_angles(pts, verbose=verbose)
    elev, flex = ang['elev'], ang['flex']
    oe, of = ang['ok_elev'], ang['ok_flex']

    T = len(pts)
    d = comp_df[(comp_df.start_frame + comp_df.duration_frames <= T) &
                (comp_df.syllable != exclude)]
    if verbose and len(d) < len(comp_df):
        print(f'{len(d)} of {len(comp_df)} instances fall inside the '
              f'{T}-frame array')

    rows = []
    for r in d.itertuples(index=False):
        sl = slice(r.start_frame, r.start_frame + r.duration_frames)
        me, mf = oe[sl], of[sl]
        if me.sum() < min_frames:
            continue
        ev = elev[sl][me]
        row = {'syllable': int(r.syllable), 'start_frame': int(r.start_frame),
               'duration': int(r.duration_frames),
               'n_elev': int(me.sum()), 'frac_elev': float(me.mean()),
               'elevation': float(np.median(ev)),
               'elev_range': float(ev.max() - ev.min()),
               'n_flex': int(mf.sum())}
        if mf.sum() >= min_frames:
            fv = flex[sl][mf]
            row['flexion'] = float(np.median(fv))
            row['flex_range'] = float(fv.max() - fv.min())
        else:
            row['flexion'] = np.nan
            row['flex_range'] = np.nan
        row['transitional'] = row['elev_range'] > trans_range
        rows.append(row)

    I = pd.DataFrame(rows)
    if verbose and len(I):
        print(f'{len(I)} instances, {I.syllable.nunique()} syllables; '
              f'transitional (range > {trans_range:g} deg): '
              f'{I.transitional.sum()} ({I.transitional.mean():.1%})')
        print(f'flexion missing for {I.flexion.isna().sum()} '
              f'({I.flexion.isna().mean():.1%}) -- stricter two-segment filter')
    return I


def posture_by_syllable(I, min_inst=MIN_INSTANCES, drop_transitional=True):
    """
    Aggregate the per-instance table to one row per syllable.

    IQR is the column to read: a narrow IQR means one consistently held posture,
    a wide one means the syllable spans several. Elevation and flexion disagree
    about which syllables are tight, so both are needed -- on the test excerpt
    syllables 13 and 11 look impure on elevation (IQR 26-27) but tight on
    flexion (IQR 11-17), and syllables 0 and 10 are the reverse.
    """
    J = I[~I.transitional] if drop_transitional else I
    rows = []
    for s, g in J.groupby('syllable'):
        if len(g) < min_inst:
            continue
        f = g.flexion.dropna()
        rows.append({
            'syllable':  s,
            'n':         len(g),
            'n_flex':    len(f),
            'dur_med':   g.duration.median(),
            'elev_med':  g.elevation.median(),
            'elev_iqr':  g.elevation.quantile(.75) - g.elevation.quantile(.25),
            'flex_med':  f.median() if len(f) else np.nan,
            'flex_iqr':  (f.quantile(.75) - f.quantile(.25)) if len(f) > 1 else np.nan,
            'retention': g.frac_elev.median(),
            'n_trans':   int(I[(I.syllable == s)].transitional.sum()),
        })
    return pd.DataFrame(rows).set_index('syllable').sort_values('elev_med')


def find_trough(x, lo=0, hi=90, bw=0.18, bins=18, verbose=True):
    """
    Locate a split point, and report honestly how well supported it is.

    Relative dip depth on a smoothed histogram is the operational test: a value
    near 0 means the "trough" is a ripple, not a boundary, and any threshold
    placed there is a reporting convention rather than a natural division.
    KDE bandwidth strongly drives how many modes appear, so several are tried.
    """
    x = np.asarray(x); x = x[np.isfinite(x)]
    cts, edges = np.histogram(x, bins=np.linspace(lo, hi, bins + 1))
    sm = np.convolve(cts.astype(float), np.ones(3) / 3, mode='same')

    best = None
    for i in range(1, len(sm) - 1):
        if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1]:
            depth = 1 - sm[i] / max(min(sm[i - 1], sm[i + 1]), 1e-9)
            c = (edges[i] + edges[i + 1]) / 2
            if best is None or depth > best[1]:
                best = (float(c), float(depth))

    if verbose:
        print(f'\nn = {len(x)}')
        if best:
            print(f'deepest smoothed trough at {best[0]:.0f} deg, '
                  f'relative depth {best[1]:.2f}')
            if best[1] < 0.2:
                print('  WARNING: shallow. Treat the threshold as a convention, '
                      'not a natural boundary.')
        else:
            print('no interior trough -- distribution is unimodal')
        try:
            from scipy.stats import gaussian_kde
            for b in (0.10, 0.18, 0.30):
                g = np.linspace(lo, hi, 181)
                k = gaussian_kde(x, bw_method=b)(g)
                mo = [round(g[i]) for i in range(1, 180)
                      if k[i] >= k[i-1] and k[i] >= k[i+1]]
                print(f'  KDE bw {b:.2f}: modes at {mo}')
            print('  modes surviving all bandwidths are the believable ones')
        except ImportError:
            pass
    return best


def split_posture(I, elev_thresh=None, trough=None, verbose=True):
    """Assign each non-transitional instance to 'upright' or 'low'."""
    if elev_thresh is None:
        elev_thresh = trough[0] if trough else 45.0
        src = 'trough' if trough else 'default 45 deg'
    else:
        src = 'user supplied'
    out = I.copy()
    out['posture'] = np.where(
        out.transitional, 'transitional',
        np.where(out.elevation >= elev_thresh, 'upright', 'low'))
    out.attrs['elev_thresh'] = float(elev_thresh)
    if verbose:
        print(f'\nthreshold {elev_thresh:.1f} deg ({src})')
        print(out.posture.value_counts().to_string())
    return out


def validate_split(sp, verbose=True):
    """
    Does the split cut ACROSS syllables?

    frac_upright near 0 or 1 means the syllable already encoded posture;
    near 0.5 means it contained both and the per-instance split recovers
    structure the labels did not carry. mixedness rescales that to 0..1.
    """
    d = sp[sp.posture != 'transitional']
    t = (d.assign(u=(d.posture == 'upright'))
           .groupby('syllable')
           .agg(n=('u', 'size'), frac_upright=('u', 'mean'),
                elev_med=('elevation', 'median'),
                flex_med=('flexion', 'median'))
           .sort_values('frac_upright'))
    t['mixedness'] = 1 - 2 * (t.frac_upright - 0.5).abs()
    if verbose:
        print('\nper-syllable composition:')
        print(t.round(3).to_string())
        mx = t[t.mixedness > 0.5]
        print(f'\nsubstantially mixed (>0.5): {sorted(mx.index.tolist())}')
        print('\ngroup comparison:')
        print(d.groupby('posture')
               .agg(n=('elevation', 'size'), elev=('elevation', 'median'),
                    flex=('flexion', 'median'), dur=('duration', 'median'))
               .round(2).to_string())
    return t


def plot_by_syllable(I, metric='elevation', ncol=5, min_inst=MIN_INSTANCES,
                     drop_transitional=True, bins=None, syllables=None):
    """
    Grid of per-instance histograms, one panel per syllable, sorted by median.

    syllables : optional list to restrict to. When given, min_inst is ignored so
                that explicitly requested syllables are shown even if sparse.
    """
    J = I[~I.transitional] if drop_transitional else I
    J = J[J[metric].notna()]
    if syllables is not None:
        J = J[J.syllable.isin(list(syllables))]
        grp = {s: g[metric].values for s, g in J.groupby('syllable')}
    else:
        grp = {s: g[metric].values for s, g in J.groupby('syllable')
               if len(g) >= min_inst}
    if not grp:
        raise ValueError('no syllables to plot')
    if bins is None:
        bins = np.arange(0, 91, 5) if metric == 'elevation' else np.arange(60, 181, 5)
    order = sorted(grp, key=lambda s: np.median(grp[s]))
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow),
                             sharex=True, squeeze=False)
    for ax, s in zip(axes.ravel(), order):
        v = grp[s]
        ax.hist(v, bins=bins, color='steelblue', edgecolor='white', linewidth=.3)
        ax.axvline(np.median(v), color='firebrick', ls='--', lw=1)
        iqr = np.percentile(v, 75) - np.percentile(v, 25)
        ax.set_title(f'syl {s}  n={len(v)}\nmed {np.median(v):.0f}  IQR {iqr:.0f}',
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(order):]:
        ax.set_axis_off()
    fig.supxlabel(f'{metric} (deg)')
    fig.tight_layout()
    plt.show()
    return fig


def plot_split(sp, trough=None, syllables=None, figsize=(17, 5)):
    """
    Pooled histogram, elevation-flexion scatter, per-syllable composition.

    syllables : optional list to restrict the plot to, e.g. the mixed ones.
                The threshold is NOT recomputed -- it stays wherever
                split_posture put it, so panels remain comparable across
                different subsets. To place the threshold using only a subset,
                call find_trough on that subset's elevations and pass the
                result to split_posture.
    """
    thr = sp.attrs.get('elev_thresh')
    if syllables is not None:
        keep = sp[sp.syllable.isin(list(syllables))]
        keep.attrs['elev_thresh'] = thr
        sp = keep
        if not len(sp):
            raise ValueError(f'no instances for syllables {list(syllables)}')
    d = sp[sp.posture != 'transitional']
    col = {'upright': '#ee6677', 'low': '#4477aa', 'transitional': '#bbbbbb'}
    fig, ax = plt.subplots(1, 3, figsize=figsize,
                           gridspec_kw={'width_ratios': [1.2, 1.2, 1]})

    ax[0].hist(sp.elevation, bins=np.arange(0, 91, 5), color='lightsteelblue',
               edgecolor='white', linewidth=.4)
    if thr:
        ax[0].axvline(thr, color='firebrick', ls='--', lw=1.2,
                      label=f'{thr:.0f}°')
        ax[0].legend(fontsize=8, frameon=False)
    dep = f" (dip depth {trough[1]:.2f})" if trough else ''
    sub = f"  syl {sorted(sp.syllable.unique())}" if syllables is not None else ''
    ax[0].set_title(f'Per-instance elevation{dep}\nn={len(sp)}{sub}', fontsize=9)
    ax[0].set_xlabel('elevation (deg)', fontsize=9)

    for p, g in sp.groupby('posture'):
        ax[1].scatter(g.flexion, g.elevation, s=14, alpha=.6,
                      color=col.get(p, 'grey'), label=p, edgecolors='none')
    if thr:
        ax[1].axhline(thr, color='firebrick', ls='--', lw=1)
    ax[1].set_xlabel('flexion (deg) — 180 = extended', fontsize=9)
    ax[1].set_ylabel('elevation (deg)', fontsize=9)
    ax[1].set_title('Per-instance posture space', fontsize=10)
    ax[1].legend(fontsize=8, frameon=False, markerscale=1.5)

    fr = (d.assign(u=(d.posture == 'upright'))
            .groupby('syllable').u.mean().sort_values())
    y = np.arange(len(fr))
    ax[2].barh(y, fr.values,
               color=['#4477aa' if v < .5 else '#ee6677' for v in fr.values])
    ax[2].axvline(.5, color='k', ls=':', lw=.9)
    ax[2].set_yticks(y); ax[2].set_yticklabels(fr.index, fontsize=8)
    ax[2].set_xlim(0, 1)
    ax[2].set_xlabel('fraction upright', fontsize=9)
    ax[2].set_title('Composition per syllable\nnear 0.5 = mixed', fontsize=10)

    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.show()
    return fig


def _kde_modes(x, grid, bw, min_mass=0.20):
    """
    KDE maxima and minima. A maximum only counts if its density reaches
    min_mass of the global peak, which excludes ripples in the tails.
    """
    from scipy.stats import gaussian_kde
    k = gaussian_kde(x, bw_method=bw)(grid)
    thresh = min_mass * k.max()
    maxs = [i for i in range(1, len(k) - 1)
            if k[i] >= k[i-1] and k[i] >= k[i+1] and k[i] >= thresh]
    mins = [i for i in range(1, len(k) - 1)
            if k[i] <= k[i-1] and k[i] <= k[i+1]]
    return k, maxs, mins


def _best_trough(x, lo=0, hi=90, bw=0.30, min_mass=0.20, min_sep=10.0):
    """
    Deepest trough that lies strictly BETWEEN two qualifying modes.

    This flanking requirement is the fix for a real failure mode: an empty tail
    at the edge of the distribution is trivially a local minimum with depth
    1.00, which previously outranked every genuine dip and produced meaningless
    "perfect" troughs for sparse syllables.

    min_sep rejects mode pairs closer together than this many degrees, which
    would be a shoulder rather than two states.

    Returns (trough_deg, depth, mode_lo, mode_hi) or None.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return None
    grid = np.linspace(lo, hi, 181)
    try:
        k, maxs, mins = _kde_modes(x, grid, bw, min_mass)
    except Exception:
        return None
    if len(maxs) < 2:
        return None

    best = None
    for a, b in zip(maxs[:-1], maxs[1:]):
        if grid[b] - grid[a] < min_sep:
            continue
        inner = [i for i in mins if a < i < b]
        if not inner:
            continue
        t = min(inner, key=lambda i: k[i])
        depth = 1 - k[t] / min(k[a], k[b])
        if best is None or depth > best[1]:
            best = (float(grid[t]), float(depth), float(grid[a]), float(grid[b]))
    return best


def find_syllable_troughs(I, syllables=None, min_inst=20, n_boot=200,
                          bw=0.30, drop_transitional=True, seed=0,
                          verbose=True):
    """
    Test each syllable's elevation distribution for two modes.

    Rather than excluding small syllables by a cutoff, every syllable is
    reported with a bootstrap support value: the fraction of resamples in which
    a qualifying trough is found at all. Low support IS the answer for a sparse
    syllable, and boot_iqr shows how unstable the location is.

    Returns DataFrame indexed by syllable:
        n, trough, depth, mode_lo, mode_hi, support, boot_med, boot_iqr
    """
    rng = np.random.default_rng(seed)
    J = I[~I.transitional] if drop_transitional else I
    if syllables is not None:
        J = J[J.syllable.isin(list(syllables))]

    rows = []
    for s, g in J.groupby('syllable'):
        e = g.elevation.dropna().values
        if len(e) < min_inst:
            if verbose:
                print(f'  syllable {s}: n={len(e)} < {min_inst}, skipped')
            continue
        res = _best_trough(e, bw=bw)

        hits = []
        for _ in range(n_boot):
            r = _best_trough(rng.choice(e, len(e), replace=True), bw=bw)
            if r is not None:
                hits.append(r[0])
        support = len(hits) / n_boot
        rows.append({
            'syllable': s, 'n': len(e),
            'trough':  res[0] if res else np.nan,
            'depth':   res[1] if res else np.nan,
            'mode_lo': res[2] if res else np.nan,
            'mode_hi': res[3] if res else np.nan,
            'support': support,
            'boot_med': float(np.median(hits)) if hits else np.nan,
            'boot_iqr': float(np.percentile(hits, 75) - np.percentile(hits, 25))
                        if len(hits) > 3 else np.nan,
        })

    t = pd.DataFrame(rows).set_index('syllable').sort_values(
        'depth', ascending=False, na_position='last')
    if verbose and len(t):
        print(f'\nper-syllable elevation troughs (KDE bw {bw}, '
              f'{n_boot} bootstrap resamples):')
        print(t.round(2).to_string())
        strong = t[(t.depth > 0.20) & (t.support > 0.80)]
        print(f'\nwell-supported bimodal (depth > 0.20, support > 0.80): '
              f'{sorted(strong.index.tolist())}')
        print('  boot_iqr is the trough location spread across resamples; '
              'a wide value means the split point is not pinned down '
              'even where a trough exists')
    return t


def plot_syllable_troughs(I, troughs, ncol=4, bw=0.30,
                          drop_transitional=True, figsize_scale=(3.4, 2.5)):
    """Per-syllable elevation KDE with detected modes and trough marked."""
    from scipy.stats import gaussian_kde
    J = I[~I.transitional] if drop_transitional else I
    syls = list(troughs.index)
    nrow = int(np.ceil(len(syls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, sharex=True, squeeze=False,
                             figsize=(figsize_scale[0]*ncol,
                                      figsize_scale[1]*nrow))
    grid = np.linspace(0, 90, 181)
    for ax, s in zip(axes.ravel(), syls):
        e = J.loc[J.syllable == s, 'elevation'].dropna().values
        ax.hist(e, bins=np.arange(0, 91, 5), density=True,
                color='lightsteelblue', edgecolor='white', linewidth=.3)
        try:
            ax.plot(grid, gaussian_kde(e, bw_method=bw)(grid),
                    color='darkslateblue', lw=1.3)
        except Exception:
            pass
        r = troughs.loc[s]
        if np.isfinite(r.trough):
            ax.axvline(r.trough, color='firebrick', ls='--', lw=1.2)
            for m in (r.mode_lo, r.mode_hi):
                ax.axvline(m, color='seagreen', ls=':', lw=1)
            ttl = (f'syl {s}  n={int(r.n)}\ntrough {r.trough:.0f}°  '
                   f'depth {r.depth:.2f}  sup {r.support:.0%}')
        else:
            ttl = f'syl {s}  n={int(r.n)}\nno qualifying trough'
        ax.set_title(ttl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    for ax in axes.ravel()[len(syls):]:
        ax.set_axis_off()
    fig.supxlabel('trunk elevation (deg)')
    fig.tight_layout()
    plt.show()
    return fig