#!/usr/bin/env python3
"""
scripts/diagnostics.py -- one-off data-quality reports.

These are reports you read, not building blocks you compose, which is why they
live in scripts/ rather than in the package. Promote a function into marmo/ if
you find yourself wanting to call it from another analysis.

Run:  python scripts/diagnostics.py
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

import marmo
from marmo import io
from marmo.config import TAIL, SPINE, NECK, Z_SIGN
from marmo.geometry import segment_length, segment_quality

ARRAY  = 'combined_eg.npy'
COMPDF = 'comp_df_v2.csv'


def trunk_quality_report(a):
    """
    Is trunk-length instability worse near the tracking boundary?

    Tested three ways: frames to the nearest full dropout, trunk elevation
    (the animals go untracked when they descend), and height. On the test data
    the answer is emphatically yes -- error is 5.5x worse adjacent to a dropout
    and 145% for head-down frames, which makes trunk length a usable per-frame
    quality filter rather than merely a diagnostic.
    """
    L   = segment_length(a, TAIL, SPINE)
    err = segment_quality(a, TAIL, SPINE)
    dropout = np.isnan(a[:, :, 0]).all(axis=1)
    dist = distance_transform_edt(~dropout)

    ax   = a[:, SPINE, :] - a[:, TAIL, :]
    run  = np.linalg.norm(ax[:, [0, 1]], axis=1)
    elev = np.degrees(np.arctan2(ax[:, 2] * Z_SIGN, run))
    height = a[:, TAIL, 2] * Z_SIGN

    ok = np.isfinite(err)
    print(f'trunk median {np.nanmedian(L):.1f} mm | {ok.sum()} measurable frames')
    print(f'dropout frames: {dropout.sum()} ({dropout.mean():.1%})\n')

    def binned(x, name, edges):
        print(f'=== trunk length error vs {name} ===')
        print(f'{"bin":>18} {"n":>7} {"med err":>9} {"frac >25%":>10}')
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (x >= lo) & (x < hi)
            if m.sum() < 20:
                continue
            print(f'{f"{lo:g} to {hi:g}":>18} {m.sum():>7} '
                  f'{np.median(err[m]):>9.3f} {np.mean(err[m] > .25):>10.1%}')
        print()

    binned(dist, 'frames from dropout', [1, 2, 3, 5, 10, 25, 50, 100, 1e9])
    binned(elev, 'trunk elevation (deg)', [-90, -45, -20, 0, 20, 40, 60, 90])
    binned(height, 'tailbase height',
           list(np.nanpercentile(height[ok], [0, 10, 25, 50, 75, 90, 100])))

    print('=== Spearman correlation with trunk length error ===')
    for x, name in ((dist, 'distance from dropout'), (elev, 'elevation'),
                    (height, 'height')):
        m = ok & np.isfinite(x)
        r, p = spearmanr(x[m], err[m])
        print(f'  {name:<24} rho {r:+.3f}   p {p:.1e}')


def retention_by_syllable(a, comp_df, trunk_thr=0.25):
    """
    Per-syllable retention under the trunk filter.

    Always report this alongside a filtered result: retention ranges 34-94%
    across syllables, so a syllable at 34% is a different kind of claim than one
    at 94%. The bias is not predictable from duration -- check, do not assume.
    """
    q  = segment_quality(a, TAIL, SPINE)
    tr = np.isfinite(q)
    d  = io.in_range(comp_df, len(a))
    print(f'\n{"syl":>5}{"n_inst":>8}{"med_dur":>9}{"frames":>9}{"retention":>11}')
    for s, g in d.groupby('syllable'):
        idx = np.concatenate([np.arange(r.start_frame, r.start_frame + r.duration_frames)
                              for r in g.itertuples(index=False)])
        t = tr[idx]
        if t.sum() < 30:
            continue
        print(f'{s:>5}{len(g):>8}{g.duration_frames.median():>9.0f}'
              f'{int(t.sum()):>9}{(q[idx][t] < trunk_thr).mean():>11.0%}')


def nan_structure(a):
    """Per-keypoint NaN rates, and how NaN is distributed within frames."""
    nan_kp = np.isnan(a[:, :, 0])
    n_per  = nan_kp.sum(axis=1)
    print(f'\nframes: {len(a)}  any-NaN {(n_per>0).mean():.1%}  '
          f'all-NaN {(n_per==a.shape[1]).mean():.1%}')
    print('NaN keypoints per mixed frame:')
    for c in range(1, a.shape[1]):
        n = (n_per == c).sum()
        if n:
            print(f'  {c:>2}: {n:>6}')
    print('\nper-keypoint NaN fraction:')
    for k, nm in enumerate(marmo.BODYPARTS):
        f = nan_kp[:, k].mean()
        print(f'  {nm:<12} {f:6.1%} {"#" * int(45 * f)}')


if __name__ == '__main__':
    a = io.load_combined(ARRAY)
    comp_df = io.load_comp_df(COMPDF)
    nan_structure(a)
    trunk_quality_report(a)
    retention_by_syllable(a, comp_df)