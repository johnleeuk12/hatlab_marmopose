"""
marmo.clustering -- per-instance postural clustering and relabelling.

One row per syllable instance = the MEDIAN egocentric position of each
keypoint across that instance's frames. Medians rather than means for
robustness; per-instance rather than per-frame so long instances do not
dominate.

This asks a different question from the elevation analysis. Elevation and
flexion describe the trunk only; this uses the full postural configuration
including limbs, which is what a trunk-angle pair cannot see. On the real data
the resulting clusters differ significantly on both angles (elevation
p ~ 1e-19, flexion p ~ 1e-21), and clusters that share an elevation are
separated by flexion or by limb configuration alone.

Clustering is Ward hierarchical on the standardised feature matrix, not on the
embedding: UMAP distorts distances to make clusters look tighter than they are,
so merge heights computed on an embedding would be meaningless. Ward is also
deterministic, unlike k-means, and the tree can be inspected to see whether a
given cut is supported.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')

from marmo.config import (TAIL, SPINE, NECK, BODYPARTS,
                          DEFAULT_EXCLUDE_BODYPARTS, TRUNK_THR, MIN_FRAMES)
from marmo.geometry import ego_frame, body_length, segment_quality

LABEL_OFFSET = 100
UNASSIGNED   = 199


def _palette(n):
    try:
        import colorcet as cc
        return [cc.glasbey[i % len(cc.glasbey)] for i in range(n)]
    except ImportError:
        return [plt.cm.tab20(i % 20) for i in range(n)]


def instance_pose_features(pts, comp_df, syllables,
                           bodyparts=BODYPARTS,
                           exclude_bodyparts=DEFAULT_EXCLUDE_BODYPARTS,
                           trunk_thr=TRUNK_THR, use_quality_filter=True,
                           min_frames=MIN_FRAMES, min_coverage=0.5,
                           verbose=True):
    """
    Median egocentric keypoint position per instance, standardised.

    syllables : the set to include -- pass your stationary list.

    trunk_thr : frames whose tailbase->spinemid length deviates from its median
                by more than this fraction are dropped. That segment is
                bone-to-bone and should be constant, so its deviation is a
                per-frame localization-quality signal.

    Coordinates are divided by a single median body length for the whole
    recording rather than a per-frame length, so that noise in the
    neck-tailbase distance is not injected into every keypoint.

    Returns dict with X, cols, labels, meta.
    """
    q = segment_quality(pts, TAIL, SPINE)
    scale = body_length(pts, TAIL, NECK)
    ego = ego_frame(pts)          # centroid origin, tailbase->neck axis

    frame_ok = np.isfinite(ego[:, SPINE, 0])
    if use_quality_filter:
        frame_ok &= (q < trunk_thr)

    excl = set(exclude_bodyparts or ())
    keep = [(k, b) for k, b in enumerate(bodyparts) if b not in excl]

    T = len(pts)
    d = comp_df[(comp_df.start_frame + comp_df.duration_frames <= T) &
                (comp_df.syllable.isin(list(syllables)))]
    if verbose:
        print(f'{len(d)} instances across {d.syllable.nunique()} syllables; '
              f'body length {scale:.1f} mm; '
              f'usable frames {frame_ok.mean():.0%}')

    rows, labels, meta = [], [], []
    for r in d.itertuples(index=False):
        sl = slice(r.start_frame, r.start_frame + r.duration_frames)
        m = frame_ok[sl]
        if m.sum() < min_frames:
            continue
        E = ego[sl][m]
        feat = {}
        for k, b in keep:
            P = E[:, k, :]
            g = np.isfinite(P).all(axis=1)
            if g.sum() >= min_frames:
                med = np.median(P[g], axis=0) / scale
                for i, axn in enumerate(('u', 'v', 'w')):
                    feat[f'{b}_{axn}'] = med[i]
            else:
                for axn in ('u', 'v', 'w'):
                    feat[f'{b}_{axn}'] = np.nan
        rows.append(feat)
        labels.append(int(r.syllable))
        meta.append({'syllable': int(r.syllable),
                     'start_frame': int(r.start_frame),
                     'duration': int(r.duration_frames),
                     'n_good': int(m.sum())})

    F = pd.DataFrame(rows)
    labels = np.asarray(labels)
    meta = pd.DataFrame(meta)

    # Drop chronically missing features BEFORE complete-case filtering. The
    # other order discards most instances because of a few occluded extremities.
    cov = F.notna().mean(axis=0)
    bad = cov[cov < min_coverage].index.tolist()
    F = F.drop(columns=bad)
    if verbose and bad:
        print(f'dropped {len(bad)} low-coverage features (<{min_coverage:.0%}): {bad}')
    comp = F.notna().all(axis=1).values
    if verbose:
        print(f'complete cases: {comp.sum()} of {len(F)} ({comp.mean():.0%})')
    F = F[comp]
    labels = labels[comp]
    meta = meta[comp].reset_index(drop=True)

    X = F.values.astype(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    X = (X - mu) / sd
    if verbose:
        print(f'feature matrix: {X.shape[0]} instances x {X.shape[1]} features')
    return {'X': X, 'cols': list(F.columns), 'labels': labels, 'meta': meta}


def fit_umap(feats, n_components=3, n_neighbors=25, min_dist=0.1,
             random_state=0, verbose=True):
    """
    3D UMAP by default.

    n_neighbors should stay well below the per-syllable instance count,
    otherwise neighbourhoods span several syllables and the embedding blurs the
    structure being tested.
    """
    import umap
    X = feats['X']
    if verbose:
        n_min = pd.Series(feats['labels']).value_counts().min()
        if n_neighbors > n_min:
            print(f'  note: n_neighbors={n_neighbors} exceeds the smallest '
                  f'syllable count ({n_min})')
    return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                     min_dist=min_dist,
                     random_state=random_state).fit_transform(X)


def fit_pca(feats, n_components=3, verbose=True):
    """PCA alternative -- linear, deterministic, and the loadings are readable."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=n_components, random_state=0)
    emb = p.fit_transform(feats['X'])
    if verbose:
        print('explained variance:', np.round(p.explained_variance_ratio_, 3),
              f'(total {p.explained_variance_ratio_.sum():.1%})')
    return emb, p


def plot_umap_3d(emb, feats, labels=None, label_name=None,
                 figsize=(10, 9), point_size=8, elev=22, azim=45,
                 label_centroids=True):
    """
    Single rotatable 3D scatter, coloured categorically.

    labels : defaults to feats['labels'] (syllable); pass the cluster vector
             from cluster_instances to colour by cluster instead.
    """
    if labels is None:
        y = feats['labels']
        label_name = label_name or 'syllable'
    else:
        y = np.asarray(labels)
        label_name = label_name or 'group'
        if len(y) != len(emb):
            raise ValueError(f'labels has {len(y)} entries, emb has {len(emb)}')
    syls = sorted(np.unique(y))
    pal = _palette(len(syls))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for i, s in enumerate(syls):
        m = y == s
        ax.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=point_size, alpha=.6,
                   color=pal[i], edgecolors='none', label=str(s))
        if label_centroids:
            ax.text(np.median(emb[m, 0]), np.median(emb[m, 1]),
                    np.median(emb[m, 2]), str(s), fontsize=10, weight='bold')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2'); ax.set_zlabel('UMAP 3')
    ax.set_title(f'Per-instance pose, 3D UMAP -- coloured by {label_name}\n'
                 f'{len(y)} instances, {feats["X"].shape[1]} features',
                 fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=7, ncol=2, markerscale=2, frameon=False,
              loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_umap_projections(emb, feats, color_by=None, cmap='viridis',
                          labels=None, label_name=None,
                          figsize=(18, 5.5), point_size=8):
    """
    Three 2D projections of the 3D embedding -- dims 1-2, 1-3, 2-3 -- matching
    the XY/XZ/YZ layout in analysis_postproc.py.

    labels : CATEGORICAL colouring. Defaults to feats['labels'] (syllable);
             pass the cluster vector from cluster_instances to colour by cluster
             instead. Uses a discrete palette with a legend and per-group
             centroid labels.
    label_name : name for the legend/title, inferred as 'syllable' when labels
             is None and 'group' otherwise.
    color_by : CONTINUOUS colouring, a dict {name: values}, e.g.
             {'elevation': elev_array}. Takes precedence over labels. Do not use
             this for cluster ids -- a continuous colormap gives adjacent
             cluster numbers near-identical colours.
    """
    if labels is None:
        y = feats['labels']
        label_name = label_name or 'syllable'
    else:
        y = np.asarray(labels)
        label_name = label_name or 'group'
        if len(y) != len(emb):
            raise ValueError(f'labels has {len(y)} entries, emb has {len(emb)}')
    syls = sorted(np.unique(y))
    pal = _palette(len(syls))
    projs = [(0, 1, 'UMAP 1 vs 2'), (0, 2, 'UMAP 1 vs 3'), (1, 2, 'UMAP 2 vs 3')]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (i, j, ttl) in zip(axes, projs):
        if color_by is None:
            for c, s in enumerate(syls):
                m = y == s
                ax.scatter(emb[m, i], emb[m, j], s=point_size, alpha=.6,
                           color=pal[c], edgecolors='none',
                           label=str(s) if ax is axes[0] else None)
                ax.annotate(str(s), (np.median(emb[m, i]), np.median(emb[m, j])),
                            fontsize=8, weight='bold',
                            bbox=dict(boxstyle='round,pad=.12', fc='white',
                                      alpha=.6, ec='none'))
        else:
            name, vals = next(iter(color_by.items()))
            sc = ax.scatter(emb[:, i], emb[:, j], s=point_size, c=vals,
                            cmap=cmap, alpha=.75, edgecolors='none')
            if ax is axes[-1]:
                plt.colorbar(sc, ax=ax, fraction=.046, pad=.04, label=name)
        ax.set_xlabel(f'dim {i+1}', fontsize=9)
        ax.set_ylabel(f'dim {j+1}', fontsize=9)
        ax.set_title(ttl, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if color_by is None:
        axes[0].legend(fontsize=6, ncol=2, markerscale=2, frameon=False)
    ttl = f'coloured by {label_name}' if color_by is None else \
          f'coloured by {next(iter(color_by))}'
    fig.suptitle(f'Per-instance pose UMAP projections — {ttl}', fontsize=11)
    fig.tight_layout()
    plt.show()
    return fig


def syllable_dendrogram(feats, method='ward', metric='euclidean',
                        figsize=(13, 6), color_threshold_frac=0.7,
                        verbose=True):
    """
    Ward dendrogram of SYLLABLE CENTROIDS in feature space.

    This is the direct test of whether pooling is justified: syllables merging
    at a low height are postural neighbours, and one joining only near the root
    is distinct and should stay separate.

    Centroids are medians across each syllable's instances, and the tree is
    built in feature space rather than on the embedding, since UMAP distorts
    distances and would make the merge heights meaningless.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist, squareform

    X, y = feats['X'], feats['labels']
    syls = sorted(np.unique(y))
    C = np.stack([np.median(X[y == s], axis=0) for s in syls])
    n = np.array([np.sum(y == s) for s in syls])

    Z = linkage(pdist(C, metric=metric), method=method)
    labels = [f'{s} (n={c})' for s, c in zip(syls, n)]

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [1.5, 1]})
    dendrogram(Z, labels=labels, ax=axes[0], leaf_font_size=9,
               color_threshold=color_threshold_frac * Z[:, 2].max(),
               above_threshold_color='grey')
    axes[0].set_title(f'Syllable centroid dendrogram ({method})\n'
                      'low merge height = postural neighbours, poolable',
                      fontsize=10)
    axes[0].set_ylabel('distance', fontsize=9)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    D = squareform(pdist(C, metric=metric))
    from scipy.cluster.hierarchy import leaves_list
    o = leaves_list(Z)
    im = axes[1].imshow(D[np.ix_(o, o)], cmap='viridis_r', aspect='equal')
    axes[1].set_xticks(range(len(syls)))
    axes[1].set_xticklabels([str(syls[i]) for i in o], fontsize=8, rotation=90)
    axes[1].set_yticks(range(len(syls)))
    axes[1].set_yticklabels([str(syls[i]) for i in o], fontsize=8)
    axes[1].set_title('Centroid distance, reordered', fontsize=10)
    plt.colorbar(im, ax=axes[1], fraction=.046, pad=.04)

    fig.tight_layout()
    plt.show()

    if verbose:
        pairs = [(syls[i], syls[j], D[i, j])
                 for i in range(len(syls)) for j in range(i + 1, len(syls))]
        print('closest syllable pairs (most poolable):')
        for a, b, dd in sorted(pairs, key=lambda x: x[2])[:8]:
            print(f'  {a:>3} - {b:>3}   {dd:.2f}')
        print('\nmost distant from all others (least poolable):')
        far = pd.Series(D.sum(0) / (len(syls) - 1), index=syls)
        print(far.sort_values(ascending=False).head(4).round(2).to_string())
    return Z, C, syls


def cluster_instances(feats, k=4, method='ward', verbose=True):
    """
    Cluster INSTANCES, ignoring syllable labels, then cross-tabulate.

    If postural structure exists but is not syllable-shaped, this is what shows
    it: clusters that cut across many syllables mean the labels and the postures
    are close to independent.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    X, y = feats['X'], feats['labels']
    Z = linkage(X, method=method)
    cl = fcluster(Z, t=k, criterion='maxclust')

    ct = pd.crosstab(pd.Series(y, name='syllable'),
                     pd.Series(cl, name='cluster'))
    frac = ct.div(ct.sum(axis=1), axis=0)
    if verbose:
        print(f'\ninstance clustering into k={k} ({method}):')
        print('counts:'); print(ct.to_string())
        print('\nrow-normalised (fraction of each syllable per cluster):')
        print(frac.round(2).to_string())
        purity = frac.max(axis=1)
        print(f'\nsyllable purity (max fraction in one cluster): '
              f'median {purity.median():.2f}')
        print('  near 1.0 = syllable maps onto one postural cluster')
        print('  near 1/k = syllable spread evenly, labels and posture '
              'nearly independent')
    return cl, ct, Z


def relabel_from_clusters(comp_df, feats, cl,
                          stationary=None,
                          n_frames=None,
                          label_offset=LABEL_OFFSET,
                          unassigned=UNASSIGNED,
                          merge_adjacent=False,
                          verbose=True):
    """
    Build comp_df_post from a clustering of stationary instances.

    Parameters
    ----------
    comp_df    : original instance table (syllable, start_frame, duration_frames)
    feats      : output of instance_pose_features -- feats['meta'] supplies the
                 syllable and start_frame of every clustered instance
    cl         : cluster vector from cluster_instances, aligned row-for-row with
                 feats['meta']
    stationary : the syllable list passed to instance_pose_features. If None it
                 is inferred from feats['meta'], which is correct as long as
                 every stationary syllable contributed at least one surviving
                 instance -- pass it explicitly to be safe, otherwise a syllable
                 that was fully filtered out would be treated as non-stationary
                 and silently keep its original label.
    n_frames   : length of the keypoint array that was clustered. Instances
                 ending beyond this were never examined, so they keep their
                 original label and are flagged out_of_range instead of being
                 marked unassigned. Leave as None when the array covers the
                 whole recording.
    merge_adjacent : merge temporally adjacent instances that end up with the
                 same new label. Off by default because it changes durations.
                 Turn it on before computing transition matrices, where two
                 abutting instances sharing a label would otherwise register as
                 a self-transition that cannot really occur.

    Returns
    -------
    comp_df_post : DataFrame with syllable, start_frame, duration_frames,
                   syllable_orig, cluster, source
    """
    meta = feats['meta']
    cl = np.asarray(cl)
    if len(cl) != len(meta):
        raise ValueError(f'cl has {len(cl)} entries, feats["meta"] has {len(meta)}')
    if not comp_df.start_frame.is_unique:
        raise ValueError('comp_df.start_frame is not unique -- cannot join on it')

    if stationary is None:
        stationary = sorted(meta.syllable.unique())
        if verbose:
            print(f'stationary set inferred from feats["meta"]: {stationary}')
            print('  pass stationary= explicitly if any stationary syllable '
                  'lost all of its instances to filtering')
    stationary = set(int(s) for s in stationary)

    # start_frame -> new label, for clustered instances only
    mapping = dict(zip(meta.start_frame.astype(int),
                       (cl + label_offset).astype(int)))

    out = comp_df.copy()
    out['syllable_orig'] = out.syllable.astype(int)
    is_stat = out.syllable_orig.isin(stationary)
    new_lab = out.start_frame.astype(int).map(mapping)

    if n_frames is None:
        in_range = pd.Series(True, index=out.index)
    else:
        in_range = (out.start_frame + out.duration_frames) <= n_frames

    # clustered -> cluster label; stationary & examined but unclustered ->
    # sentinel; stationary but never examined -> keep original label
    out['syllable'] = np.where(
        new_lab.notna(), new_lab.fillna(-1),
        np.where(is_stat & in_range, unassigned, out.syllable_orig)).astype(int)

    out['cluster'] = np.where(new_lab.notna(),
                              new_lab.fillna(-1) - label_offset, np.nan)
    out['source'] = np.select(
        [new_lab.notna(), is_stat & in_range, is_stat & ~in_range],
        ['cluster', 'stationary_unassigned', 'out_of_range'],
        default='unchanged')

    if merge_adjacent:
        out = _merge_adjacent(out, verbose=verbose)

    if verbose:
        _report(out, meta, cl, label_offset, stationary)
    return out


def _merge_adjacent(df, verbose=True):
    """
    Merge instances that share a label and abut in time.

    Adjacency is tested as start_frame[i+1] == start_frame[i] + duration[i], so
    a NaN gap or a session boundary between two same-label instances correctly
    prevents the merge.
    """
    d = df.sort_values('start_frame').reset_index(drop=True)
    rows = []
    for r in d.itertuples(index=False):
        rec = r._asdict()
        if rows and rows[-1]['syllable'] == rec['syllable'] and \
                rows[-1]['start_frame'] + rows[-1]['duration_frames'] == rec['start_frame']:
            rows[-1]['duration_frames'] += rec['duration_frames']
        else:
            rows.append(rec)
    merged = pd.DataFrame(rows)
    if verbose:
        print(f'merge_adjacent: {len(d)} -> {len(merged)} instances '
              f'({len(d) - len(merged)} merged)')
    return merged


def _report(out, meta, cl, offset, stationary):
    print(f'\ncomp_df_post: {len(out)} instances')
    print(out.source.value_counts().to_string())

    # which original syllables feed each cluster -- the key diagnostic. A cluster
    # drawing from several syllables means the clustering cut across the labels.
    m = meta.copy()
    m['cluster'] = cl
    ct = pd.crosstab(m.syllable, m.cluster)
    print('\noriginal syllable x cluster (instances):')
    print(ct.to_string())

    print('\nnew label composition:')
    for c in sorted(m.cluster.unique()):
        sub = m[m.cluster == c]
        comp = sub.syllable.value_counts()
        parts = ', '.join(f'{s}({n})' for s, n in comp.items())
        print(f'  {offset + c}: n={len(sub):>4}  from syllables {parts}')

    n_un = (out.source == 'stationary_unassigned').sum()
    if n_un:
        lost = out.loc[out.source == 'stationary_unassigned', 'syllable_orig']
        n_stat_examined = n_un + (out.source == 'cluster').sum()
        print(f'\nunassigned stationary instances: {n_un} of '
              f'{n_stat_examined} examined ({n_un/n_stat_examined:.0%})')
        print('  by original syllable: ' +
              ', '.join(f'{s}({n})' for s, n in lost.value_counts().items()))
        print('  dropped by instance_pose_features (too few good frames, or an '
              'incomplete feature vector); they carry the sentinel label so '
              'they cannot contaminate the cluster grid movies')

    n_oor = (out.source == 'out_of_range').sum()
    if n_oor:
        print(f'\nout-of-range stationary instances: {n_oor}')
        print('  these end beyond n_frames, so they were never examined and '
              'keep their ORIGINAL syllable label. comp_df_post therefore '
              'mixes cluster labels with original ones -- filter on '
              'source == "cluster" for any cluster-only analysis')


def save_comp_df_post(out, path, verbose=True):
    """Write to csv. index=False avoids an unnamed extra column on read."""
    out.to_csv(path, index=False)
    if verbose:
        print(f'wrote {path}  ({len(out)} rows)')