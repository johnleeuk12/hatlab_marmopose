# %%
# =============================================================================
# Figure 1, rebuilt on SYLLABLE INSTANCE MEDIANS instead of individual frames.

# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list


# -----------------------------------------------------------------------------
# Velocity per instance (the 4th dimension of the original figure)
# -----------------------------------------------------------------------------

def instance_velocity(pts, meta, trunk_keypoints=(9, 8, 3), fps=25,
                      window_frames=None):
    """
    Median centroid speed over each instance, in mm/s.

    instance_pose_features returns pose only, so velocity is computed here and
    concatenated as a 4th dimension. Median rather than mean, and computed from
    a central difference, so a single bad frame does not dominate a short bout.

    meta must be feats['meta'] -- the row order has to match the feature matrix.
    """
    c = np.nanmean(pts[:, list(trunk_keypoints), :], axis=1)
    v = np.full(len(c), np.nan)
    v[1:-1] = np.linalg.norm(c[2:] - c[:-2], axis=1) / 2 * fps

    out = np.full(len(meta), np.nan)
    for i, r in enumerate(meta.itertuples(index=False)):
        dur = r.duration if window_frames is None else min(r.duration, window_frames)
        seg = v[r.start_frame:r.start_frame + dur]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


# -----------------------------------------------------------------------------
# 4D space: UMAP(3) + velocity, then per-syllable centroids
# -----------------------------------------------------------------------------

def build_4d_space(X, velocity, n_neighbors=15, min_dist=0.1,
                   velocity_weight=2.0, random_state=0, verbose=True):
    """
    UMAP to 3 components on the instance medians, with normalised velocity as a
    4th dimension.

    n_neighbors defaults far lower than the per-frame version's 25, because n is
    now instances rather than frames. It must stay well below the smallest
    per-syllable instance count or neighbourhoods span syllables.

    Velocity is scaled to [0,1] then multiplied by velocity_weight so it is
    commensurate with the UMAP axes rather than being swamped by them.
    """
    import umap
    if verbose:
        print(f'UMAP on {X.shape[0]} instances x {X.shape[1]} features, '
              f'n_neighbors={n_neighbors}')
    emb = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                    random_state=random_state).fit_transform(X)

    v = np.asarray(velocity, float).reshape(-1, 1)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    vnorm = (v - vmin) / (vmax - vmin + 1e-8) * velocity_weight
    return np.concatenate([emb, vnorm], axis=1), emb


def syllable_centroids(emb4d, labels, min_inst=3, robust=True, verbose=True):
    """
    Per-syllable centroid and spread across INSTANCES.

    robust=True uses median and MAD rather than mean and std. With instances as
    the unit there are far fewer points per syllable than there were frames, so
    a single outlying bout has much more leverage on a mean.

    Spread here is between-bout variability: low means the syllable lands in the
    same place every time it occurs.
    """
    syls = sorted(np.unique(labels))
    cents, spreads, keep = {}, {}, []
    for s in syls:
        m = labels == s
        if m.sum() < min_inst:
            continue
        pts = emb4d[m]
        if robust:
            c = np.median(pts, axis=0)
            sp = np.median(np.abs(pts - c), axis=0)
        else:
            c, sp = np.nanmean(pts, axis=0), np.nanstd(pts, axis=0)
        cents[s], spreads[s] = c, sp
        keep.append(s)
    if verbose:
        dropped = [s for s in syls if s not in keep]
        print(f'{len(keep)} syllables with >= {min_inst} instances'
              + (f'; dropped {dropped}' if dropped else ''))
    C = np.stack([cents[s] for s in keep])
    S = np.stack([spreads[s] for s in keep])
    return keep, C, S


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------


def plot_figure1(syl_include, C, S, labels, trans_matrix,
                 spread_label='MAD', figsize=(23, 13)):
    """
    trans_matrix : DataFrame from get_transition_matrix, indexed by label. Only
                   the rows/columns in syl_include are used, so a matrix built
                   on the full table can be passed directly.
 
    Dendrogram, instance-frequency bar, centroid-distance heatmap, transition
    heatmap, spread heatmap.
 
    The two heatmaps are ordered by their OWN clustering, not a shared one:
    the distance heatmap follows the centroid tree, the transition heatmap
    follows a tree built on transition profiles. Comparing the two orderings is
    the point -- postural similarity and transition similarity are different
    things, and forcing one order onto both hides that.
    """
    n = len(syl_include)
    lab = [str(s) for s in syl_include]
 
    D = cdist(C, C, metric='euclidean')
    Z = linkage(squareform(D, checks=False), method='ward')
    o1 = leaves_list(Z)
 
    # trans_matrix is a DataFrame indexed by label; restrict to the syllables
    # that survived min_inst, then work with the values
    T_sub = trans_matrix.loc[syl_include, syl_include].values
    Tsym = (T_sub + T_sub.T) / 2
    Z2 = linkage(squareform(cdist(Tsym, Tsym), checks=False), method='ward')
    o2 = leaves_list(Z2)
 
    counts = pd.Series(labels).value_counts()
    freq = np.array([counts.get(s, 0) for s in syl_include], float)
    freq /= freq.sum()
 
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Syllable structure from INSTANCE MEDIANS '
                 '(4D: UMAP + velocity)', fontsize=13)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2.2, 1.6, 1.3],
                           height_ratios=[1, 2], hspace=.35, wspace=.35)
 
    ax = fig.add_subplot(gs[0, 0])
    dendrogram(Z, labels=lab, ax=ax, leaf_font_size=9,
               color_threshold=.7 * Z[:, 2].max(), above_threshold_color='grey')
    ax.set_title('Centroid clustering (Ward)', fontsize=10)
    ax.set_ylabel('distance', fontsize=9)
 
    # Sorted by descending frequency rather than label order. The label order
    # carries no meaning, so sorting makes the distribution's shape readable --
    # whether a few syllables dominate or the mass is spread evenly.
    ofreq = np.argsort(freq)[::-1]
    ax = fig.add_subplot(gs[0, 1:])
    ax.bar(range(n), freq[ofreq],
           color=[plt.cm.tab20(i % 20) for i in ofreq])
    ax.set_xticks(range(n))
    ax.set_xticklabels([lab[i] for i in ofreq], fontsize=15, rotation=90)
    ax.set_ylabel('instance frequency', fontsize=16)
    ax.set_title('Instance frequency, descending (each bout counts once)',
                 fontsize=10)
 
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(D[np.ix_(o1, o1)], cmap='viridis_r', aspect='equal')
    ax.set_xticks(range(n)); ax.set_xticklabels([lab[i] for i in o1], fontsize=7, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels([lab[i] for i in o1], fontsize=7)
    ax.set_title('Centroid distance\n(ordered by centroid tree)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
 
    ax = fig.add_subplot(gs[1, 1])
    Tr = T_sub[np.ix_(o2, o2)]
    im = ax.imshow(Tr, cmap='hot_r', aspect='equal',
                   vmin=0, vmax=np.percentile(Tr[Tr > 0], 95) if (Tr > 0).any() else 1)
    ax.set_xticks(range(n)); ax.set_xticklabels([lab[i] for i in o2], fontsize=7, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels([lab[i] for i in o2], fontsize=7)
    ax.set_title('Transition matrix (bigram)\n(ordered by transition tree)', fontsize=10)
    ax.set_xlabel('to', fontsize=8); ax.set_ylabel('from', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
 
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(S, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['UMAP-1', 'UMAP-2', 'UMAP-3', 'velocity'],
                       fontsize=8, rotation=30)
    ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=8)
    ax.set_title(f'Between-instance spread ({spread_label})', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
 
    fig.tight_layout()
    plt.show()
 
    pairs = [(syl_include[i], syl_include[j], D[i, j])
             for i in range(n) for j in range(i + 1, n)]
    print('\nclosest syllable pairs by instance-median centroid:')
    for a, b, dd in sorted(pairs, key=lambda x: x[2])[:8]:
        print(f'  {a:>4} - {b:>4}   {dd:.2f}')
    return fig, Z, D

# -----------------------------------------------------------------------------
# Transition graph
# -----------------------------------------------------------------------------
def plot_transition_graph(syl_include, labels, trans_matrix,
                          top_n=40, edge_thresh=None, weight_label=None,
                          show_negative=False, layout='circular',
                          figsize=(11, 11), verbose=True):
    """
    Directed transition graph over the same syllables as plot_figure1.
 
    Takes whichever matrix you pass, so the choice of quantity is made upstream
    in get_transition_matrix:
 
        raw probability    normalize='bigram'      -- frequent syllables dominate
        source-normalised  normalize='rows'        -- P(j|i)
        frequency-free     normalize='enrichment'  -- observed / expected, 1 = chance
        signed version     normalize='pmi'         -- log2 enrichment, 0 = chance
 
    For comparing syllables with very different instance counts, use enrichment
    or pmi. On this data the strongest cells under 'bigram' are simply the five
    most frequent syllables, whereas pmi shows no residual correlation with
    instance frequency.
 
    NaN cells (masked by min_count) are skipped rather than treated as zero, so
    "too rare to judge" does not become "does not happen".
 
    show_negative : with pmi, negative weights mean a transition occurs LESS
                    often than chance -- an avoided transition. Off by default
                    because mixing avoided and preferred edges on one graph is
                    hard to read; when on they are drawn dashed and blue.
 
    top_n selects the N strongest edges and is scale-free, which matters because
    an absolute threshold does not transfer between quantities or label sets: a
    0.005 cut gives 48 edges on a bigram matrix of comp_df and zero on
    comp_df_post. With enrichment, edge_thresh=1.0 is meaningful (above chance)
    and is a reasonable alternative.
    """
    import networkx as nx
 
    M = trans_matrix.loc[syl_include, syl_include].to_numpy(copy=True)
    np.fill_diagonal(M, np.nan)          # self-transitions are not between-behaviour
 
    signed = weight_label == 'pmi' or (np.nanmin(M) < 0 if np.isfinite(M).any() else False)
    score = np.abs(M) if (signed and show_negative) else M
    valid = np.isfinite(score) & (score > 0 if not signed or not show_negative
                                  else np.isfinite(score))
    if signed and not show_negative:
        valid &= (M > 0)
 
    if edge_thresh is not None:
        cut = edge_thresh
    else:
        pool = score[valid]
        cut = np.sort(pool)[-min(top_n, len(pool))] if len(pool) else np.inf
 
    counts = pd.Series(labels).value_counts()
    freq = {s: counts.get(s, 0) for s in syl_include}
    ftot = sum(freq.values()) or 1
    freq = {s: v / ftot for s, v in freq.items()}
    fmax = max(freq.values()) or 1.0
 
    G = nx.DiGraph()
    G.add_nodes_from(syl_include)
    ii, jj = np.nonzero(valid & (score >= cut))
    for i, j in zip(ii, jj):
        G.add_edge(syl_include[i], syl_include[j],
                   weight=float(score[i, j]), signed=float(M[i, j]))
    if verbose and G.number_of_edges() == 0:
        print('no edges drawn -- lower edge_thresh or raise top_n')
 
    pos = (nx.circular_layout(G) if layout == 'circular'
           else nx.spring_layout(G, seed=42, k=2.0))
    wmax = max((d['weight'] for _, _, d in G.edges(data=True)), default=1.0)
    colors = {s: plt.cm.tab20(i % 20) for i, s in enumerate(syl_include)}
 
    lab = weight_label or trans_matrix.attrs.get('normalize', 'weight')
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    cutlab = f'top {top_n}' if edge_thresh is None else f'> {edge_thresh}'
    ax.set_title(f'Syllable transition graph -- edge weight = {lab}\n'
                 f'{cutlab} edges, self-transitions removed, '
                 'node size proportional to instance frequency', fontsize=11)
 
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=[200 + 3000 * freq[s] / fmax for s in G.nodes()],
        node_color=[colors[s] for s in G.nodes()],
        edgecolors='black', linewidths=.6, alpha=.9)
 
    pos_e = [(u, v) for u, v, d in G.edges(data=True) if d['signed'] > 0]
    neg_e = [(u, v) for u, v, d in G.edges(data=True) if d['signed'] <= 0]
    for elist, col, style in ((pos_e, '0.35', 'solid'),
                              (neg_e, 'steelblue', 'dashed')):
        if not elist:
            continue
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=elist,
            width=[0.4 + 4.0 * G[u][v]['weight'] / wmax for u, v in elist],
            edge_color=col, style=style, alpha=.55, arrows=True, arrowsize=11,
            connectionstyle='arc3,rad=0.12', node_size=600)
 
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    fig.tight_layout()
    plt.show()
 
    if verbose:
        n = len(syl_include)
        print(f'{G.number_of_edges()} of {n*(n-1)} possible directed edges '
              f'(cut {cut:.4f})')
        cts = trans_matrix.attrs.get('counts')
        top = sorted(((d['signed'], a, b) for a, b, d in G.edges(data=True)),
                     key=lambda t: abs(t[0]), reverse=True)[:8]
        print(f'strongest transitions by {lab}:')
        for w, a, b in top:
            extra = (f'  (n={int(cts.loc[a, b])})' if cts is not None else '')
            print(f'  {a:>4} -> {b:<4} {w:+.3f}{extra}')
    return fig, G
