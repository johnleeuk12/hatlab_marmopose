"""
marmo.transitions -- syllable frequency, transition matrices, context motifs.

Two frequency conventions, following keypoint-moseq:

  instance (runlength)  each bout counts once regardless of duration. This is
                        the MoSeq default, and it stops long-duration syllables
                        from dominating purely by lasting longer.
  frame                 each frame counts equally.

They diverge sharply in practice, so "most common syllable" depends on which
you mean. The difference between them for a syllable is informative about its
typical duration.

Transitions are instance-based: only the moment of switching matters. Because
get_syllable_instances already breaks at NaN frames, consecutive rows of the
instance table are valid transitions, except where a gap intervenes -- context
windows test that with start_frame[i+1] == start_frame[i] + duration[i], which
covers session boundaries, the track1/track2 junction and mid-session gaps in
one check.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter



 
def get_transition_matrix(inst_df, labels=None, normalize='bigram',
                          min_count=0, drop_self=False):
    """
    Bigram transition matrix as a DataFrame indexed and columned by label.
 
    Sized by the number of DISTINCT labels present, not by max(label) + 1, so a
    relabelled table carrying clusters 101-104 and a 199 sentinel gives a
    compact matrix rather than a 200x200 grid of mostly structural zeros. Rows
    and columns are the labels themselves, so it stays addressable through .loc.
 
    normalize
    ---------
    'bigram'      joint probability P(i,j); divides by the total count. The
                  MoSeq default. Frequent syllables dominate, because a syllable
                  with 400 instances simply has more transitions than one with
                  20 -- which is the problem the options below address.
    'rows'        conditional P(j|i). Removes the SOURCE's frequency: "given the
                  animal just left i, where does it go?" Each row sums to 1.
                  Does NOT remove the target's frequency, so every row still
                  points toward whichever syllables are common overall.
    'columns'     conditional P(i|j), the mirror case.
    'enrichment'  observed / expected, where expected is the independence model
                  for a contingency table, outer(row_marginal, col_marginal).
                  Removes BOTH frequencies. 1.0 means exactly as often as the
                  two marginals predict, >1 more often than chance.
    'pmi'         log2 of enrichment. Symmetric around 0 and better behaved for
                  plotting, since enrichment is bounded below by 0 but unbounded
                  above.
    None          raw counts.
 
    min_count : cells with fewer than this many OBSERVED transitions are set to
                NaN. Strongly recommended with 'enrichment' or 'pmi', where a
                rare pair explodes: 2 observed against 0.1 expected gives 20,
                which is noise rather than signal. Around 5 is a reasonable
                floor. Masking rather than zeroing keeps "too rare to judge"
                distinct from "does not happen".
 
    drop_self : zero the diagonal before normalising. Self-transitions are the
                largest entries in a relabelled table, because adjacent
                instances of different syllables that landed in the same cluster
                now share a label. Note this changes the marginals, so the
                enrichment null is then conditioned on a transition having
                occurred between two different labels.
 
    Raw counts are kept in .attrs['counts'] whatever the normalisation.
    """
    syl = inst_df['syllable'].values.astype(int)
    present = sorted(np.unique(syl))
    order = present if labels is None else [int(x) for x in labels]
    missing = [s for s in order if s not in set(present)]
    if missing:
        raise KeyError(f'labels {missing} do not occur in inst_df')
 
    pos = {s: i for i, s in enumerate(present)}
    n = len(present)
    counts = np.zeros((n, n), dtype=float)
    if len(syl) >= 2:
        a = np.fromiter((pos[x] for x in syl[:-1]), int, len(syl) - 1)
        b = np.fromiter((pos[x] for x in syl[1:]),  int, len(syl) - 1)
        np.add.at(counts, (a, b), 1)
    if drop_self:
        np.fill_diagonal(counts, 0.0)
 
    tot = counts.sum()
    P = counts / tot if tot > 0 else counts.copy()
 
    if normalize is None:
        M = counts.copy()
    elif normalize == 'bigram':
        M = P
    elif normalize == 'rows':
        rs = counts.sum(axis=1, keepdims=True)
        M = counts / np.where(rs == 0, 1, rs)
    elif normalize == 'columns':
        cs = counts.sum(axis=0, keepdims=True)
        M = counts / np.where(cs == 0, 1, cs)
    elif normalize in ('enrichment', 'pmi'):
        # independence model for a contingency table: expected(i,j) =
        # P(i as source) * P(j as target). Using the transition marginals rather
        # than instance frequencies is what makes this the correct null -- it
        # already accounts for how often each label appears on each side.
        pr = P.sum(axis=1, keepdims=True)
        pc = P.sum(axis=0, keepdims=True)
        exp = pr @ pc
        with np.errstate(divide='ignore', invalid='ignore'):
            M = np.where(exp > 0, P / exp, np.nan)
        if normalize == 'pmi':
            with np.errstate(divide='ignore', invalid='ignore'):
                M = np.where(M > 0, np.log2(M), np.nan)
    else:
        raise ValueError("normalize must be 'bigram', 'rows', 'columns', "
                         "'enrichment', 'pmi' or None")
 
    if min_count > 0:
        M = np.where(counts >= min_count, M, np.nan)
 
    out = pd.DataFrame(M, index=pd.Index(present, name='from'),
                       columns=pd.Index(present, name='to')).loc[order, order]
    out.attrs['counts'] = pd.DataFrame(
        counts, index=pd.Index(present, name='from'),
        columns=pd.Index(present, name='to')).loc[order, order]
    out.attrs['normalize'] = normalize
    out.attrs['min_count'] = min_count
    return out
 
 
def get_frequencies(inst_df, labels=None, runlength=True):
    """
    Syllable frequency as a Series indexed by label.
 
    runlength=True counts each bout once regardless of duration -- the MoSeq
    default, which stops long syllables dominating purely by lasting longer.
    False weights by frame count. The gap between the two for a syllable is
    informative about its typical duration.
    """
    syl = inst_df['syllable'].values.astype(int)
    w = None if runlength else inst_df['duration_frames'].values.astype(float)
    counts = pd.Series(1.0 if w is None else w,
                       index=syl).groupby(level=0).sum()
    counts = counts / counts.sum()
    counts.index.name = 'syllable'
    return counts if labels is None else counts.reindex([int(x) for x in labels])

def _instance_groups(comp_df):
    """
    Split instances into contiguous runs.

    Returns
    -------
    syl   : (n,) int array of syllable labels
    group : (n,) int array; equal values mean uninterrupted succession
    """
    syl   = comp_df['syllable'].values.astype(int)
    start = comp_df['start_frame'].values.astype(np.int64)
    dur   = comp_df['duration_frames'].values.astype(np.int64)

    if len(syl) < 2:
        return syl, np.zeros(len(syl), dtype=int)

    contiguous = start[1:] == start[:-1] + dur[:-1]
    group = np.concatenate([[0], np.cumsum(~contiguous)])
    return syl, group


def get_context_sequences(comp_df, target, n=2, direction='before',
                          exclude=(99,), top_k=10):
    """
    Count the n-syllable sequences that precede or follow a target syllable.

    Parameters
    ----------
    comp_df   : DataFrame with syllable, start_frame, duration_frames
    target    : int, the syllable whose context is examined
    n         : int, sequence length (2 or 3 is usually readable)
    direction : 'before' or 'after'
    exclude   : iterable of syllables; any window containing one is discarded
    top_k     : int, number of sequences returned

    Returns
    -------
    DataFrame with columns sequence, label, count, frequency, expected, ratio
        frequency : count / n_windows
        expected  : count predicted if syllables were drawn independently
        ratio     : frequency / expected_frequency, so >1 means the sequence
                    occurs more often than its constituent syllables alone
                    would predict
    """
    if direction not in ('before', 'after'):
        raise ValueError("direction must be 'before' or 'after'")

    syl, group = _instance_groups(comp_df)
    exclude = set(exclude)
    n_inst = len(syl)

    # Marginal instance probabilities for the independence null
    counts = Counter(s for s in syl if s not in exclude)
    total  = sum(counts.values())
    p_marg = {s: c / total for s, c in counts.items()} if total else {}

    windows = []
    for i in np.flatnonzero(syl == target):
        if direction == 'before':
            lo, hi = i - n, i
        else:
            lo, hi = i + 1, i + 1 + n

        if lo < 0 or hi > n_inst:
            continue
        # Whole window plus the target must lie in one contiguous run
        if group[lo] != group[i] or group[hi - 1] != group[i]:
            continue

        win = tuple(syl[lo:hi])
        if exclude & set(win):
            continue
        windows.append(win)

    if not windows:
        return pd.DataFrame(columns=['sequence', 'label', 'count',
                                     'frequency', 'expected', 'ratio'])

    tally = Counter(windows)
    n_win = len(windows)

    rows = []
    for win, c in tally.most_common(top_k):
        p_exp = np.prod([p_marg.get(s, 0.0) for s in win])
        label = (' > '.join(map(str, win)) + f' > [{target}]'
                 if direction == 'before'
                 else f'[{target}] > ' + ' > '.join(map(str, win)))
        rows.append({
            'sequence':  win,
            'label':     label,
            'count':     c,
            'frequency': c / n_win,
            'expected':  p_exp * n_win,
            'ratio':     (c / n_win) / p_exp if p_exp > 0 else np.nan,
        })

    out = pd.DataFrame(rows)
    out.attrs['n_windows'] = n_win
    out.attrs['n_target']  = int((syl == target).sum())
    return out


def plot_context_sequences(comp_df, targets=(3, 6), n=2, top_k=10,
                           exclude=(99,), show_expected=True):
    """
    Grid of horizontal histograms: one row per target, columns before / after.

    When show_expected is True, a hollow marker shows the count predicted under
    independence, so a tall bar that merely reflects two common syllables is
    distinguishable from a genuinely stereotyped sequence.
    """
    targets = list(targets)
    fig, axes = plt.subplots(len(targets), 2,
                             figsize=(15, 3.2 * len(targets) + 1),
                             squeeze=False)

    results = {}
    for r, target in enumerate(targets):
        for c, direction in enumerate(['before', 'after']):
            ax = axes[r][c]
            df = get_context_sequences(comp_df, target, n=n,
                                       direction=direction,
                                       exclude=exclude, top_k=top_k)
            results[(target, direction)] = df

            if df.empty:
                ax.text(0.5, 0.5, f'no windows for syllable {target}',
                        ha='center', va='center', fontsize=9)
                ax.set_axis_off()
                continue

            y = np.arange(len(df))[::-1]        # most frequent at the top
            ax.barh(y, df['count'], color='steelblue', edgecolor='none')

            if show_expected:
                ax.scatter(df['expected'], y, s=28, facecolors='none',
                           edgecolors='crimson', linewidths=1.2, zorder=3,
                           label='expected if independent')
                ax.legend(fontsize=7, loc='lower right')

            ax.set_yticks(y)
            ax.set_yticklabels(df['label'], fontsize=8, family='monospace')
            ax.set_xlabel('instance count', fontsize=9)
            ax.set_title(
                f'{n}-syllable sequences {direction} syllable {target}   '
                f'(n={df.attrs["n_windows"]} of '
                f'{df.attrs["n_target"]} occurrences)',
                fontsize=9
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.show()
    return results