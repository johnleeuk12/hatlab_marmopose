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



def get_transition_matrix(inst_df, n_states=100, normalize='bigram'):
    """
    Bigram transition matrix from an instance table.

    normalize : 'bigram' divides by the total count, giving joint
                probabilities P(i,j) -- the MoSeq default, because it keeps
                information about how often each syllable occurs.
                'rows' gives conditional P(j|i), which discards that.
    """
    syl = inst_df['syllable'].values.astype(int)
    trans = np.zeros((n_states, n_states), dtype=float)
    np.add.at(trans, (syl[:-1], syl[1:]), 1)

    if normalize == 'bigram':
        tot = trans.sum()
        if tot > 0:
            trans /= tot
    elif normalize == 'rows':
        rs = trans.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        trans /= rs
    return trans


def get_frequencies(inst_df, n_states=100, runlength=True):
    """Instance-based (runlength) or frame-based syllable frequency."""
    syl = inst_df['syllable'].values.astype(int)
    if runlength:
        counts = np.bincount(syl, minlength=n_states).astype(float)
    else:
        counts = np.bincount(syl, weights=inst_df['duration_frames'].values,
                             minlength=n_states).astype(float)
    return counts / counts.sum()


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