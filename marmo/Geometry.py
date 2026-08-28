"""
marmo.geometry -- coordinate frames, segment lengths, quality signals.

Shared primitives used by posture, dispersion and clustering. Imports only
config, so it sits at the bottom of the dependency order.
"""

import numpy as np

from marmo.config import TAIL, SPINE, NECK, TRUNK_THR


# -----------------------------------------------------------------------------
# Egocentric frame
# -----------------------------------------------------------------------------

def ego_frame(pts, origin_idx=None, axis_from=TAIL, axis_to=NECK):
    """
    Rotate and translate each frame into a body-centred frame.

    This unifies what were previously two near-duplicate transforms. The only
    difference between them was the origin, which is now a parameter:

      origin_idx=None  origin is the CENTROID of (axis_from, SPINE, axis_to).
                       No keypoint is forced to exactly zero, so all remain
                       informative. Use this for pose feature vectors.
      origin_idx=k     origin is keypoint k. That keypoint then has exactly
                       zero coordinates in every frame, and the axis keypoint
                       is near-zero, which shows up as dead rows in a
                       per-keypoint table.

    The long axis is axis_from->axis_to, defaulting to tailbase->neck: the most
    stable segment in this data (CV 0.20, versus 0.43 for tailbase->spinemid).
    The second axis is horizontal and perpendicular to it. When the body axis is
    near-vertical that horizontal reference degenerates -- the same singularity
    that destabilises a 2D-projected heading for a clinging marmoset -- so world
    +X is substituted for those frames.

    Parameters
    ----------
    pts : (T, K, 3) world coordinates, NaN where missing

    Returns
    -------
    (T, K, 3) in the body frame, axes (u = long axis, v = lateral, w = normal).
    NaN wherever the frame could not be constructed.
    """
    if origin_idx is None:
        o = np.nanmean(pts[:, [axis_from, SPINE, axis_to], :], axis=1)
    else:
        o = pts[:, origin_idx, :]

    u = pts[:, axis_to, :] - pts[:, axis_from, :]
    nu = np.linalg.norm(u, axis=1)
    ok = np.isfinite(nu) & (nu > 1e-6) & np.isfinite(o).all(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        u = u / nu[:, None]

    v = np.cross(np.array([0.0, 0.0, 1.0]), u)
    nv = np.linalg.norm(v, axis=1)
    degen = ~np.isfinite(nv) | (nv < 1e-3)
    v_alt = np.cross(np.array([1.0, 0.0, 0.0]), u)
    nva = np.linalg.norm(v_alt, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        v = np.where(degen[:, None],
                     v_alt / np.maximum(nva, 1e-12)[:, None],
                     v / np.maximum(nv, 1e-12)[:, None])
    w = np.cross(u, v)

    R = np.stack([u, v, w], axis=1)
    out = np.einsum('nij,nkj->nki', R, pts - o[:, None, :])
    out[~ok] = np.nan
    return out


# -----------------------------------------------------------------------------
# Segment lengths
# -----------------------------------------------------------------------------

def segment_length(pts, i, j):
    """Per-frame distance between two keypoints."""
    return np.linalg.norm(pts[:, j, :] - pts[:, i, :], axis=1)


def body_length(pts, i=TAIL, j=NECK):
    """
    Median segment length over the whole recording, as a scale unit.

    A single constant is used rather than a per-frame length so that noise in
    the segment is not injected into every normalised keypoint.
    """
    return float(np.nanmedian(segment_length(pts, i, j)))


def segment_quality(pts, i=TAIL, j=SPINE):
    """
    Fractional deviation of a segment's length from its own median.

    tailbase->spinemid is bone-to-bone and should be constant, so deviation is
    a per-frame localization-quality signal rather than a property of posture.
    Empirically the error is 5.5x larger in frames adjacent to a tracking
    dropout than in frames far from one, and reaches 145% for head-down frames,
    which are therefore mostly tracking failure rather than real posture.

    Note that elevation depends only on the segment's DIRECTION, not its length.
    Length error does not propagate arithmetically into the angle; it is a proxy
    for a mislocalized endpoint, which does rotate the vector.
    """
    L = segment_length(pts, i, j)
    med = np.nanmedian(L)
    return np.abs(L - med) / med


def quality_mask(pts, trunk_thr=TRUNK_THR, upper=False):
    """
    Boolean per-frame mask of usable frames.

    upper=False  gates on tailbase->spinemid only. Sufficient for elevation,
                 which is measured on that segment.
    upper=True   additionally gates on spinemid->neck. Required for flexion,
                 which is the angle AT spinemid and so depends on both
                 segments; a mislocalized neck corrupts it even when the trunk
                 is fine. Costs roughly 79% -> 63% retention.
    """
    ok = segment_quality(pts, TAIL, SPINE) < trunk_thr
    if upper:
        ok &= segment_quality(pts, SPINE, NECK) < trunk_thr
    return ok


# -----------------------------------------------------------------------------
# Robust dispersion primitive
# -----------------------------------------------------------------------------

def robust_dispersion(P, min_frames):
    """
    Median distance from the median position -- a 3D analogue of the MAD.

    P : (n, 3), may contain NaN rows. Returns NaN below min_frames valid rows.

    Chosen over variance because MAD has a 50% breakdown point (the maximum
    possible for a scale estimator) whereas variance has 0: one mislabelled
    frame moves it without limit, since squared deviations grow quadratically.
    The cost is low Gaussian efficiency, about 37%, so roughly 2.7x the samples
    are needed for equal precision on clean data.

    The centre is the marginal (per-axis) median. The statistically proper 3D
    centre is the geometric median, but the difference is small for a compact
    cloud and the marginal version is non-iterative.

    No 1.4826 scale factor is applied: that constant makes the 1D MAD a
    consistent estimator of sigma under normality and has no 3D equivalent.
    """
    ok = np.isfinite(P).all(axis=1)
    if ok.sum() < min_frames:
        return np.nan
    Q = P[ok]
    c = np.median(Q, axis=0)
    return float(np.median(np.linalg.norm(Q - c, axis=1)))


def median_mad(x, min_n=3):
    """Median and median absolute deviation of a 1D array, ignoring NaN."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < min_n:
        return np.nan, np.nan
    med = float(np.median(x))
    return med, float(np.median(np.abs(x - med)))