"""
marmo.config -- structural constants.

These describe the DATA FORMAT, not analysis choices. Changing one of them
alone would make the rest of the package silently inconsistent, so they are
module constants read by the functions rather than parameters passed per call.

Analysis choices (paths, which syllables count as stationary, k, window
lengths) belong outside the package -- in your console or a project-level
settings.py -- and are passed to functions as arguments. The values below that
appear as function defaults (TRUNK_THR, MIN_FRAMES, ...) are overridable at the
call site; the keypoint indices are not.
"""

# --- keypoint layout of the MarmoPose output ---------------------------------
BODYPARTS = [
    'head', 'leftear', 'rightear', 'neck',
    'leftelbow', 'rightelbow', 'lefthand', 'righthand',
    'spinemid', 'tailbase', 'leftknee', 'rightknee',
    'leftfoot', 'rightfoot', 'tailmid', 'tailend',
]
N_KEYPOINTS = len(BODYPARTS)

HEAD, LEFTEAR, RIGHTEAR, NECK          = 0, 1, 2, 3
LEFTELBOW, RIGHTELBOW                  = 4, 5
LEFTHAND, RIGHTHAND                    = 6, 7
SPINEMID, TAILBASE                     = 8, 9
LEFTKNEE, RIGHTKNEE                    = 10, 11
LEFTFOOT, RIGHTFOOT                    = 12, 13
TAILMID, TAILEND                       = 14, 15

# short aliases used throughout
TAIL, SPINE = TAILBASE, SPINEMID

# tail tip keypoints are too noisy to carry information; excluded by default
DEFAULT_EXCLUDE_BODYPARTS = ('tailmid', 'tailend')

TRUNK_KEYPOINTS  = ('tailbase', 'spinemid')
DISTAL_KEYPOINTS = ('lefthand', 'righthand', 'leftfoot', 'rightfoot', 'head')

# --- acquisition ------------------------------------------------------------
FPS = 25
DT  = 1.0 / FPS

# MarmoPose world coordinates increase DOWNWARD, so elevation needs the sign
# flipped. Verified empirically: median(z_neck - z_tailbase) = -89.1 in raw
# coordinates, with only 3% of frames head-up -- impossible for a marmoset.
# With Z_SIGN = -1 that becomes 97% head-up.
Z_SIGN = -1

# Frames inserted at each session/track boundary by preprocess.py
N_BOUNDARY_FRAMES = 5

# Low-frequency catch-all syllable label
EXCLUDE_SYLLABLE = 99

# --- default analysis thresholds (overridable per call) ---------------------
# tailbase->spinemid is bone-to-bone and should be constant, but measures
# 50-130 mm (CV 0.43). Fractional deviation from its median is a per-frame
# localization-quality signal: error is 5.5x worse adjacent to a tracking
# dropout than far from one. 0.25 retains ~79% of instances.
TRUNK_THR    = 0.25
MIN_FRAMES   = 3      # valid frames needed for a per-instance estimate
MIN_INSTANCES = 5     # instances needed to report a syllable
FOLD_ABS     = True   # fold head-down elevations onto the positive side
TRANS_RANGE  = 20.0   # within-instance elevation range flagging a transition