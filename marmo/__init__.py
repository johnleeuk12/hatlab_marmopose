"""
marmo -- 3D pose and behaviour analysis for in-cage marmosets.

Layout
------
    config       structural constants (keypoint indices, FPS, Z_SIGN)
    io           loading arrays, instance tables, MoSeq results
    geometry     egocentric frames, segment lengths, quality masks
    instances    frame sequence -> syllable instances
    posture      trunk elevation and spinal flexion
    dispersion   per-keypoint movement magnitude, motion classes
    clustering   per-instance pose features, UMAP, Ward clustering, relabelling
    transitions  frequencies, transition matrices, context motifs

Dependency order runs upward: config imports nothing, geometry imports config,
everything else imports those. Nothing imports downward, so there are no cycles.

Only config is re-exported here. Nothing is pulled from io, deliberately: a name
missing from io would make __init__ raise and break EVERY import of the package,
including marmo.config. Import io explicitly instead:

    from marmo import io, config
    from marmo.posture import frame_angles, instance_posture

No install is needed as long as the folder containing marmo/ is your working
directory, since Python puts the script's directory on sys.path. `pip install -e .`
only becomes useful when you want to import it from elsewhere.

Typical console session
-----------------------
    %load_ext autoreload
    %autoreload 2

    from marmo import io, config
    from marmo.posture import frame_angles, instance_posture
    from marmo.clustering import instance_pose_features, cluster_instances
"""

__version__ = '0.1.0'

from marmo import config
from marmo.config import (BODYPARTS, TAIL, SPINE, NECK, TAILBASE, SPINEMID,
                          FPS, DT, Z_SIGN, EXCLUDE_SYLLABLE)

__all__ = ['config', 'BODYPARTS', 'TAIL', 'SPINE', 'NECK', 'TAILBASE',
           'SPINEMID', 'FPS', 'DT', 'Z_SIGN', 'EXCLUDE_SYLLABLE']