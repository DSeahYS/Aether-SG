"""AETHER-SG: offline planning twin over a simulated 200x200 m slice.

Data is synthetic (see aether.slice_data) and results are screening-grade
advisory only.  No regulatory or safety claim is made (Research sec.5).

Exports: build_slice, save_slice, load_slice (slice_data);
build_sdf, sdf_at, sdf_sample, gnss_shadow, shadow_at, shadow_sample,
flood_tensor, score_path, WHEELCHAIR_BLOCK_M, EVAC_REJECT_M, PREPARE_M
(evaluator).
"""

from aether.evaluator import (
    EVAC_REJECT_M,
    PREPARE_M,
    WHEELCHAIR_BLOCK_M,
    build_sdf,
    flood_tensor,
    gnss_shadow,
    score_path,
    sdf_at,
    sdf_sample,
    shadow_at,
    shadow_sample,
)
from aether.slice_data import build_slice, load_slice, save_slice

__all__ = [
    "build_slice", "save_slice", "load_slice",
    "build_sdf", "sdf_at", "sdf_sample",
    "gnss_shadow", "shadow_at", "shadow_sample",
    "flood_tensor", "score_path",
    "WHEELCHAIR_BLOCK_M", "EVAC_REJECT_M", "PREPARE_M",
]
__version__ = "0.1.0"
