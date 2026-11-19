"""AETHER-SG scientific metrics + LOS/swath engine (screening grade).

Computes flight, line-of-sight / swath, environment and ground-route
metrics over the real 200x200 m slice (slice coords: x east 0..200,
y north 0..200, z AGL).  The browser viewer uses scene-centred coords
(viewer = slice - 100) with drone points [x, alt, z]; backend helpers
below convert explicitly so the two frames are never confused.

All physics is screening grade (Research-AETHER-SG sec.6): SDF lookup
is a box-approximated clearance proxy, flood is a bathtub screening
tensor (PUB owns operational hydrology), GNSS shadow is an
occlusion surrogate (not ray-traced ephemeris), wind is not modelled
(``wind`` is None placeholder).  Routes are advisory, not for
navigation.  Google-derived capture layers are research-only.

Deps: Python 3.11+ stdlib + numpy only.  Sibling modules
(real_slice / routing / evaluator) are imported lazily inside
functions so this file compiles standalone.

Coordinate frames
-----------------
- Backend slice: x east 0..200, y north 0..200, z AGL 0..60 m.
  Buildings: {cx, cy, w, d, height_m} boxes from z=0 to z=height_m
  (ground datum = DEM minimum = 0).  DEM: (H, W) @ 1 m, index
  [iy, ix] with iy north, row 0 = south edge.
- Viewer / capture.json: x east -100..100, z north -100..100,
  drone points [x, alt, z] (alt second).  ``viewer = slice - 100``.

Energy assumption (IMPORTANT, screening only)
---------------------------------------------
``energy_proxy`` is the routing.py drone surrogate (unitless work index):

    energy_proxy = 0.10 * length_m + 0.05 * total_climb_m

mirroring ``drone_astar`` (0.10 per metre + 0.05 per metre climb).
``est_energy_wh = energy_proxy * 0.9 * 1000 / 3600`` i.e. 0.25 Wh per
proxy unit.  This assumes a 900 W mean electrical draw scaled by the
proxy-hours; it is an order-of-magnitude screening estimate only, not
a battery / propulsion model.  The factor is kept exactly as specified
(``energy_proxy * 0.9 / 3600 * 1000``).

Score breakdown mirrors the TRUE scalar in scripts/run_real.py::

    total = -(1.0*time_s + 0.3*energy_proxy + 0.4*shadow_exposure_m)

with wind contributing 0 until wind data exists.  Components are
signed penalties (more negative = worse) so they sum to ``total``.

Public API
----------
flight_metrics(path_xyz, buildings, sdf_bundle, shadow_bundle, dem)
los_swath_frames(path_xyz, buildings, target_xy, dem, fov_deg=60)
environment_metrics(slice_dict, flood_depth)
ground_metrics(route_result, edges_used=None)
viewer <-> slice converters + load_capture_json()
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

EXTENT_M = 200.0
DRONE_SPEED_MPS = 12.0       # 0-60 m cruise for time estimate
WHEELCHAIR_SPEED_MPS = 1.1   # mixed-surface wheelchair cruise
WHEELCHAIR_BLOCK_M = 0.10    # [INFERENCE] wheelchair flood block (screening)
ENERGY_PER_M = 0.10          # routing.py surrogate (per metre)
ENERGY_CLIMB_PER_M = 0.05    # routing.py surrogate (per metre climb)
TRUE_TIME_W = 1.0            # run_real.py TRUE scalar weights
TRUE_ENERGY_W = 0.3
TRUE_SHADOW_W = 0.4
TRUE_WIND_W = 0.2
LOS_STEP_M = 1.0             # LOS sampling step (metres)
COVERAGE_RES_M = 2.0         # swath coverage grid (metres)
DENSIFY_STEP_M = 1.0         # flight-metric densification step
ROUTE_RES_M = 4.0            # routing.py grid resolution (ndarray bundles)
EVAL_VOXEL_M = 2.0           # evaluator.py voxel size (bundle dicts)

__all__ = [
    "flight_metrics",
    "los_swath_frames",
    "environment_metrics",
    "ground_metrics",
    "viewer_to_slice_xy",
    "slice_to_viewer_xy",
    "viewer_drone_to_slice",
    "slice_drone_to_viewer",
    "capture_buildings_to_slice",
    "load_capture_json",
]


# ---------------------------------------------------------------------------
# Frame converters (viewer = slice - 100)
# ---------------------------------------------------------------------------

def viewer_to_slice_xy(xv: float, zv: float) -> tuple[float, float]:
    """Viewer (x, z) scene-centred -100..100 -> slice (x, y) 0..200."""
    return (float(xv) + 100.0, float(zv) + 100.0)


def slice_to_viewer_xy(x: float, y: float) -> tuple[float, float]:
    """Slice (x, y) 0..200 -> viewer (x, z) scene-centred -100..100."""
    return (float(x) - 100.0, float(y) - 100.0)


def viewer_drone_to_slice(points):
    """Viewer drone points [x, alt, z] -> slice (x, y, z AGL) list.

    The altitude is the SECOND component in capture.json; it becomes the
    THIRD component in backend order.  This reorder is load-bearing:
    swapping it silently flies the path on its side.
    """
    out = []
    for p in points:
        xv, alt, zv = float(p[0]), float(p[1]), float(p[2])
        out.append([xv + 100.0, zv + 100.0, alt])
    return out


def slice_drone_to_viewer(points):
    """Slice (x, y, z AGL) -> viewer [x, alt, z] (inverse of above)."""
    out = []
    for p in points:
        xs, ys, zs = float(p[0]), float(p[1]), float(p[2])
        out.append([xs - 100.0, zs, ys - 100.0])
    return out


def capture_buildings_to_slice(buildings) -> list[dict]:
    """capture.json buildings {x,z,w,d,h} -> slice {cx,cy,w,d,height_m}."""
    out = []
    for i, b in enumerate(buildings or []):
        if "cx" in b and "height_m" in b:
            out.append(dict(b))
            continue
        out.append({
            "id": int(b.get("id", i)),
            "cx": float(b["x"]) + 100.0,
            "cy": float(b["z"]) + 100.0,
            "w": float(b["w"]),
            "d": float(b["d"]),
            "height_m": float(b["h"]),
        })
    return out


def load_capture_json(path) -> dict:
    """Read viewer/capture/capture.json (real capture only, never simulated)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Internal helpers (numpy only)
# ---------------------------------------------------------------------------

def _dem_at(dem, x: float, y: float) -> float:
    """Nearest-cell DEM lookup, index [iy, ix], iy north, row 0 = south."""
    dem = np.asarray(dem)
    h, w = dem.shape
    ix = int(np.clip(round(float(x)), 0, w - 1))
    iy = int(np.clip(round(float(y)), 0, h - 1))
    return float(dem[iy, ix])


def _building_arrays(buildings):
    """(cx, cy, w, d, h) float arrays from slice-format building dicts."""
    b = list(buildings or [])
    if not b:
        z = np.zeros(0, dtype=np.float64)
        return z, z.copy(), z.copy(), z.copy(), z.copy()
    cx = np.array([float(d.get("cx", d.get("x"))) for d in b], dtype=np.float64)
    cy = np.array([float(d.get("cy", d.get("z", d.get("y")))) for d in b],
                  dtype=np.float64)
    w = np.array([float(d.get("w", 0.0)) for d in b], dtype=np.float64)
    dd = np.array([float(d.get("d", 0.0)) for d in b], dtype=np.float64)
    h = np.array([float(d.get("height_m", d.get("h", 0.0))) for d in b],
                 dtype=np.float64)
    return cx, cy, w, dd, h


def _box_sdf_values(px, py, pz, buildings):
    """Analytic signed distance to nearest box for (N,) points.

    Boxes are axis-aligned, z in [0, height_m] (evaluator formula:
    outside = norm(max(q,0)), inside = min(max(q),0)).  Returns
    (N, M) per-building distances and (N,) min + argmin index.
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64)
    cx, cy, w, dd, h = _building_arrays(buildings)
    if cx.size == 0:
        # No buildings: clearance is the altitude itself (ground plane).
        d = np.zeros((px.shape[0], 0), dtype=np.float64)
        return d, pz.copy(), np.full(px.shape[0], -1, dtype=np.int64)
    qx = np.abs(px[:, None] - cx[None, :]) - (w / 2.0)[None, :]
    qy = np.abs(py[:, None] - cy[None, :]) - (dd / 2.0)[None, :]
    qz = pz[:, None] - h[None, :]
    ox = np.maximum(qx, 0.0)
    oy = np.maximum(qy, 0.0)
    oz = np.maximum(qz, 0.0)
    outside = np.sqrt(ox**2 + oy**2 + oz**2)
    inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    d = outside + inside
    idx = np.argmin(d, axis=1)
    return d, d[np.arange(d.shape[0]), idx], idx


def _analytic_sdf(pts, buildings):
    """Min box distance per point (no voxel quantisation)."""
    pts = np.asarray(pts, dtype=np.float64)
    _, mins, _ = _box_sdf_values(pts[:, 0], pts[:, 1], pts[:, 2], buildings)
    return mins


def _sample_sdf(bundle, pts, buildings):
    """Sample SDF at (N, 3) points; accepts evaluator/routing bundles.

    - dict bundle (evaluator): ``{"sdf": (nz,ny,nx), "voxel_m": ...}``
    - ndarray bundle: routing (nx,ny,nz) @ 4 m or evaluator (nz,ny,nx)
      @ 2 m, auto-detected from shape (50,50,16 vs 30,100,100)
    - None: analytic box SDF (building-only, no voxel quantisation)

    Returns (N,) float clearance; negative means inside a building box.
    The evaluator grid is ground-plane inclusive (min with z), so far
    from buildings the value equals the altitude, like evaluator.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if bundle is None:
        return _analytic_sdf(pts, buildings)
    arr, voxel, layout = _bundle_array(bundle, "sdf")
    if layout == "routing":
        nx, ny, nz = arr.shape
        ix = np.clip((pts[:, 0] / voxel).astype(np.int64), 0, nx - 1)
        iy = np.clip((pts[:, 1] / voxel).astype(np.int64), 0, ny - 1)
        iz = np.clip((pts[:, 2] / voxel).astype(np.int64), 0, nz - 1)
        return arr[ix, iy, iz].astype(np.float64)
    nz, ny, nx = arr.shape
    ix = np.clip((pts[:, 0] / voxel).astype(np.int64), 0, nx - 1)
    iy = np.clip((pts[:, 1] / voxel).astype(np.int64), 0, ny - 1)
    iz = np.clip((pts[:, 2] / voxel).astype(np.int64), 0, nz - 1)
    return arr[iz, iy, ix].astype(np.float64)


def _sample_shadow(bundle, pts):
    """Sample GNSS shadow at (N, 3) points; returns (N,) float in [0, 1]."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if bundle is None:
        return np.zeros(pts.shape[0], dtype=np.float64)
    arr, voxel, layout = _bundle_array(bundle, "shadow")
    if layout == "routing":
        nx, ny, nz = arr.shape
        ix = np.clip((pts[:, 0] / voxel).astype(np.int64), 0, nx - 1)
        iy = np.clip((pts[:, 1] / voxel).astype(np.int64), 0, ny - 1)
        iz = np.clip((pts[:, 2] / voxel).astype(np.int64), 0, nz - 1)
        return arr[ix, iy, iz].astype(np.float64)
    nz, ny, nx = arr.shape
    ix = np.clip((pts[:, 0] / voxel).astype(np.int64), 0, nx - 1)
    iy = np.clip((pts[:, 1] / voxel).astype(np.int64), 0, ny - 1)
    iz = np.clip((pts[:, 2] / voxel).astype(np.int64), 0, nz - 1)
    return arr[iz, iy, ix].astype(np.float64)


def _bundle_array(bundle, key: str):
    """Normalise a bundle to (array, voxel_m, layout)."""
    if isinstance(bundle, dict):
        if key in bundle:
            arr = np.asarray(bundle[key])
            voxel = float(bundle.get("voxel_m", EVAL_VOXEL_M))
            layout = "routing" if tuple(arr.shape)[2] <= 32 and tuple(arr.shape)[0] == tuple(arr.shape)[1] else "evaluator"
            return arr, voxel, layout
        # dict without the key: fall back to analytic/zeros upstream
        raise KeyError(f"shadow/SDF bundle missing key {key!r}")
    arr = np.asarray(bundle)
    if arr.ndim != 3:
        raise ValueError(f"bundle array must be 3-D, got shape {arr.shape}")
    # routing (nx,ny,nz) = (50,50,16) vs evaluator (nz,ny,nx) = (30,100,100)
    if arr.shape[0] == arr.shape[1] and arr.shape[2] <= 32:
        return arr, ROUTE_RES_M, "routing"
    return arr, EVAL_VOXEL_M, "evaluator"


def _densify_path(pts, step: float = DENSIFY_STEP_M, cap: int = 4000):
    """Resample polyline at ~``step`` metre spacing (keeps vertices).

    Returns (M, 3) array with M < cap; identical to evaluator densify
    so a thin box is never skipped over between samples.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(seg.sum())
    if length <= 0:
        return pts.copy()
    n = int(np.clip(math.ceil(length / step) + 1, pts.shape[0], cap))
    t = np.linspace(0.0, 1.0, n)
    old = np.linspace(0.0, 1.0, pts.shape[0])
    return np.column_stack([np.interp(t, old, pts[:, k]) for k in range(3)])


def _path_length(pts) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _as_xyz(path_xyz, dem=None):
    """Coerce path to (N, 3) float; lift (N, 2) ground points via DEM."""
    pts = np.asarray(path_xyz, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.shape[1] == 2:
        if dem is None:
            raise ValueError("2-D path needs a DEM to lift z")
        z = np.array([_dem_at(dem, p[0], p[1]) for p in pts])
        pts = np.column_stack([pts, z])
    if pts.shape[1] != 3:
        raise ValueError(f"path must be (N,2) or (N,3), got {pts.shape}")
    for k in range(3):
        if not np.all(np.isfinite(pts[:, k])):
            raise ValueError("path contains non-finite coordinates")
    if pts.shape[0] < 2:
        raise ValueError("path needs at least 2 points")
    return pts


# ---------------------------------------------------------------------------
# (1) Flight metrics
# ---------------------------------------------------------------------------

def flight_metrics(path_xyz_slice_coords, buildings=None, sdf_bundle=None,
                   shadow_bundle=None, dem=None) -> dict:
    """Scientific flight metrics for one drone path (screening grade).

    The positional contract is (path, buildings, sdf_bundle,
    shadow_bundle, dem); the trailing four are optional so a bare
    path-only call still yields length / time / altitude / climb.

    Args:
        path_xyz_slice_coords: (N, 3) slice coords (x, y, z AGL); (N, 2)
            ground paths are lifted via ``dem``.
        buildings: slice-format boxes ({cx,cy,w,d,height_m}); may be
            viewer-format ({x,z,w,d,h}) — both are accepted.
        sdf_bundle: evaluator bundle dict, routing ndarray, or None for
            analytic box SDF (building-only clearance).
        shadow_bundle: evaluator bundle dict, routing ndarray, or None
            (then shadow exposure is 0 %).
        dem: (H, W) ground DEM @ 1 m (only used to lift 2-D paths).

    Returns dict with length_m, time_s, energy_proxy, est_energy_wh,
    min_clearance_m, mean_clearance_m, altitude_min/max/mean_m,
    total_climb_m, shadow_exposure_pct, shadow_exposure_m, collisions,
    wind (None placeholder), score / score_components breakdown.
    """
    pts = _as_xyz(path_xyz_slice_coords, dem=dem)
    buildings = list(buildings or [])
    length = _path_length(pts)
    time_s = length / DRONE_SPEED_MPS if length > 0 else 0.0
    dz = np.diff(pts[:, 2])
    total_climb = float(np.clip(dz, 0.0, None).sum())
    energy_proxy = ENERGY_PER_M * length + ENERGY_CLIMB_PER_M * total_climb
    # Specified screening conversion: proxy * 0.9 / 3600 * 1000 = * 0.25 Wh.
    est_energy_wh = energy_proxy * 0.9 / 3600.0 * 1000.0

    dense = _densify_path(pts)
    sdf_vals = _sample_sdf(sdf_bundle, dense, buildings)
    sh_vals = _sample_shadow(shadow_bundle, dense)
    # Arc-length weighting on the densified polyline (uniform ~1 m steps,
    # so the sample fraction and the length fraction nearly coincide).
    seg_w = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    if seg_w.size and float(seg_w.sum()) > 0:
        shadow_m = float((sh_vals[:-1] * seg_w).sum())
    else:
        shadow_m = 0.0
    shadow_pct = 100.0 * shadow_m / length if length > 0 else 0.0
    min_clear = float(sdf_vals.min()) if sdf_vals.size else 0.0
    mean_clear = float(sdf_vals.mean()) if sdf_vals.size else 0.0
    collisions = int((sdf_vals < 0).sum())

    comp_time = -TRUE_TIME_W * time_s
    comp_energy = -TRUE_ENERGY_W * energy_proxy
    comp_shadow = -TRUE_SHADOW_W * shadow_m
    comp_wind = 0.0  # no wind field in this slice: placeholder
    total = comp_time + comp_energy + comp_shadow + comp_wind
    score = {"time": float(comp_time), "energy": float(comp_energy),
             "shadow": float(comp_shadow), "wind": float(comp_wind),
             "total": float(total)}
    return {
        "length_m": float(length),
        "time_s": float(time_s),
        "energy_proxy": float(energy_proxy),
        "est_energy_wh": float(est_energy_wh),
        "min_clearance_m": min_clear,
        "mean_clearance_m": mean_clear,
        "min_sdf_m": min_clear,  # alias of capture.json naming
        "altitude_min_m": float(pts[:, 2].min()),
        "altitude_max_m": float(pts[:, 2].max()),
        "altitude_mean_m": float(pts[:, 2].mean()),
        "max_alt_m": float(pts[:, 2].max()),  # alias
        "climb_m": total_climb,  # alias
        "total_climb_m": total_climb,
        "shadow_exposure_pct": float(shadow_pct),
        "shadow_exposure_m": float(shadow_m),
        "collisions": collisions,
        "wind": None,  # placeholder: no wind data in this slice
        "score": dict(score),
        "score_components": dict(score),
        "total_score": float(total),
    }


# ---------------------------------------------------------------------------
# (2) LOS + swath frames
# ---------------------------------------------------------------------------

def _los_check(drone, target, buildings):
    """LOS drone->target sampled vs boxes; returns (clear, min, blocker)."""
    drone = np.asarray(drone, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dist = float(np.linalg.norm(target - drone))
    n = max(2, int(math.ceil(dist / LOS_STEP_M)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    samp = drone[None, :] + t * (target - drone)[None, :]
    dmat, mins, idx = _box_sdf_values(samp[:, 0], samp[:, 1], samp[:, 2],
                                      buildings)
    if not buildings:
        return True, float(mins.min()) if mins.size else 1e3, None
    k = int(np.argmin(mins))
    best = float(mins[k])
    blocker = int(idx[k]) if int(idx[k]) >= 0 else None
    clear = bool(best > 1e-9)
    return clear, best, (blocker if not clear else None)


def los_swath_frames(path_xyz, buildings, target_xy, dem, fov_deg=60):
    """Per-point LOS + swath frames along ``path_xyz`` (slice coords).

    For every path point:

    - ``los_clear`` / ``los_min_clearance_m`` / ``los_blocked_by``: the
      drone->target segment is sampled every ~1 m and the minimum
      analytic box SDF (evaluator formula, buildings only — DEM terrain
      masking is not modelled) decides clearance.  ``los_blocked_by`` is
      the index into ``buildings`` of the closest box, or None.
    - ``swath_radius_m = alt_agl * tan(fov_deg / 2)`` and
      ``swath_area_m2 = pi * radius**2`` (full-disc approximation;
      ``alt_agl`` is the path z, which is AGL by backend convention).
    - ``coverage_pct`` (alias ``cumulative_coverage_pct``): union
      approximation of the nadir swath discs rasterised on a 2 m grid
      over the 200x200 slice, accumulated in path order.

    The camera is assumed nadir-pointing (disc centred on drone (x, y));
    tilt is not modelled.  Target height is DEM at ``target_xy``;
    drone absolute height is DEM at drone + alt_agl.

    Returns a list of dicts, one per path point, in path order.
    """
    if not (0.0 < float(fov_deg) < 180.0):
        raise ValueError("fov_deg must be in (0, 180)")
    pts = _as_xyz(path_xyz, dem=dem)
    buildings = list(buildings or [])
    tx, ty = float(target_xy[0]), float(target_xy[1])
    target_z = _dem_at(dem, tx, ty) if dem is not None else 0.0
    tan_half = math.tan(math.radians(float(fov_deg) / 2.0))

    n_cov = int(round(EXTENT_M / COVERAGE_RES_M))  # 100 cells per side
    centers = (np.arange(n_cov) + 0.5) * COVERAGE_RES_M
    gx, gy = np.meshgrid(centers, centers)  # (ny, nx), row 0 = south
    covered = np.zeros((n_cov, n_cov), dtype=bool)
    total_cells = n_cov * n_cov

    frames = []
    for i, p in enumerate(pts):
        x, y, alt = float(p[0]), float(p[1]), float(p[2])
        drone_abs = (x, y, (_dem_at(dem, x, y) if dem is not None else 0.0) + alt)
        clear, minc, blocked = _los_check(drone_abs, (tx, ty, target_z),
                                          buildings)
        alt_agl = max(float(alt), 0.0)
        radius = alt_agl * tan_half
        area = math.pi * radius * radius
        if radius > 0:
            covered |= ((gx - x) ** 2 + (gy - y) ** 2) <= radius * radius
        pct = 100.0 * float(covered.sum()) / float(total_cells)
        frames.append({
            "index": int(i),
            "x": x,
            "y": y,
            "z": alt,
            "alt_agl_m": float(alt_agl),
            "los_clear": bool(clear),
            "los_min_clearance_m": float(minc),
            "los_blocked_by": blocked,
            "swath_radius_m": float(radius),
            "swath_area_m2": float(area),
            "coverage_pct": float(pct),
            "cumulative_coverage_pct": float(pct),
            "covered_cells": int(covered.sum()),
        })
    return frames


def summarize_frames(frames) -> dict:
    """Compact summary for CLI: clear %, clearance, coverage, swath."""
    frames = list(frames or [])
    if not frames:
        return {"n_frames": 0, "los_clear_pct": 0.0, "final_coverage_pct": 0.0}
    n = len(frames)
    clear_n = sum(1 for f in frames if f["los_clear"])
    mins = np.array([f["los_min_clearance_m"] for f in frames])
    radii = np.array([f["swath_radius_m"] for f in frames])
    areas = np.array([f["swath_area_m2"] for f in frames])
    return {
        "n_frames": int(n),
        "los_clear_pct": float(100.0 * clear_n / n),
        "los_blocked_count": int(n - clear_n),
        "los_min_clearance_mean_m": float(mins.mean()),
        "los_min_clearance_min_m": float(mins.min()),
        "final_coverage_pct": float(frames[-1]["cumulative_coverage_pct"]),
        "swath_radius_mean_m": float(radii.mean()),
        "swath_area_mean_m2": float(areas.mean()),
        "swath_area_max_m2": float(areas.max()),
    }


# ---------------------------------------------------------------------------
# (3) Environment metrics
# ---------------------------------------------------------------------------

def _flood_depth_array(flood_depth) -> np.ndarray:
    """Coerce flood input to a (H, W) depth array."""
    if flood_depth is None:
        return None
    if isinstance(flood_depth, dict):
        if "depth" in flood_depth:
            return np.asarray(flood_depth["depth"], dtype=np.float64)
        if "flood" in flood_depth:
            return np.asarray(flood_depth["flood"], dtype=np.float64)
        raise KeyError("flood dict needs 'depth'")
    return np.asarray(flood_depth, dtype=np.float64)


def _gnss_at_12m_pct(slice_dict) -> float:
    """Mean GNSS shadow at 12 m as a percentage (screening surrogate).

    Resolution order: routing ``gnss_shadow`` layer @ 12 m (iz=3 @ 4 m),
    evaluator ``gnss_shadow`` bundle layer @ 12 m, capture 41x41 ``gnss``
    grid, else footprint-below-roofline fallback.  Always returns %.
    """
    sh = slice_dict.get("gnss_shadow", slice_dict.get("gnss")) \
        if isinstance(slice_dict, dict) else None
    if sh is not None:
        try:
            if isinstance(sh, dict) and "values" in sh:
                vals = np.asarray(sh["values"], dtype=np.float64)
                return float(100.0 * float(vals.mean()))
            if isinstance(sh, dict) and "shadow" in sh:
                arr = np.asarray(sh["shadow"], dtype=np.float64)
                vox = float(sh.get("voxel_m", EVAL_VOXEL_M))
                iz = int(np.clip(int(12.0 / vox), 0, arr.shape[0] - 1))
                return float(100.0 * float(arr[iz].mean()))
            arr = np.asarray(sh, dtype=np.float64)
            if arr.ndim == 3:
                if arr.shape[0] == arr.shape[1] and arr.shape[2] <= 32:
                    iz = int(np.clip(3, 0, arr.shape[2] - 1))  # 12 m @ 4 m
                    return float(100.0 * float(arr[:, :, iz].mean()))
                iz = int(np.clip(int(12.0 / EVAL_VOXEL_M), 0, arr.shape[0] - 1))
                return float(100.0 * float(arr[iz].mean()))
            if arr.ndim == 2:
                return float(100.0 * float(arr.mean()))
        except (TypeError, ValueError, KeyError, IndexError):
            pass
    # Fallback: ground cells inside a footprint taller than 12 m.
    dem = np.asarray(slice_dict.get("dem", np.zeros((200, 200))),
                     dtype=np.float64)
    h, w = dem.shape
    buildings = slice_dict.get("buildings", []) if isinstance(slice_dict, dict) else []
    cx, cy, bw, bd, bh = _building_arrays(buildings)
    if cx.size == 0:
        return 0.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    inside = np.zeros((h, w), dtype=bool)
    for k in range(cx.size):
        if bh[k] <= 12.0:
            continue
        inside |= ((np.abs(xs - cx[k]) <= bw[k] / 2.0)
                   & (np.abs(ys - cy[k]) <= bd[k] / 2.0))
    return float(100.0 * float(inside.mean()))


def _flood_bathtub_connectivity(dem, water_level_m):
    """Bathtub + connectivity screening flood (mirrors evaluator).

    Depth = clip(water - dem, 0) kept only on cells 4-connected to the
    south seed band (rows 0..11), so a disconnected depression stays dry.
    Returns (depth, wet) float64/bool arrays.  Screening only.
    """
    from collections import deque

    dem = np.asarray(dem, dtype=np.float64)
    depth_raw = np.clip(float(water_level_m) - dem, 0.0, None)
    candidate = depth_raw > 0
    h, w = dem.shape
    wet = np.zeros((h, w), dtype=bool)
    q = deque()
    for iy in range(min(12, h)):
        for ix in range(w):
            if candidate[iy, ix] and not wet[iy, ix]:
                wet[iy, ix] = True
                q.append((iy, ix))
    while q:
        iy, ix = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cy, cx = iy + dy, ix + dx
            if (0 <= cy < h and 0 <= cx < w and candidate[cy, cx]
                    and not wet[cy, cx]):
                wet[cy, cx] = True
                q.append((cy, cx))
    depth = np.where(wet, depth_raw, 0.0)
    return depth, wet


def _flood_result_dict(depth, wet) -> dict:
    """Flood-style dict understood by generic flood consumers."""
    depth = np.asarray(depth, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    blocked_n = int((depth >= WHEELCHAIR_BLOCK_M).sum())
    wet_n = int(wet.sum())
    return {
        "depth": depth,
        "wet": wet,
        "blocked_cells": blocked_n,
        "wet_cells": wet_n,
        "flood_blocked_cells": blocked_n,
        "flood_wet_cells": wet_n,
        "meta": {"wet_cells": wet_n, "blocked_cells": blocked_n,
                 "max_depth_m": float(depth.max()) if depth.size else 0.0,
                 "label": "bathtub+connectivity screening"},
    }


def environment_metrics(slice_dict, flood_depth=None) -> dict:
    """Environment summary for one 200x200 slice (screening grade).

    Args:
        slice_dict: real_slice.build_real_slice() output (dem +
            buildings); may also carry ``gnss_shadow`` / ``gnss``.
            As a convenience overload, a bare (H, W) DEM array is also
            accepted together with a numeric water level, returning the
            bathtub+connectivity flood result for that level.
        flood_depth: (H, W) depth array, evaluator flood dict, numeric
            water level (computed via connectivity above), or None.

    Returns dem_min/max/mean_m, relief_m, building_count, height
    mean/max, footprint union coverage % (rasterised, overlaps counted
    once), flood wet / blocked (>= 0.10 m) cell counts and the mean
    GNSS shadow at 12 m as a percentage.  ``blocked_cells`` /
    ``wet_cells`` / ``meta`` / ``wet`` / ``depth`` aliases are included
    so generic flood consumers can read the same result.
    """
    # Bare-DEM overload: environment_metrics(dem, water_level).
    if not isinstance(slice_dict, dict):
        dem_arr = np.asarray(slice_dict, dtype=np.float64)
        if isinstance(flood_depth, (int, float)) and not isinstance(
                flood_depth, bool):
            depth, wet = _flood_bathtub_connectivity(dem_arr, flood_depth)
            return _flood_result_dict(depth, wet)
        if isinstance(flood_depth, np.ndarray) or isinstance(flood_depth, dict):
            depth = _flood_depth_array(flood_depth)
            return _flood_result_dict(depth, depth > 0)
        raise TypeError("environment_metrics needs a slice dict or "
                        "(dem, water_level)")
    if "dem" not in slice_dict:
        raise KeyError("slice_dict needs 'dem'")
    dem = np.asarray(slice_dict["dem"], dtype=np.float64)
    buildings = list(slice_dict.get("buildings", []) or [])
    cx, cy, bw, bd, bh = _building_arrays(buildings)

    h, w = dem.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    inside = np.zeros((h, w), dtype=bool)
    for k in range(cx.size):
        inside |= ((np.abs(xs - cx[k]) <= bw[k] / 2.0)
                   & (np.abs(ys - cy[k]) <= bd[k] / 2.0))

    if isinstance(flood_depth, (int, float)) and not isinstance(
            flood_depth, bool):
        depth, wet = _flood_bathtub_connectivity(dem, flood_depth)
    else:
        depth = _flood_depth_array(flood_depth)
        wet = (depth > 0) if depth is not None else None
    if depth is None:
        wet_n = blocked_n = 0
    else:
        if depth.shape != dem.shape:
            raise ValueError(f"flood shape {depth.shape} != DEM {dem.shape}")
        wet_n = int(wet.sum()) if wet is not None else int((depth > 0).sum())
        blocked_n = int((depth >= WHEELCHAIR_BLOCK_M).sum())

    height_mean = float(bh.mean()) if bh.size else 0.0
    height_max = float(bh.max()) if bh.size else 0.0
    return {
        "dem_min_m": float(dem.min()),
        "dem_max_m": float(dem.max()),
        "dem_mean_m": float(dem.mean()),
        "relief_m": float(dem.max() - dem.min()),
        "building_count": int(cx.size),
        "height_mean_m": height_mean,
        "height_max_m": height_max,
        "footprint_coverage_pct": float(100.0 * float(inside.mean())),
        "flood_wet_cells": int(wet_n),
        "flood_blocked_cells": int(blocked_n),
        "wet_cells": int(wet_n),
        "blocked_cells": int(blocked_n),
        "meta": {"wet_cells": int(wet_n), "blocked_cells": int(blocked_n),
                 "max_depth_m": (float(depth.max()) if depth is not None
                                 else 0.0)},
        "gnss_shadow_mean_pct": float(_gnss_at_12m_pct(slice_dict)),
    }


# ---------------------------------------------------------------------------
# (4) Ground metrics
# ---------------------------------------------------------------------------

def _sheltered_pct(edges_used):
    """Length-weighted sheltered % if computable, else None."""
    if not edges_used:
        return None
    edges = list(edges_used)
    total = 0.0
    sheltered = 0.0
    for e in edges:
        if not isinstance(e, dict) or "sheltered" not in e:
            continue
        length = float(e.get("length_m", 1.0))
        total += length
        if e["sheltered"]:
            sheltered += length
    if total <= 0:
        return None
    return float(100.0 * sheltered / total)


def ground_metrics(route_result, edges_used=None) -> dict:
    """Ground-route summary at wheelchair cruise speed (screening grade).

    Args:
        route_result: routing.RouteResult, capture.json ground dict
            ({success, length_m, time_s, max_flood_m}), or a slice
            polyline [[x, y], ...].  Time is ``length / 1.1`` m/s when
            the result does not carry a finite time.
        edges_used: optional list of edge dicts with ``sheltered`` (+
            ``length_m``) to compute the sheltered share.  Without it,
            ``sheltered_pct`` is None (not computable).

    Returns length_m, time_s, max_flood_m, sheltered_pct (+ success if
    the result carries one).  Flood uses the wheelchair >= 0.10 m
    screening block; grades are exterior assembly / linkway (advisory).
    """
    if route_result is None:
        raise ValueError("route_result is None")
    success, length, time_s, max_flood = None, None, None, None
    if hasattr(route_result, "length_m") or hasattr(route_result, "success"):
        success = bool(getattr(route_result, "success", True))
        length = float(getattr(route_result, "length_m", 0.0) or 0.0)
        raw_t = getattr(route_result, "time_s", None)
        time_s = float(raw_t) if raw_t is not None and math.isfinite(float(raw_t)) else None
        raw_f = getattr(route_result, "max_flood_m", None)
        max_flood = float(raw_f) if raw_f is not None else None
    elif isinstance(route_result, dict):
        success = bool(route_result.get("success", True))
        length = float(route_result.get("length_m", 0.0) or 0.0)
        raw_t = route_result.get("time_s", route_result.get("time"))
        time_s = float(raw_t) if raw_t is not None else None
        raw_f = route_result.get("max_flood_m", route_result.get("max_flood"))
        max_flood = float(raw_f) if raw_f is not None else None
        if length <= 0 and "path" in route_result:
            pts = np.asarray(route_result["path"], dtype=np.float64)
            length = _path_length(pts)
    elif isinstance(route_result, (list, tuple)):
        pts = np.asarray(route_result, dtype=np.float64)
        length = _path_length(pts)
    else:
        raise TypeError(f"unsupported route_result type: {type(route_result)}")
    if length is None or not math.isfinite(float(length)):
        length = 0.0
    length = float(length)
    if time_s is None or not math.isfinite(float(time_s)):
        time_s = length / WHEELCHAIR_SPEED_MPS
    else:
        time_s = float(time_s)
    return {
        "length_m": float(length),
        "time_s": float(time_s),
        "max_flood_m": (float(max_flood) if max_flood is not None else None),
        "sheltered_pct": _sheltered_pct(edges_used),
        "success": success,
    }


# ---------------------------------------------------------------------------
# CLI: real slice (real_slice + routing) Gen0 blind route + frames
# ---------------------------------------------------------------------------

def _pick_swne_pair(nodes, edges):
    """SW->NE wheelchair-passable pair (mirrors scripts/run_real.py)."""
    adj: dict[int, list[int]] = {}
    for e in edges:
        if e["slope"] > 1.0 / 12.0 or e["width_m"] < 1.2:
            continue
        adj.setdefault(e["u"], []).append(e["v"])
        adj.setdefault(e["v"], []).append(e["u"])
    comps, seen = [], set()
    for n in nodes:
        s = n["id"]
        if s in seen or s not in adj:
            continue
        seen.add(s)
        stack, comp = [s], [s]
        while stack:
            cur = stack.pop()
            for nb in adj.get(cur, []):
                if nb not in seen:
                    seen.add(nb)
                    comp.append(nb)
                    stack.append(nb)
        comps.append(comp)
    by_id = {n["id"]: n for n in nodes}
    best = None
    for comp in comps:
        if len(comp) < 2:
            continue
        pool = [by_id[i] for i in comp]
        a = min(pool, key=lambda n: (n["x"] - 20.0) ** 2 + (n["y"] - 20.0) ** 2)
        b = min(pool, key=lambda n: (n["x"] - 180.0) ** 2 + (n["y"] - 180.0) ** 2)
        total = (math.hypot(a["x"] - 20.0, a["y"] - 20.0)
                 + math.hypot(b["x"] - 180.0, b["y"] - 180.0))
        key = (total, -len(comp))
        if best is None or key < best[0]:
            best = (key, int(a["id"]), int(b["id"]), comp)
    if best is None:
        raise ValueError("no wheelchair-passable component with >= 2 nodes")
    return best[1], best[2], best[3]


def _nearest_pad(pads, tx: float, ty: float) -> dict:
    return min(pads, key=lambda p: (p["x"] - tx) ** 2 + (p["y"] - ty) ** 2)


def main(argv=None) -> int:
    """Load the real slice and print Gen0 blind-route + frames summary."""
    import argparse

    ap = argparse.ArgumentParser(description="AETHER-SG metrics smoke on real slice")
    ap.add_argument("capture_dir", nargs="?", default=None,
                    help="Aleph capture directory (default: project capture)")
    ap.add_argument("--capture-json", default=None,
                    help="Optional viewer/capture/capture.json for cross-check")
    args = ap.parse_args(argv)
    here = Path(__file__).resolve()
    root = here.parents[2]
    if args.capture_dir:
        capture_dir = Path(args.capture_dir)
    else:
        capture_dir = root / "captures" / "aleph-20261119T022946Z-1upsr2v9"
    if not capture_dir.is_dir():
        print(f"capture dir missing: {capture_dir}")
        return 2

    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from aether.real_slice import build_real_slice
    from aether import evaluator, routing

    s = build_real_slice(capture_dir)
    dem = s["dem"]
    print(f"slice: {len(s['buildings'])} buildings, {len(s['nodes'])} nodes, "
          f"{len(s['edges'])} edges, dem {s['dem'].shape} "
          f"min {float(dem.min()):.1f} max {float(dem.max()):.1f} m")

    sdf_bundle = evaluator.build_sdf(s["buildings"])
    shadow_bundle = evaluator.gnss_shadow(s["buildings"])
    flood = evaluator.flood_tensor(dem, float(dem.min()) + 0.3)
    flood_depth = np.asarray(flood["depth"], dtype=np.float32)
    layers = {"sdf": routing.build_sdf(s),
              "gnss": routing.build_gnss_shadow(s),
              "wind": routing.build_wind(s)}

    start_pad = _nearest_pad(s["landing_pads"], 20.0, 20.0)
    goal_pad = _nearest_pad(s["landing_pads"], 180.0, 180.0)
    drone_start = (float(start_pad["x"]), float(start_pad["y"]),
                   float(start_pad["z"]))
    drone_goal = (float(goal_pad["x"]), float(goal_pad["y"]),
                  float(goal_pad["z"]))
    print(f"drone pads: SW {start_pad['id']} {drone_start} -> "
          f"NE {goal_pad['id']} {drone_goal}")

    blind = {"w_time": 1.0, "w_energy": 0.0, "w_shadow": 0.0,
             "w_wind": 0.0, "w_detour": 0.0}
    r = routing.drone_astar(s, drone_start, goal_xyz=drone_goal, weights=blind,
                            sdf=layers["sdf"], gnss_shadow=layers["gnss"],
                            wind=layers["wind"])
    if not r.success:
        print(f"Gen0 blind FAILED: {r.notes}")
        return 1
    m = flight_metrics(np.asarray(r.path, dtype=np.float64), s["buildings"],
                       sdf_bundle, shadow_bundle, dem)
    print(f"Gen0 blind (routing A*, time-only): len {m['length_m']:.1f} m, "
          f"time {m['time_s']:.1f} s, energy_proxy {m['energy_proxy']:.1f} "
          f"({m['est_energy_wh']:.2f} Wh), clearance "
          f"{m['min_clearance_m']:.1f}/{m['mean_clearance_m']:.1f} m, "
          f"shadow {m['shadow_exposure_pct']:.1f}%, score {m['total_score']:.1f}")
    print(f"  altitude {m['altitude_min_m']:.0f}-{m['altitude_max_m']:.0f} m "
          f"(mean {m['altitude_mean_m']:.1f}), climb {m['total_climb_m']:.1f} m, "
          f"collisions {m['collisions']}, wind {m['wind']}")
    print(f"  score breakdown: {m['score']} (total {m['total_score']:.1f})")
    print(f"  routing result: len {r.length_m:.1f} m, time {r.time_s:.1f} s, "
          f"energy {r.energy:.1f}, shadow {r.shadow_exposure:.1f} m, "
          f"min_sdf {r.min_sdf_m:.1f} m")
    print("  note: evaluator shadow is binary occlusion (often 0% above roofs); "
          "routing shadow is a float surrogate - see module docstring.")
    print("  note: est_energy_wh = energy_proxy*0.9/3600*1000 assumes 900 W "
          "mean draw; screening estimate only.")

    # Cross-check against the shipped Gen0 path, if present.
    cap_path = (Path(args.capture_json) if args.capture_json
                else root / "viewer" / "capture" / "capture.json")
    if cap_path.is_file():
        cap = load_capture_json(cap_path)
        gen0_v = (cap.get("dronePaths") or {}).get("Gen0")
        if gen0_v:
            gen0_s = viewer_drone_to_slice(gen0_v)
            m2 = flight_metrics(np.array(gen0_s), s["buildings"],
                                sdf_bundle, shadow_bundle, dem)
            print(f"capture.json Gen0 cross-check: len {m2['length_m']:.1f} m "
                  f"vs shipped {float(cap['metrics']['Gen0']['length_m']):.1f} m; "
                  f"time {m2['time_s']:.1f} s, clearance min "
                  f"{m2['min_clearance_m']:.2f} m vs shipped "
                  f"{float(cap['metrics']['Gen0']['min_sdf_m']):.2f} m")

    frames = los_swath_frames(np.asarray(r.path, dtype=np.float64),
                              s["buildings"], (drone_goal[0], drone_goal[1]),
                              dem, fov_deg=60.0)
    fs = summarize_frames(frames)
    print(f"LOS/swath frames (target {drone_goal[0]:.0f},{drone_goal[1]:.0f}): "
          f"{fs['n_frames']} frames, LOS clear {fs['los_clear_pct']:.1f}% "
          f"({fs['los_blocked_count']} blocked), mean min-clearance "
          f"{fs['los_min_clearance_mean_m']:.1f} m, final coverage "
          f"{fs['final_coverage_pct']:.1f}%, mean swath area "
          f"{fs['swath_area_mean_m2']:.0f} m2")
    print("  note: LOS to ground crosses rooftops (82% union footprint, blind "
          "altitude 12-24 m below mean roof 20 m), so blocked views are expected; "
          "the evolved 48 m route would score higher.")

    env = environment_metrics({**s, "gnss_shadow": layers["gnss"]},
                              flood_depth)
    print(f"environment: relief {env['relief_m']:.1f} m, buildings "
          f"{env['building_count']}, height mean {env['height_mean_m']:.1f} / "
          f"max {env['height_max_m']:.1f} m, footprint {env['footprint_coverage_pct']:.1f}%, "
          f"flood wet {env['flood_wet_cells']} / blocked {env['flood_blocked_cells']} cells, "
          f"GNSS @12 m {env['gnss_shadow_mean_pct']:.1f}%")

    g_start, g_goal, _comp = _pick_swne_pair(s["nodes"], s["edges"])
    g = routing.ground_astar(s, g_start, g_goal, flood_tensor=flood_depth)
    if g.success:
        gm = ground_metrics(g)
        print(f"ground SW->NE: len {gm['length_m']:.1f} m, time "
              f"{gm['time_s']:.1f} s @1.1 m/s, max_flood "
              f"{gm['max_flood_m']:.2f} m, sheltered {gm['sheltered_pct']}")
    else:
        print(f"ground: FAILED: {g.notes}")
        return 1
    print("metrics.py OK - advisory, screening grade, real capture only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


