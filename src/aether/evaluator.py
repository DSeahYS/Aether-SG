"""AETHER-SG fast tensor evaluator: voxel SDF + GNSS shadow + flood screening.

All physics here is SCREENING-GRADE, not certified.  Per Research-AETHER-SG
(section 6): SDF lookup is grade-A concept; flood is grade-B screening
(bathtub+connectivity, NOT 2D hydrodynamics); GNSS shadow is a
building-occlusion proxy, NOT ray-traced ephemeris.  Never advertise
"guaranteed" or "regulation-compliant" behaviour.

Drone envelope is 0-60 m AGL (Research section 1 correction: 200 ft AMSL
default ceiling ~61 m, no CAAS corridor network claimed).

Primary API (for routing / evolve agents)
-----------------------------------------
build_sdf(buildings, extent_m=200.0, voxel_m=2.0, z_max_m=60.0)
    -> dict with "sdf" float32 (nz, ny, nx), "voxel_m", "extent_m",
    "z_max_m", "meta".  Negative = inside a building box, positive =
    free space; also clipped by the ground plane z >= dem surface datum 0.
sdf_at(sdf_bundle, x, y, z) -> float
    Fast nearest-voxel SDF lookup (numpy indexing, O(1)).
sdf_sample(sdf_bundle, points) -> ndarray
    Vectorised lookup for an (N, 3) points array.
gnss_shadow(buildings, extent_m=200.0, voxel_m=2.0, z_max_m=60.0)
    -> uint8 (nz, ny, nx) 0/1 shadow mask (occluded-below-roofline proxy).
    shadow_at(bundle, x, y, z) -> int  scalar lookup.
flood_tensor(dem, water_level_m, dem_res_m=1.0, source="south")
    -> dict {"depth": (H, W) float32, "wet": bool, "meta": {...}}.
    Bathtub depth = max(0, water_level - dem) restricted to cells
    connected to the drain-channel source.  Screening only.
score_path(points, sdf_bundle, shadow_bundle, dem, dem_res_m=1.0,
           flood=None, speed_mps=8.0, energy_j_per_m=10.0, ...)
    -> float  Single scalar to MAXIMISE (CD contract, Research section 4):
    score = -(time_s + energy) - 1000 * collisions
            - flood_penalty - shadow_penalty
    Hard rejects return -inf: any SDF collision sample, or ground-path
    flood depth > EVAC_REJECT_M (1.0 m).  Wheelchair block threshold
    WHEELCHAIR_BLOCK_M = 0.10 m (Research section 1 [INFERENCE]) accrues
    penalty below the hard-reject depth.
    points: (N, 3) array (x, y, z AGL) OR (N, 2) ground path (z from DEM).
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

WHEELCHAIR_BLOCK_M = 0.10   # [INFERENCE] Research sec.1: block >= 0.10 m
EVAC_REJECT_M = 1.0         # evacuate threshold (Research sec.1 / PUB 1.0 m)
PREPARE_M = 0.3             # prepare threshold (observed flash-flood max)


def _grid(extent_m: float, voxel_m: float, z_max_m: float):
    nx = int(round(extent_m / voxel_m))
    ny = int(round(extent_m / voxel_m))
    nz = int(round(z_max_m / voxel_m))
    xs = (np.arange(nx) + 0.5) * voxel_m
    ys = (np.arange(ny) + 0.5) * voxel_m
    zs = (np.arange(nz) + 0.5) * voxel_m
    return nx, ny, nz, xs, ys, zs


def build_sdf(buildings: list[dict], extent_m: float = 200.0,
              voxel_m: float = 2.0, z_max_m: float = 60.0) -> dict:
    """Voxel SDF over axis-aligned building boxes (sparse-friendly).

    Dense float32 at 2 m / 0-60 m is (30, 100, 100) = 300k voxels, small
    enough for fast dense numpy lookups while remaining exportable as a
    sparse structure if a future island-scale grid needs it.
    """
    nx, ny, nz, xs, ys, zs = _grid(extent_m, voxel_m, z_max_m)
    # Start from ground-plane distance (z above datum).
    Z = zs[:, None, None]
    sdf = np.broadcast_to(Z, (nz, ny, nx)).astype(np.float32).copy()
    # min() over box SDFs (axis-aligned box SDF, negative inside).
    X, Y = np.meshgrid(xs, ys)  # each (ny, nx)
    for b in buildings:
        hx, hy, hz = b["w"] / 2.0, b["d"] / 2.0, b["height_m"]
        qx = np.abs(X[None] - b["cx"]) - hx          # (1, ny, nx)
        qy = np.abs(Y[None] - b["cy"]) - hy          # broadcast to z too
        qz = Z - hz                                  # (nz, 1, 1)
        # box SDF: outside = norm(max(q,0)), inside = min(max(q), 0)
        ox = np.maximum(qx, 0.0); oy = np.maximum(qy, 0.0); oz = np.maximum(qz, 0.0)
        outside = np.sqrt(ox**2 + oy**2 + oz**2)
        inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
        box = (outside + inside).astype(np.float32)
        np.minimum(sdf, box, out=sdf)
    return {"sdf": sdf, "voxel_m": float(voxel_m), "extent_m": float(extent_m),
            "z_max_m": float(z_max_m),
            "meta": {"nx": nx, "ny": ny, "nz": nz, "n_voxels": int(sdf.size),
                     "n_buildings": len(buildings),
                     "label": "SIMULATED screening SDF (box approx, not mesh)"}}


def sdf_sample(bundle: dict, points: np.ndarray) -> np.ndarray:
    """Vectorised nearest-voxel SDF lookup. points: (N, 3) x,y,z AGL."""
    sdf, v = bundle["sdf"], bundle["voxel_m"]
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    ix = np.clip((pts[:, 0] / v).astype(np.int64), 0, sdf.shape[2] - 1)
    iy = np.clip((pts[:, 1] / v).astype(np.int64), 0, sdf.shape[1] - 1)
    iz = np.clip((pts[:, 2] / v).astype(np.int64), 0, sdf.shape[0] - 1)
    return sdf[iz, iy, ix]


def sdf_at(bundle: dict, x: float, y: float, z: float) -> float:
    """Scalar SDF lookup (negative = inside building)."""
    return float(sdf_sample(bundle, np.array([[x, y, z]]))[0])


def gnss_shadow(buildings: list[dict], extent_m: float = 200.0,
                voxel_m: float = 2.0, z_max_m: float = 60.0) -> dict:
    """Building-occlusion GNSS shadow proxy: 1 where z < roofline inside
    a footprint.  NOT ray-traced ephemeris (Research sec.6 grade A/B)."""
    nx, ny, nz, xs, ys, zs = _grid(extent_m, voxel_m, z_max_m)
    shadow = np.zeros((nz, ny, nx), dtype=np.uint8)
    for b in buildings:
        ix0 = int(max(0, np.floor((b["cx"] - b["w"] / 2) / voxel_m)))
        ix1 = int(min(nx, np.ceil((b["cx"] + b["w"] / 2) / voxel_m)))
        iy0 = int(max(0, np.floor((b["cy"] - b["d"] / 2) / voxel_m)))
        iy1 = int(min(ny, np.ceil((b["cy"] + b["d"] / 2) / voxel_m)))
        iz1 = int(min(nz, np.ceil(b["height_m"] / voxel_m)))
        shadow[:iz1, iy0:iy1, ix0:ix1] = 1
    return {"shadow": shadow, "voxel_m": float(voxel_m),
            "extent_m": float(extent_m), "z_max_m": float(z_max_m),
            "meta": {"nx": nx, "ny": ny, "nz": nz,
                     "label": "SIMULATED occlusion proxy (not ephemeris)"}}


def shadow_sample(bundle: dict, points: np.ndarray) -> np.ndarray:
    sh, v = bundle["shadow"], bundle["voxel_m"]
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    ix = np.clip((pts[:, 0] / v).astype(np.int64), 0, sh.shape[2] - 1)
    iy = np.clip((pts[:, 1] / v).astype(np.int64), 0, sh.shape[1] - 1)
    iz = np.clip((pts[:, 2] / v).astype(np.int64), 0, sh.shape[0] - 1)
    return sh[iz, iy, ix].astype(np.float32)


def shadow_at(bundle: dict, x: float, y: float, z: float) -> int:
    return int(shadow_sample(bundle, np.array([[x, y, z]]))[0])


def flood_tensor(dem: np.ndarray, water_level_m: float,
                 dem_res_m: float = 1.0, source: str = "south") -> dict:
    """Bathtub + connectivity screening flood (NOT hydrodynamics).

    Depth = max(0, water_level - dem) but only where connected to the
    south drain channel (guards against over-estimating behind linkways,
    Research sec.6).  water_level_m is an absolute datum in DEM metres.
    """
    dem = np.asarray(dem, dtype=np.float32)
    depth_raw = np.clip(water_level_m - dem, 0.0, None).astype(np.float32)
    candidate = depth_raw > 0
    H, W = dem.shape
    wet = np.zeros((H, W), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    if source == "south":
        # Seed from the drain-channel band along the south edge (trough
        # cells, not only the border row, so a channel a few cells inland
        # still seeds correctly).
        seed_cells = [(iy, x) for iy in range(min(12, H)) for x in range(W)
                      if candidate[iy, x]]
    else:
        seed_cells = [(H - 1, x) for x in range(W) if candidate[H - 1, x]]
    for iy, ix in seed_cells:
        if not wet[iy, ix]:
            wet[iy, ix] = True
            q.append((iy, ix))
    while q:
        iy, ix = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cy, cx = iy + dy, ix + dx
            if 0 <= cy < H and 0 <= cx < W and candidate[cy, cx] and not wet[cy, cx]:
                wet[cy, cx] = True
                q.append((cy, cx))
    depth = np.where(wet, depth_raw, 0.0).astype(np.float32)
    return {"depth": depth, "wet": wet,
            "meta": {"water_level_m": float(water_level_m),
                     "wet_cells": int(wet.sum()),
                     "max_depth_m": float(depth.max()) if depth.size else 0.0,
                     "label": "SIMULATED bathtub+connectivity screening"}}


def dem_at(dem: np.ndarray, dem_res_m: float, x: float, y: float) -> float:
    H, W = dem.shape
    ix = int(np.clip(round(x / dem_res_m), 0, W - 1))
    iy = int(np.clip(round(y / dem_res_m), 0, H - 1))
    return float(dem[iy, ix])


def score_path(points: np.ndarray, sdf_bundle: dict, shadow_bundle: dict,
               dem: np.ndarray, dem_res_m: float = 1.0,
               flood: dict | None = None, speed_mps: float = 8.0,
               energy_j_per_m: float = 10.0,
               flood_weight: float = 100.0, shadow_weight: float = 20.0,
               ground: bool = False) -> float:
    """Single scalar score to MAXIMISE (CD evaluator contract).

    score = -(time_s + energy) - 1000 * collisions
            - flood_weight * mean_flood_excess - shadow_weight * mean_shadow

    Hard rejects (return -inf, unscored by CD per Research sec.4):
      * any sample with SDF < 0  (inside a building box), or
      * any ground sample with flood depth > EVAC_REJECT_M (1.0 m).
    Ground paths (points shape (N, 2)) use DEM height for z and accrue
    flood penalty for depth above WHEELCHAIR_BLOCK_M (0.10 m).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return float("-inf")
    if pts.shape[1] == 2:  # ground path: lift z from DEM
        xy = pts
        z = np.array([dem_at(dem, dem_res_m, p[0], p[1]) for p in xy])
        pts3 = np.column_stack([xy, z])
        ground = True
    else:
        pts3 = pts
    # Densify so no segment skips over a box between samples.
    seg = np.hypot(np.diff(pts3[:, 0]), np.diff(pts3[:, 1]))
    length = float(seg.sum())
    if length <= 0:
        return float("-inf")
    n_dense = int(np.clip(int(length / 1.0), len(pts3), 4000))
    t = np.linspace(0, 1, n_dense)
    dense = np.column_stack([np.interp(t, np.linspace(0, 1, len(pts3)), pts3[:, k])
                             for k in range(3)])
    # Altitude reject: solver envelope is 0-60 m (Research sec.1).
    if bool((dense[:, 2] < 0).any()) or bool((dense[:, 2] > 60).any()):
        return float("-inf")
    sdf_vals = sdf_sample(sdf_bundle, dense)
    collisions = int((sdf_vals < 0).sum())
    if collisions > 0:
        return float("-inf")  # hard reject
    time_s = length / max(speed_mps, 1e-6)
    energy = energy_j_per_m * length
    shadow_mean = float(shadow_sample(shadow_bundle, dense).mean())
    flood_excess = 0.0
    if flood is not None and ground:
        dep = flood["depth"]
        vals = []
        for p in dense:
            ix = int(np.clip(round(p[0] / dem_res_m), 0, dep.shape[1] - 1))
            iy = int(np.clip(round(p[1] / dem_res_m), 0, dep.shape[0] - 1))
            vals.append(float(dep[iy, ix]))
        vals = np.asarray(vals)
        if bool((vals > EVAC_REJECT_M).any()):
            return float("-inf")  # hard reject: evacuation threshold hit
        excess = np.clip(vals - WHEELCHAIR_BLOCK_M, 0.0, None)
        # Bonus pull toward sheltered/ramped geometry is owned by the
        # routing agent; here we only charge depth above the block level.
        flood_excess = float(excess.mean()) if excess.size else 0.0
    return (-(time_s + energy)
            - 1000.0 * collisions
            - flood_weight * flood_excess
            - shadow_weight * shadow_mean)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aether.slice_data import build_slice

    sl = build_slice(seed=7)
    t0 = time.perf_counter()
    sdf_b = build_sdf(sl["buildings"])
    t1 = time.perf_counter()
    sh_b = gnss_shadow(sl["buildings"])
    print(f"SDF build: {t1 - t0:.3f}s shape={sdf_b['sdf'].shape} "
          f"min={sdf_b['sdf'].min():.1f} max={sdf_b['sdf'].max():.1f}")

    # SDF lookup timing benchmark: 10k vectorised + 1k scalar lookups.
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 200, 10000),
                           rng.uniform(0, 200, 10000),
                           rng.uniform(0, 60, 10000)])
    t0 = time.perf_counter()
    vals = sdf_sample(sdf_b, pts)
    t1 = time.perf_counter()
    print(f"Vectorised SDF lookup (10k pts): {(t1 - t0) * 1e3:.2f} ms "
          f"-> {(t1 - t0) / 10:.4f} ms per 1k pts; "
          f"inside-box samples={int((vals < 0).sum())}")
    t0 = time.perf_counter()
    for p in pts[:1000]:
        sdf_at(sdf_b, float(p[0]), float(p[1]), float(p[2]))
    t1 = time.perf_counter()
    print(f"Scalar SDF lookup (1k): {(t1 - t0) * 1e3:.2f} ms")

    fl = flood_tensor(sl["dem"], water_level_m=float(sl["dem"].min()) + 0.3)
    print("Flood screening meta:", fl["meta"])
    clear_path = np.array([[10.0, 100.0, 30.0], [100.0, 100.0, 30.0],
                           [190.0, 100.0, 30.0]])
    print("Clear-path score:", score_path(clear_path, sdf_b, sh_b,
                                          sl["dem"], flood=fl))
    print("SDF/SDF-allocation note: research allows <10ms per candidate "
          "only as precomputed lookup (Research sec.6).")
