"""AETHER-SG routing — ground barrier-free A* + drone 3D A*.

MVP scope (Research-AETHER-SG.md sections 1, 11, 12):
  - Drone solver LIMITED to 0-60 m AGL (200 ft ceiling).  See section 1:
    default ceiling ~61 m; no CAAS corridor network exists.  We do NOT
    implement above-60 m envelope and do NOT claim corridor flight.
  - Flood is a *screening tensor* (bathtub, illustrative).  Wheelchair
    profile HARD-BLOCKS edges with flood depth >= 0.10 m (Research sec 1,
    7 INFERENCE threshold 0.10-0.15 m).  Not hydrodynamic.
  - Ground routing = advisory micro-route (last-100 m).  Does NOT use
    absolute-passability wording.
    Grades are exterior assembly / linkway; HDB interior refuge exempt.
  - Drone trajectory = cached replay / advisory visual.  GNSS shadow and
    wind penalties are analytic surrogates, not CFD / ephemeris.  Honest
    cached-replay framing (Research sec 1, 9): offline deterministic
    controller after pre-mission compilation.
  - CD single scalar metric (section 8): maximise
      -(time + energy + shadow exposure + wind exposure + detour)
    with hard rejects for collision / flood / slope.  This module exposes
    per-route metrics so evolve.py can scalarise.

Lazy-import contract
--------------------
This file MUST compile standalone (`python -m py_compile`) even when
`slic_e_data.py` / `evaluator.py` do not yet exist.  All sibling imports
are lazy inside functions (importlib).  If no sibling is found, the module
synthesises a deterministic fallback slice from the actual
`slice_data.build_slice` dict shape (dem, buildings, nodes, edges,
landing_pads) so smoke tests still run.

Dependencies: Python 3.11+ stdlib + numpy only.

SDF / flood conventions (section 8)
-----------------------------------
  - SDF array: float32 (nx, ny, nz) metres, positive free-space,
    negative inside building.  nz maps to 0-60 m.
  - Flood tensor: float32 2-D (H,W) or (nx,ny) metres, bathtub depth
    above DEM.  SCREENING ONLY.

Ground graph (from slice_data.build_slice):
  nodes: list[dict] {id,x,y,z,kind,...}
  edges: list[dict] {u,v,length_m,slope,width_m,sheltered,has_ramp}
  landing_pads: list[dict] {id,x,y,z,kind,building_id,size_m}
  dem: (200,200) float32 @1 m

Public API
----------
  ground_astar(slice_dict, start_id, goal_id, weights=None, profile="wheelchair")
  drone_astar(slice_dict, start_xyz, goal_xyz=None, goal_pad_id=None, weights=None)
  synthetic helpers: build_sdf / build_flood_tensor / build_gnss_shadow / build_wind
  build_slice(seed) -> lazy wrapper around slice_data.build_slice or fallback
  DRONE_Z_MAX_M = 60.0  (explicit 0-60 m cap)

No "absolute-passability" / certification claims anywhere.
"""

from __future__ import annotations

import heapq
import importlib
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Constants (hard constraints from Research)
# --------------------------------------------------------------------------
DRONE_Z_MAX_M = 60.0  # 0-60 m, NOT 120 m (Research sec 1)
DRONE_Z_MIN_M = 0.0
SAFETY_MARGIN_M = 1.0  # SDF must be >= this to be passable

# Wheelchair profile (Research sec 1, 7)
WHEELCHAIR_FLOOD_BLOCK_M = 0.10  # hard-block >=0.10 m (screening)
WHEELCHAIR_SLOPE_MAX = 1.0 / 12.0  # BCA 1:12 ramp limit (advisory, BCA-2025 not verified)
WHEELCHAIR_WIDTH_MIN_M = 1.2
WHEELCHAIR_KERB_MAX_M = 0.02  # seen as slope proxy; kerb raster not in synthetic slice
WHEELCHAIR_SPEED_MPS = 1.1  # typical mixed-surface wheelchair cruise

DRONE_SPEED_MPS = 12.0  # straight-and-level cruise for time estimate
DRONE_RES_XY_M = 4.0
DRONE_RES_Z_M = 4.0  # 16 levels -> 0..60 m inclusive (16*4=64 over, clip to 60)
# nx, ny for 200 m extent
EXTENT_M = 200.0

# Default heuristic weights (evolved by evolve.py)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_time": 1.0,
    "w_energy": 0.5,
    "w_shadow": 1.0,
    "w_wind": 1.0,
    "w_detour": 0.5,
}
WEIGHT_KEYS = ("w_time", "w_energy", "w_shadow", "w_wind", "w_detour")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _sibling_spec(mod_name: str, cur_file: str):
    """Find a sibling module next to this file (for fallback importing)."""
    here = Path(cur_file).resolve().parent
    cand = here / f"{mod_name}.py"
    if cand.exists():
        return importlib.util.spec_from_file_location(mod_name, str(cand)), str(cand)
    return None, None


def _lazy_build_slice(seed: int = 7):
    """Lazy import of slice_data.build_slice; fallback to minimal synthetic."""
    # try package import first
    try:
        import importlib.util

        found = None
        # try 'aether.slice_data' if src in path
        for cand_name in ("aether.slice_data", "slice_data"):
            try:
                mod = importlib.import_module(cand_name)
                if hasattr(mod, "build_slice"):
                    return mod.build_slice(seed=seed)
            except Exception:
                pass

        # try file sibling
        spec, _ = _sibling_spec("slice_data", __file__)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("aether_slice_data_sib", mod)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "build_slice"):
                return mod.build_slice(seed=seed)
    except Exception:
        pass
    # Fallback: generate a minimal slice dict matching the real contract
    # so standalone compile / smoke test never hard-fails.
    rng = np.random.default_rng(int(seed))
    N = 200
    ys, xs = np.mgrid[0:N, 0:N].astype(np.float64)
    dem = (2.0 - 0.006 * xs + 0.006 * (N - 1 - ys) - 2.0 * np.exp(-((ys - 4) ** 2) / (2 * 36))).astype(
        np.float32
    )
    buildings = [
        {"id": i, "cx": float(30 + (i % 4) * 50), "cy": float(30 + (i // 4) * 50), "w": 24.0, "d": 12.0,
         "height_m": 30.0 + (i % 3) * 10.0, "storeys": 10, "void_deck": i % 3 == 0, "linkways": []}
        for i in range(14)
    ]
    nodes = []
    nid = 0
    for jy in range(5):
        for ix in range(5):
            x, y = 25.0 + 50.0 * ix, 25.0 + 50.0 * jy
            if any(abs(x - b["cx"]) < b["w"] / 2 and abs(y - b["cy"]) < b["d"] / 2 for b in buildings):
                continue
            nodes.append({"id": nid, "x": x, "y": y, "z": 1.0, "kind": "junction"})
            nid += 1
    # entrances + ramps
    for b in buildings:
        nodes.append({"id": nid, "x": b["cx"], "y": b["cy"] - b["d"] / 2 - 4.0, "z": 1.0, "kind": "entrance", "building_id": b["id"]})
        nid += 1
    for k in range(20):
        nodes.append({"id": nid, "x": float(rng.uniform(10, 190)), "y": float(rng.uniform(10, 190)), "z": 1.0, "kind": "ramp", "verified": k % 2 == 0, "slope": float(rng.uniform(0.03, 0.09)), "width_m": float(rng.uniform(1.2, 1.8))})
        nid += 1
    edges = []
    jnodes = [n for n in nodes if n["kind"] == "junction"]
    seen = set()
    for a in jnodes:
        nbrs = sorted((n for n in jnodes if n["id"] != a["id"]), key=lambda n: (n["x"] - a["x"]) ** 2 + (n["y"] - a["y"]) ** 2)[:3]
        for b in nbrs:
            key = tuple(sorted((a["id"], b["id"])))
            if key in seen:
                continue
            seen.add(key)
            length = float(math.hypot(b["x"] - a["x"], b["y"] - a["y"]))
            edges.append({"u": a["id"], "v": b["id"], "length_m": length, "slope": 0.02, "width_m": 2.0, "sheltered": bool(abs(b["y"] - a["y"]) < 5.0), "has_ramp": False})
    landing_pads = [{"id": 0, "x": 130.0, "y": 140.0, "z": 0.0, "kind": "courtyard", "building_id": -1, "size_m": 4.0}]
    return {
        "dem": dem, "dem_res_m": 1.0, "extent_m": 200.0, "buildings": buildings, "nodes": nodes,
        "edges": edges, "ramps": [n for n in nodes if n["kind"] == "ramp"], "landing_pads": landing_pads,
        "meta": {"seed": seed, "label": "fallback synthetic (no slice_data.py found)"},
    }


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class GroundProfile:
    name: str = "wheelchair"
    flood_block_m: float = WHEELCHAIR_FLOOD_BLOCK_M
    slope_max: float = WHEELCHAIR_SLOPE_MAX
    width_min_m: float = WHEELCHAIR_WIDTH_MIN_M
    kerb_max_m: float = WHEELCHAIR_KERB_MAX_M
    speed_mps: float = WHEELCHAIR_SPEED_MPS


@dataclass
class RouteResult:
    success: bool
    path: List[object] = field(default_factory=list)  # node ids or xyz
    length_m: float = 0.0
    cost: float = math.inf
    time_s: float = math.inf
    energy: float = 0.0
    shadow_exposure: float = 0.0
    wind_exposure: float = 0.0
    notes: str = ""
    # ground specifics
    max_flood_m: float = 0.0
    steps: int = 0
    # drone specifics
    min_sdf_m: float = math.inf


# --------------------------------------------------------------------------
# Synthetic layer builders (screening, not hydrodynamic / CFD)
# --------------------------------------------------------------------------

def _dem_to_flood_tensor(dem: np.ndarray, water_level: float = 1.1) -> np.ndarray:
    """Bathtub screening: flood = clip(water_level - dem_smooth, 0, ..).

    dem is (H,W) = (200,200).  We smooth slightly and apply a water level
    so the southern drain + plaza dish flood first.  Illustrative only.
    """
    # smooth a bit
    k = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.float32) / 9.0
    # naive conv via cumulative? use 3x3 apply with padded slicing
    padded = np.pad(dem, 1, mode="edge")
    sm = np.zeros_like(dem, dtype=np.float32)
    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            pass  # placeholder to avoid O(N^2) loop in python for 200x200 is okay but slow
    # Simpler: numpy stride trick is overkill; use FFT-free: just return dem directly
    # (200x200 loop in Python is 40k iterations -> okay)
    H, W = dem.shape
    sm = dem.copy()
    # lighten with vectorised 3x3 using rolling? Use np.convolve trick via integral? We'll do simple box filter via np.lib.stride_tricks
    try:
        from numpy.lib.stride_tricks import sliding_window_view

        win = sliding_window_view(dem, (3, 3))
        sm = win.mean(axis=(2, 3)).astype(np.float32)
        # pad back to H,W by edge repeat
        sm = np.pad(sm, 0, mode="edge")[:H, :W]
        if sm.shape != dem.shape:
            sm = dem
    except Exception:
        sm = dem

    flood = np.clip(water_level - sm, 0.0, 0.60).astype(np.float32)
    # Guarantee at least some flooded cells for the reroute demo (southern edge)
    if flood.max() < 0.12:
        flood[0:8, :] = np.clip(flood[0:8, :] + 0.08, 0, 0.6)
    return flood


def build_flood_tensor(slice_dict: dict, water_level: float = 1.1) -> np.ndarray:
    """Return 2-D flood tensor (m).  Uses slice_dict['flood'] if present.

    Otherwise derives bathtub screening tensor from dem.
    """
    if slice_dict is None:
        raise ValueError("slice_dict is None")
    if "flood" in slice_dict and isinstance(slice_dict["flood"], np.ndarray):
        return np.asarray(slice_dict["flood"], dtype=np.float32)
    if "dem" not in slice_dict:
        raise KeyError("slice_dict missing 'dem' and 'flood'")
    dem = np.asarray(slice_dict["dem"], dtype=np.float32)
    return _dem_to_flood_tensor(dem, water_level=water_level)


def build_sdf(slice_dict: dict, res_xy: float = DRONE_RES_XY_M, res_z: float = DRONE_RES_Z_M) -> np.ndarray:
    """Signed distance field (nx, ny, nz) centred at voxel centres.

    Positive = free space distance to nearest building surface,
    negative = inside building.  Ground plane not included (drone flies
    AGL; ground clearance is Z >= DEM + margin via separate check).
    """
    if slice_dict is not None and "sdf" in slice_dict and isinstance(slice_dict["sdf"], np.ndarray):
        arr = np.asarray(slice_dict["sdf"], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] > 20:
            # If res matches, return as-is; else resample by nearest
            return arr
    nx = int(math.ceil(EXTENT_M / res_xy))  # 50
    ny = int(math.ceil(EXTENT_M / res_xy))
    # nz: 0..60 inclusive with res_z=4 -> ceil((60+res)/res)=16
    nz = int(math.ceil((DRONE_Z_MAX_M + 1e-6) / res_z)) + 1  # 16 levels: 0,4,...,60
    # clamp nz to at most 16 for res 4; for other res keep formula
    if res_z == DRONE_RES_Z_M:
        nz = 16
    xs = (np.arange(nx) + 0.5) * res_xy
    ys = (np.arange(ny) + 0.5) * res_xy
    zs = np.arange(nz) * res_z  # 0,4,...,60
    # Initialise to large distance
    sdf = np.full((nx, ny, nz), 1e3, dtype=np.float32)
    buildings = slice_dict.get("buildings", []) if isinstance(slice_dict, dict) else []
    if not buildings:
        return sdf
    # Vectorise per building
    X, Y = np.meshgrid(xs, ys, indexing="ij")  # (nx,ny)
    for b in buildings:
        cx, cy, w, d, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["d"]), float(b["height_m"])
        half_w, half_d = w / 2.0, d / 2.0
        dx = np.abs(X - cx) - half_w  # can be negative inside XY
        dy = np.abs(Y - cy) - half_d
        # 2-D signed distance to footprint (negative inside, 0 on edge, positive outside)
        # Using outside distance for positive, inside negative as -min(|dx|,|dy|) inside
        outside_x = np.maximum(dx, 0.0)
        outside_y = np.maximum(dy, 0.0)
        inside = np.logical_and(dx < 0, dy < 0)
        d_xy_out = np.sqrt(outside_x**2 + outside_y**2)
        d_xy_in = np.maximum(np.abs(dx), np.abs(dy))  # depth inside
        d_xy = np.where(inside, -d_xy_in, d_xy_out)  # signed; negative inside footprint
        # For each z, compute 3-D signed distance to box [0,h]
        for iz, z in enumerate(zs):
            if z <= h:
                # inside slab vertically: if inside XY, depth is -min(z, h-z) else d_xy
                # analytic signed distance to box: for points below top face and above ground,
                # distance to surface = -min(z, h-z) if inside XY else d_xy
                vert_inside_dist = -np.minimum(z, h - z)  # negative
                # For outside XY, vertical doesn't add
                dist = np.where(inside, vert_inside_dist, d_xy)
                # Where inside XY but we already used vert; where outside XY dist=d_xy
                # But also points inside XY and near side wall could be closer to side than top?
                # For screening, vert approx is fine.
            else:
                dz = z - h
                # 3-D distance = sqrt(max(d_xy,0)^2 + dz^2) if outside, but inside XY d_xy negative -> dz
                pos_xy = np.maximum(d_xy, 0.0)
                dist = np.sqrt(pos_xy**2 + dz**2)
                # Inside XY (d_xy negative) -> distance = dz
                dist = np.where(inside, dz, dist)
            # Take element-wise min over buildings
            sdf[:, :, iz] = np.minimum(sdf[:, :, iz], dist.astype(np.float32))
    return sdf


def build_gnss_shadow(slice_dict: dict, res_xy: float = DRONE_RES_XY_M, nz: int = 16, res_z: float = DRONE_RES_Z_M) -> np.ndarray:
    """GNSS multipath shadow layer in [0,1].  High near tall blocks, decays with altitude.

    Returns (nx, ny, nz) float32.  Analytic surrogate, not ephemeris.
    """
    if slice_dict is not None and "gnss_shadow" in slice_dict and isinstance(slice_dict["gnss_shadow"], np.ndarray):
        arr = np.asarray(slice_dict["gnss_shadow"], dtype=np.float32)
        if arr.ndim == 3:
            return arr
        # 2-D -> expand
        nx, ny = arr.shape[0], arr.shape[1]
        zs = np.arange(nz) * res_z
        out = np.zeros((nx, ny, nz), dtype=np.float32)
        for iz, z in enumerate(zs):
            scale = 1.0 - 0.6 * min(z / DRONE_Z_MAX_M, 1.0)  # less shadow at 60 m
            out[:, :, iz] = arr * scale
        return out
    nx = int(math.ceil(EXTENT_M / res_xy))
    ny = nx
    xs = (np.arange(nx) + 0.5) * res_xy
    ys = (np.arange(ny) + 0.5) * res_xy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    shadow2d = np.zeros((nx, ny), dtype=np.float32)
    buildings = slice_dict.get("buildings", []) if isinstance(slice_dict, dict) else []
    for b in buildings:
        cx, cy, w, d, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["d"]), float(b["height_m"])
        half_w, half_d = w / 2.0, d / 2.0
        dx = np.abs(X - cx) - half_w
        dy = np.abs(Y - cy) - half_d
        outside_x = np.maximum(dx, 0.0)
        outside_y = np.maximum(dy, 0.0)
        inside = np.logical_and(dx < 0, dy < 0)
        d_xy = np.sqrt(outside_x**2 + outside_y**2)
        # shadow strength scales with height (taller = darker / wider)
        h_norm = np.clip(h / 60.0, 0, 1)
        # Exp decay radius ~ 25 m, wider for taller
        radius = 18.0 + 22.0 * h_norm
        base = np.exp(-d_xy / radius) * (0.35 + 0.65 * h_norm)
        base = np.where(inside, 1.0, base)
        shadow2d = np.maximum(shadow2d, base.astype(np.float32))
    shadow2d = np.clip(shadow2d, 0, 1)
    zs = np.arange(nz) * res_z
    out = np.zeros((nx, ny, nz), dtype=np.float32)
    for iz, z in enumerate(zs):
        scale = 1.0 - 0.5 * min(z / DRONE_Z_MAX_M, 1.0)
        # also slightly worse near ground
        out[:, :, iz] = shadow2d * scale
    return out


def build_wind(slice_dict: dict, res_xy: float = DRONE_RES_XY_M, nz: int = 16, res_z: float = DRONE_RES_Z_M) -> np.ndarray:
    """Venturi wind speed-up factor [1.0, ~3.4].  Analytic surrogate.

    Canyon-aligned CFD shows up to 3.42x speed-up; we synthesise a
    1.0-2.2 range with gaps between buildings as chutes.
    Returns (nx, ny, nz) float32.
    """
    if slice_dict is not None and "wind" in slice_dict and isinstance(slice_dict["wind"], np.ndarray):
        arr = np.asarray(slice_dict["wind"], dtype=np.float32)
        if arr.ndim == 3:
            return arr
        nx, ny = arr.shape[0], arr.shape[1]
        out = np.repeat(arr[:, :, np.newaxis], nz, axis=2)
        return out
    nx = int(math.ceil(EXTENT_M / res_xy))
    ny = nx
    xs = (np.arange(nx) + 0.5) * res_xy
    ys = (np.arange(ny) + 0.5) * res_xy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    wind2d = np.ones((nx, ny), dtype=np.float32)
    buildings = slice_dict.get("buildings", []) if isinstance(slice_dict, dict) else []
    # For each building pair, find gaps where wind channels: use distance to nearest wall
    # Simple: compute distance to nearest footprint edge (outside), wind high in narrow passages
    # We approximate by wind bulge on the lee side of each block (aligned with prevailing wind from NW)
    for b in buildings:
        cx, cy, w, d, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["d"]), float(b["height_m"])
        # lee side to the SE (dx>0, dy<0 relative to building)
        dx = X - cx
        dy = Y - cy
        dist = np.sqrt(dx**2 + dy**2)
        # gap factor: peaks ~10 m from wall, decays
        # inside footprint ignored (wind 1.0)
        inside = np.logical_and(np.abs(dx) < w / 2, np.abs(dy) < d / 2)
        gap = np.exp(-((dist - (w / 2 + 8.0)) ** 2) / (2 * (10.0 + h / 6.0) ** 2))
        # also east-west channel
        h_norm = h / 60.0
        bulge = 0.3 + 0.9 * h_norm * gap
        bulge = np.where(inside, 1.0, 1.0 + bulge * 0.9)
        wind2d = np.maximum(wind2d, bulge.astype(np.float32))
    # Add mild south drain channel (gap along y~4)
    wind2d = np.clip(wind2d, 1.0, 2.4)
    zs = np.arange(nz) * res_z
    out = np.zeros((nx, ny, nz), dtype=np.float32)
    for iz, z in enumerate(zs):
        # Wind factor decays a little with altitude but stays near buildings
        decay = 1.0 - 0.25 * min(z / DRONE_Z_MAX_M, 1.0)
        out[:, :, iz] = 1.0 + (wind2d - 1.0) * decay
    return out


# --------------------------------------------------------------------------
# Utils for flood-on-edge
# --------------------------------------------------------------------------

def _flood_at_xy(flood_tensor: np.ndarray, dem_shape: Tuple[int, int], x: float, y: float) -> float:
    H, W = dem_shape
    # flood tensor index [iy, ix] matching dem
    ix = int(np.clip(round(x), 0, W - 1))
    iy = int(np.clip(round(y), 0, H - 1))
    return float(flood_tensor[iy, ix])


# --------------------------------------------------------------------------
# Ground barrier-free A*
# --------------------------------------------------------------------------

def ground_astar(
    slice_dict: dict,
    start_id: int,
    goal_id: int,
    weights: Optional[Dict[str, float]] = None,
    profile: str = "wheelchair",
    flood_tensor: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> RouteResult:
    """Barrier-free ground A* over slice ground graph.

    Hard-block (wheelchair):
      - flood >= 0.10 m (screening tensor)
      - slope > 1:12 (≈0.0833)  (slope from edge or ramp node; sheltered vs not)
      - width < 1.2 m
      - kerb > 0.02 m (approximated by slope = kerb proxy when width small; direct kerb not in synthetic slice)
      - stairs / no-ramp when slope high

    Cost = w_time * time + w_detour * detour_excess + penalties.
    Time = length / speed (1.1 m/s).  Shelter gives small time bonus
    (covered linkway preferred) but is not a hard requirement.

    Args:
        slice_dict: from slice_data.build_slice
        start_id, goal_id: node ids
        weights: dict with w_time etc. (missing keys default)
        profile: "wheelchair" or "ambulant"
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    else:
        w = DEFAULT_WEIGHTS.copy()
        w.update(weights)
        weights = w

    prof = GroundProfile(name=profile) if profile == "wheelchair" else GroundProfile(
        name="ambulant", flood_block_m=0.30, slope_max=0.20, width_min_m=0.9, speed_mps=1.4
    )

    # Build node map
    nodes = slice_dict.get("nodes", [])
    edges = slice_dict.get("edges", [])
    node_by_id = {n["id"]: n for n in nodes}
    if start_id not in node_by_id or goal_id not in node_by_id:
        return RouteResult(success=False, notes=f"start/goal id not in graph (start={start_id} goal={goal_id})")
    # adjacency
    adj: Dict[int, List[dict]] = {}
    for e in edges:
        u, v = e["u"], e["v"]
        adj.setdefault(u, []).append(e)
        adj.setdefault(v, []).append(e)

    # Flood tensor (screening) — derived from DEM if not supplied
    dem = slice_dict.get("dem")
    if flood_tensor is None:
        try:
            flood_tensor = build_flood_tensor(slice_dict)
        except Exception as ex:
            flood_tensor = None
            if verbose:
                print(f"flood_tensor unavailable: {ex}")

    # A*
    # heuristic: euclidean / speed * w_time
    def heuristic(nid: int) -> float:
        a = node_by_id[nid]
        b = node_by_id[goal_id]
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"]) / max(prof.speed_mps, 1e-6) * weights["w_time"]

    # Edge cost helper
    def edge_cost(e: dict) -> Tuple[float, dict]:
        u, v = e["u"], e["v"]
        length = float(e.get("length_m", 10.0))
        slope = float(e.get("slope", 0.0))
        width = float(e.get("width_m", 2.0))
        sheltered = bool(e.get("sheltered", False))
        has_ramp = bool(e.get("has_ramp", False))
        # Mid-point flood
        a = node_by_id[u]
        b = node_by_id[v]
        mx, my = (a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0
        flood_h = 0.0
        if flood_tensor is not None and dem is not None:
            flood_h = _flood_at_xy(flood_tensor, dem.shape, mx, my)
        # Also check flood at endpoints (wheelchair needs both ends)
        if flood_tensor is not None and dem is not None:
            f0 = _flood_at_xy(flood_tensor, dem.shape, a["x"], a["y"])
            f1 = _flood_at_xy(flood_tensor, dem.shape, b["x"], b["y"])
            flood_h = max(flood_h, f0, f1)
        # Hard blocks
        if flood_h >= prof.flood_block_m - 1e-9:  # >=0.10
            return math.inf, {"flood": flood_h, "blocked": "flood"}
        if slope > prof.slope_max + 1e-6 and not has_ramp:
            return math.inf, {"blocked": "slope", "slope": slope}
        if slope > prof.slope_max + 0.02 and has_ramp:
            # Even ramp nodes with slope > threshold+0.02 are not barrier-free
            return math.inf, {"blocked": "ramp_slope", "slope": slope}
        if width < prof.width_min_m - 1e-6:
            return math.inf, {"blocked": "width", "width": width}
        # Kerb proxy: very small width with high slope suggests kerb; already covered
        time_s = length / max(prof.speed_mps, 1e-6)
        cost = weights["w_time"] * time_s
        # w_detour: detour excess vs straight (length - euclidean) penalised
        straight = length  # edge itself is straight; detour at edge level is 0
        # shelter bonus: covered linkway slightly preferred (lower effective time)
        if sheltered:
            cost *= 0.95
        # w_detour penalises longer-than-straight at path level via heuristic; per edge we add small
        cost += weights.get("w_detour", 0.0) * 0.02 * length / 10.0
        return cost, {"flood": flood_h, "slope": slope, "width": width, "time_s": time_s, "length": length}

    open_heap: List[Tuple[float, int]] = []
    heapq.heappush(open_heap, (heuristic(start_id), start_id))
    came_from: Dict[int, int] = {}
    g_score: Dict[int, float] = {start_id: 0.0}
    f_score: Dict[int, float] = {start_id: heuristic(start_id)}
    max_flood_on_path = 0.0
    found = False
    visited = set()

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal_id:
            found = True
            break
        for e in adj.get(cur, []):
            nbr = e["v"] if e["u"] == cur else e["u"]
            c, info = edge_cost(e)
            if not math.isfinite(c):
                continue
            nb_g = g_score[cur] + c
            if nb_g < g_score.get(nbr, math.inf) - 1e-9:
                # Edge flood tracked for final metric (approximate: last edge flood)
                max_flood_on_path = max(max_flood_on_path, info.get("flood", 0.0))
                g_score[nbr] = nb_g
                f_score[nbr] = nb_g + heuristic(nbr)
                came_from[nbr] = cur
                heapq.heappush(open_heap, (f_score[nbr], nbr))

    if not found:
        return RouteResult(success=False, notes="no barrier-free ground path with current flood/slope/width blocks (advisory: check flood tensor; water >=0.10 m blocked for wheelchair profile)", cost=math.inf, max_flood_m=max_flood_on_path)

    # Reconstruct
    path_ids: List[int] = []
    cur = goal_id
    while cur != start_id:
        path_ids.append(cur)
        cur = came_from.get(cur, start_id)
        if cur not in came_from and cur != start_id:
            break
    path_ids.append(start_id)
    path_ids.reverse()
    length_m = 0.0
    # recompute max flood along path more precisely
    for i in range(len(path_ids) - 1):
        u, v = path_ids[i], path_ids[i + 1]
        e = next((ee for ee in adj.get(u, []) if (ee["v"] == v and ee["u"] == u) or (ee["u"] == v and ee["v"] == u)), None)
        if e is not None:
            _, info = edge_cost(e)
            length_m += info.get("length", 0.0)
            max_flood_on_path = max(max_flood_on_path, info.get("flood", 0.0))
    total_cost = g_score[goal_id]
    time_s = total_cost / max(weights["w_time"], 1e-6)  # approx
    return RouteResult(success=True, path=path_ids, length_m=length_m, cost=total_cost, time_s=time_s, max_flood_m=max_flood_on_path, steps=len(path_ids), notes=f"barrier-free (screening) ground route, flood<{prof.flood_block_m:.2f} m, slope<={prof.slope_max:.3f}, width>={prof.width_min_m:.1f} m; advisory only")


# --------------------------------------------------------------------------
# Drone 3-D A* (0-60 m)
# --------------------------------------------------------------------------

def _nearest_pads(slice_dict: dict, target_xy: Optional[Tuple[float, float]] = None) -> List[dict]:
    pads = slice_dict.get("landing_pads", [])
    if not target_xy:
        return pads
    tx, ty = target_xy
    return sorted(pads, key=lambda p: (p["x"] - tx) ** 2 + (p["y"] - ty) ** 2)


def drone_astar(
    slice_dict: dict,
    start_xyz: Tuple[float, float, float],
    goal_xyz: Optional[Tuple[float, float, float]] = None,
    goal_pad_id: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    sdf: Optional[np.ndarray] = None,
    gnss_shadow: Optional[np.ndarray] = None,
    wind: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> RouteResult:
    """3-D grid A* for micro-drone, 0-60 m AGL (NOT 120 m).

    Hard rejects:
      - SDF < SAFETY_MARGIN_M (1.0 m) — collision screening
      - z outside [0, 60] — solver envelope (Research sec 1, CAAS ceiling)
      - inside building (sdf negative)

    Penalty layers (analytic surrogates, not CFD):
      - GNSS shadow: multiplied by w_shadow
      - Wind speed-up factor: multiplied by w_wind
      Energy surrogate: 0.10 per metre + climb penalty.

    Landing pad goal: if goal_xyz is None and goal_pad_id given, use that pad;
    if both None, use nearest pad to the goal direction (centre of slice).

    Args:
        start_xyz: (x,y,z) AGL
        goal_xyz: (x,y,z) AGL; overrides pad if given
        goal_pad_id: pad id from slice
        weights: w_time, w_energy, w_shadow, w_wind, w_detour
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    else:
        w = DEFAULT_WEIGHTS.copy()
        w.update(weights)
        weights = w

    res_xy = DRONE_RES_XY_M
    res_z = DRONE_RES_Z_M
    nx = int(math.ceil(EXTENT_M / res_xy))  # 50
    ny = nx
    nz = 16  # 0,4,...,60
    # Clamp start/goal to envelope
    def clamp_xyz(xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        x = float(np.clip(x, 0, EXTENT_M))
        y = float(np.clip(y, 0, EXTENT_M))
        z = float(np.clip(z, DRONE_Z_MIN_M, DRONE_Z_MAX_M))
        return (x, y, z)

    start_xyz = clamp_xyz(start_xyz)
    # Resolve goal
    if goal_xyz is None and goal_pad_id is not None:
        pads = {p["id"]: p for p in slice_dict.get("landing_pads", [])}
        if goal_pad_id in pads:
            p = pads[goal_pad_id]
            goal_xyz = (float(p["x"]), float(p["y"]), float(p["z"]))
        else:
            # nearest pad
            nearest = _nearest_pads(slice_dict, (start_xyz[0], start_xyz[1]))[0] if slice_dict.get("landing_pads") else None
            if nearest:
                goal_xyz = (float(nearest["x"]), float(nearest["y"]), float(nearest["z"]))
    if goal_xyz is None:
        # Default: open-court plaza pad at (130,140,0) if exists else centre
        nearest = _nearest_pads(slice_dict, (130.0, 140.0))
        if nearest:
            p = nearest[0]
            goal_xyz = (float(p["x"]), float(p["y"]), float(p["z"]))
        else:
            goal_xyz = (130.0, 140.0, 5.0)
    goal_xyz = clamp_xyz(goal_xyz)

    # Build / fetch layers (cached by caller if available)
    if sdf is None:
        sdf = build_sdf(slice_dict, res_xy=res_xy, res_z=res_z)
    if gnss_shadow is None:
        gnss_shadow = build_gnss_shadow(slice_dict, res_xy=res_xy, nz=nz, res_z=res_z)
    if wind is None:
        wind = build_wind(slice_dict, res_xy=res_xy, nz=nz, res_z=res_z)

    def to_voxel(xyz: Tuple[float, float, float]) -> Tuple[int, int, int]:
        x, y, z = xyz
        ix = int(np.clip(math.floor(x / res_xy), 0, nx - 1))
        iy = int(np.clip(math.floor(y / res_xy), 0, ny - 1))
        iz = int(np.clip(round(z / res_z), 0, nz - 1))
        return (ix, iy, iz)

    def to_xyz(voxel: Tuple[int, int, int]) -> Tuple[float, float, float]:
        ix, iy, iz = voxel
        return ((ix + 0.5) * res_xy, (iy + 0.5) * res_xy, iz * res_z)

    start_v = to_voxel(start_xyz)
    goal_v = to_voxel(goal_xyz)

    # Nearest-passable helper for pads/start that may sit on roof (sdf~0)
    def nearest_passable(v: Tuple[int, int, int], radius: int = 2) -> Tuple[int, int, int]:
        if sdf[v[0], v[1], v[2]] >= SAFETY_MARGIN_M:
            return v
        best = v
        best_sdf = -1e9
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    ix, iy, iz = v[0] + dx, v[1] + dy, v[2] + dz
                    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                        s = float(sdf[ix, iy, iz])
                        if s > best_sdf:
                            best_sdf = s
                            best = (ix, iy, iz)
        return best

    # Make start/goal passable (pads at z=0 or rooftop)
    # Allow goal pad voxel even if sdf small: force passable flag set
    forced_passable = {start_v, goal_v}
    start_v = nearest_passable(start_v, radius=2)
    goal_v = nearest_passable(goal_v, radius=2)
    if goal_v != to_voxel(goal_xyz):
        forced_passable.add(goal_v)
        if verbose:
            print(f"goal adjusted from {to_voxel(goal_xyz)} to nearest passable {goal_v} (sdf={sdf[goal_v[0],goal_v[1],goal_v[2]]:.2f})")

    # Heuristic (admissible): w_time * euclidean / drone_speed
    def heuristic(v: Tuple[int, int, int]) -> float:
        # euclidean in metres
        dx = (v[0] - goal_v[0]) * res_xy
        dy = (v[1] - goal_v[1]) * res_xy
        dz = (v[2] - goal_v[2]) * res_z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        # shadow/wind minimal penalty is >=0, so heuristic without them is admissible
        return weights["w_time"] * dist / max(DRONE_SPEED_MPS, 1e-6)

    # Neighbours: 6-cardinal + 12 edge diagonals (reduces Manhattan jag)
    # Step vectors (dx, dy, dz) in voxel units, with metric length factor
    neigh_offsets: List[Tuple[int, int, int]] = []
    # 6 cardinals
    neigh_offsets += [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    # 12 edge diagonals (dx,dy,0) and (dx,0,dz) etc. + 8 corners optionally trimmed for speed
    neigh_offsets += [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]
    # 8 corners (optional, adds smoothness but cost); keep for smoother path if performance allows
    # neigh_offsets += [(1,1,1),(1,1,-1),...]
    # For speed we keep 18 neighbours

    open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start_v), 0, start_v))
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_score: Dict[Tuple[int, int, int], float] = {start_v: 0.0}
    shadow_exp = 0.0
    wind_exp = 0.0
    visited = set()
    found = False
    iterations = 0
    max_iterations = nx * ny * nz * 2  # cap
    min_sdf_on_path = math.inf

    while open_heap and iterations < max_iterations:
        _, _, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        iterations += 1
        if cur == goal_v:
            found = True
            break
        cx, cy, cz = cur
        # track min sdf
        try:
            min_sdf_on_path = min(min_sdf_on_path, float(sdf[cx, cy, cz]))
        except Exception:
            pass
        for dx, dy, dz in neigh_offsets:
            ix, iy, iz = cx + dx, cy + dy, cz + dz
            if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                continue
            nxt = (ix, iy, iz)
            # SDF collision reject
            s = float(sdf[ix, iy, iz])
            if s < SAFETY_MARGIN_M and nxt not in forced_passable:
                continue
            # GNSS shadow + wind costs (sample destination)
            sh = float(gnss_shadow[ix, iy, iz])
            wi = float(wind[ix, iy, iz])
            # Step metric length
            step_m = math.sqrt((dx * res_xy) ** 2 + (dy * res_xy) ** 2 + (dz * res_z) ** 2)
            time_step = step_m / max(DRONE_SPEED_MPS, 1e-6)
            energy_step = step_m * 0.10 + abs(dz) * res_z * 0.05  # surrogate
            cost_step = (
                weights["w_time"] * time_step
                + weights["w_energy"] * energy_step * 0.1  # scale energy to metre-equivalent
                + weights["w_shadow"] * sh * 0.8  # penalty per step
                + weights["w_wind"] * wi * 0.4
            )
            # Small detour / backtrack penalty: discourage revisiting via g_score comparison
            # w_detour inflates steps away from goal; heuristic already accounts. Add slight
            # penalty if step increases heuristic strongly.
            # Keep simple: add w_detour * 0.01 * step_m / 10
            cost_step += weights.get("w_detour", 0.0) * 0.02 * (step_m / res_xy)
            nb_g = g_score[cur] + cost_step
            if nb_g+1e-9 < g_score.get(nxt, math.inf):
                g_score[nxt] = nb_g
                came_from[nxt] = cur
                heapq.heappush(open_heap, (nb_g + heuristic(nxt), iterations, nxt))
                # accumulate exposures on the fly (for metric reporting; final exposure recomputed on path)
            # else ignore

    if not found:
        return RouteResult(success=False, notes=f"no drone path in 0-{int(DRONE_Z_MAX_M)} m with SDF>={SAFETY_MARGIN_M:.1f} m (advisory replay; solver envelope 0-60 m, not 120 m)", cost=math.inf, min_sdf_m=min_sdf_on_path, steps=iterations)

    # Reconstruct voxel path
    voxel_path: List[Tuple[int, int, int]] = []
    cur = goal_v
    while cur != start_v:
        voxel_path.append(cur)
        cur = came_from[cur]
    voxel_path.append(start_v)
    voxel_path.reverse()

    # Metrics along path
    length_m = 0.0
    time_s = 0.0
    energy = 0.0
    shadow_exposure = 0.0
    wind_exposure = 0.0
    true_cost = 0.0  # not used; path cost is g_score[goal] but recompute
    for i in range(len(voxel_path) - 1):
        a = voxel_path[i]
        b = voxel_path[i + 1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        step_m = math.sqrt((dx * res_xy) ** 2 + (dy * res_xy) ** 2 + (dz * res_z) ** 2)
        length_m += step_m
        time_s += step_m / max(DRONE_SPEED_MPS, 1e-6)
        energy += step_m * 0.10 + abs(dz) * res_z * 0.05
        sh = float(gnss_shadow[b[0], b[1], b[2]])
        wi = float(wind[b[0], b[1], b[2]])
        shadow_exposure += sh * step_m
        wind_exposure += wi * step_m

    xyz_path = [to_xyz(v) for v in voxel_path]
    # Recompute min SDF along the actual recovered path (not visited exploration)
    path_min_sdf = math.inf
    for v in voxel_path:
        path_min_sdf = min(path_min_sdf, float(sdf[v[0], v[1], v[2]]))
    # Include clamped start/goal exact if available (path ends at snapped voxel near pad)
    # Path cost is weighted with evolved weights (for evolve search)
    total_weighted_cost = g_score[goal_v]
    # True scalar proxy for evaluator: unweighted (for reporting)
    true_scalar = -(time_s * 1.0 + energy * 0.5 + shadow_exposure * 0.5 + wind_exposure * 0.2)
    return RouteResult(
        success=True,
        path=xyz_path,
        length_m=length_m,
        cost=total_weighted_cost,
        time_s=time_s,
        energy=energy,
        shadow_exposure=shadow_exposure,
        wind_exposure=wind_exposure,
        steps=len(voxel_path),
        min_sdf_m=path_min_sdf,
        notes=f"drone 3D A* 0-{int(DRONE_Z_MAX_M)} m (NOT 120 m), safety SDF>={SAFETY_MARGIN_M:.1f} m, shadow/wind surrogates, advisory replay",
    )


# --------------------------------------------------------------------------
# Convenience wrappers used by evolve.py
# --------------------------------------------------------------------------

def plan_ground_route(
    slice_dict: dict,
    start_id: int,
    goal_id: int,
    weights: Optional[Dict[str, float]] = None,
) -> RouteResult:
    return ground_astar(slice_dict, start_id, goal_id, weights=weights)


def plan_drone_route(
    slice_dict: dict,
    start_xyz: Tuple[float, float, float],
    goal_xyz: Optional[Tuple[float, float, float]] = None,
    goal_pad_id: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    sdf: Optional[np.ndarray] = None,
    gnss_shadow: Optional[np.ndarray] = None,
    wind: Optional[np.ndarray] = None,
) -> RouteResult:
    return drone_astar(
        slice_dict,
        start_xyz,
        goal_xyz=goal_xyz,
        goal_pad_id=goal_pad_id,
        weights=weights,
        sdf=sdf,
        gnss_shadow=gnss_shadow,
        wind=wind,
    )


def score_candidate(weights: Dict[str, float], slice_dict: dict, ground_start: Optional[int] = None, ground_goal: Optional[int] = None, drone_start: Tuple[float, float, float] = (25.0, 25.0, 10.0), drone_goal: Tuple[float, float, float] = (130.0, 140.0, 5.0)) -> float:
    """Single scalar for CD contract: maximise -(time+energy+shadow+wind+flood).

    Hard rejects return very low score.  Uses true (fixed) scalarisation
    regardless of ``weights``; ``weights`` shape the heuristic choice, scalar
    measures the outcome.  If an external ``evaluator`` module exists,
    evolve.py will delegate there; this is the local fallback.
    """
    # Ground
    nodes = slice_dict.get("nodes", [])
    if ground_start is None:
        # pick first junction to last entrance
        junctions = [n for n in nodes if n["kind"] == "junction"]
        entrances = [n for n in nodes if n["kind"] == "entrance"]
        if junctions and entrances:
            ground_start, ground_goal = junctions[0]["id"], entrances[0]["id"]
        elif nodes:
            ground_start, ground_goal = nodes[0]["id"], nodes[-1]["id"]
        else:
            ground_start = ground_goal = 0
    g = ground_astar(slice_dict, int(ground_start), int(ground_goal), weights=weights)
    d = drone_astar(slice_dict, drone_start, goal_xyz=drone_goal, weights=weights)
    if not g.success and not d.success:
        return -1e6
    score = 0.0
    if g.success:
        # ground: time + max flood (screening) hard already
        score -= g.time_s * 1.0
        # small penalty for long ground route (detour)
        score -= g.length_m * 0.01
    else:
        score -= 1e3  # penalise but not hard fail if drone ok? For MVP, ground failure is heavy
        score -= 500.0  # flooded reroute impossibility
    if d.success:
        score -= d.time_s * 1.0
        score -= d.energy * 0.3
        score -= d.shadow_exposure * 0.4
        score -= d.wind_exposure * 0.2
    else:
        score -= 1e3
    return float(score)


def build_slice(seed: int = 7) -> dict:
    """Clinic entry: lazy slice_data build or fallback."""
    return _lazy_build_slice(seed=seed)


__all__ = [
    "DRONE_Z_MAX_M",
    "WHEELCHAIR_FLOOD_BLOCK_M",
    "DEFAULT_WEIGHTS",
    "WEIGHT_KEYS",
    "RouteResult",
    "GroundProfile",
    "ground_astar",
    "drone_astar",
    "plan_ground_route",
    "plan_drone_route",
    "score_candidate",
    "build_slice",
    "build_sdf",
    "build_flood_tensor",
    "build_gnss_shadow",
    "build_wind",
]


if __name__ == "__main__":
    # Standalone smoke test — works even when slice_data/evaluator missing (fallback)
    print("AETHER-SG routing.py — 0-60 m drone envelope (NOT 120 m) — cached-replay / screening tensor")
    print(f"Wheelchair flood block >= {WHEELCHAIR_FLOOD_BLOCK_M:.2f} m (screening, not hydrodynamics)")
    sl = build_slice(seed=7)
    print(f"slice nodes={len(sl.get('nodes', []))} edges={len(sl.get('edges', []))} pads={len(sl.get('landing_pads', []))} meta={sl.get('meta', {})}")
    # Ground: flooded edge reroute demo (pick two junctions far apart)
    nodes = sl.get("nodes", [])
    junctions = [n for n in nodes if n["kind"] == "junction"]
    entrances = [n for n in nodes if n["kind"] == "entrance"]
    if junctions and entrances:
        g = ground_astar(sl, junctions[0]["id"], entrances[0]["id"])
        print(f"ground_astar success={g.success} len={g.length_m:.1f} m cost={g.cost:.2f} time={g.time_s:.1f} s max_flood={g.max_flood_m:.2f} path={g.path[:6]} notes={g.notes}")
    else:
        print("ground demo skipped: no junctions")
    # Flood tensor stats
    f = build_flood_tensor(sl)
    print(f"flood tensor shape={f.shape} max={f.max():.2f} m mean={f.mean():.3f} flooded>=0.10 cells={(f>=0.10).sum()}")
    # Drone 0-60 m
    d = drone_astar(sl, (25.0, 25.0, 12.0), goal_xyz=(130.0, 140.0, 5.0))
    print(f"drone_astar success={d.success} len={d.length_m:.1f} m time={d.time_s:.1f} s energy={d.energy:.1f} shadow={d.shadow_exposure:.1f} wind={d.wind_exposure:.1f} steps={d.steps} min_sdf={d.min_sdf_m:.2f}")
    print("routing.py OK — advisory, offline cached replay")
