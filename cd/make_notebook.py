"""AETHER-SG: build the Computational Discovery notebook.

Builds ``cd/aether_evolve.ipynb`` (setup -> mutable -> scoring) and the
staged slice files that the notebook loads from ``DATA_DIR``.

Slice: deterministic synthetic 200 x 200 m Toa Payoh-like estate
(``src/aether`` schema, seed 7 train / seed 99 holdout).  Data is
SIMULATED; see ``Admin/Research-AETHER-SG.md`` sections 1, 11, 12.

Hard scope retained here:
  * drone planning envelope is 0-60 m AGL only.
  * flood is a screening depth tensor; wheelchair edges with depth
    at or above 0.10 m are treated as blocked.
  * the candidate is judged by one scalar score computed in the
    hidden scoring cells; single-scalar maximisation only.
  * what ships as a demo is a cached replay of an offline search:
    heavy search runs in the cloud beforehand, the chosen solver
    then runs offline with no live search on stage.

Local deps: Python 3.11+ stdlib + numpy only.
Run:  python cd/make_notebook.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (shared by builder, staged slices, notebook and tests)
# ---------------------------------------------------------------------------
DATA_DIR = "/data/input"
TRAIN_NPZ_NAME = "toapayoh_train_slice.npz"
TRAIN_JSON_NAME = "toapayoh_train_slice.json"
HOLDOUT_NPZ_NAME = "toapayoh_holdout_slice.npz"
HOLDOUT_JSON_NAME = "toapayoh_holdout_slice.json"
TRAIN_SEED = 7
HOLDOUT_SEED = 99
Z_MAX_M = 60.0
Z_MIN_M = 0.0
VOXEL_M = 2.0
WHEELCHAIR_BLOCK_M = 0.10
NOTEBOOK_NAME = "aether_evolve.ipynb"

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:  # optional: reuse sibling core when present (local builder only)
    from aether.slice_data import build_slice  # type: ignore
    from aether.evaluator import build_sdf as _eval_build_sdf  # type: ignore
    from aether.evaluator import gnss_shadow as _eval_gnss  # type: ignore
    from aether.evaluator import flood_tensor as _eval_flood  # type: ignore
    _HAS_AETHER = True
except Exception:
    build_slice = None  # type: ignore
    _HAS_AETHER = False


# ---------------------------------------------------------------------------
# Notebook cell sources.  Setup + scoring are fixed; the middle block is the
# mutable solver that an offline search may rewrite.  Only the plain
# setup/mutable/scoring cell text lives here; keep it self-contained so it
# runs in a bare CD sandbox (no sibling package needed).
# ---------------------------------------------------------------------------
SETUP_CODE = r'''# AETHER-SG setup (fixed): load the staged slice tensors.
# DATA_DIR=/data/input (CD staging root; train slice display name
# toapayoh_train_slice.npz).
import json
import os
import sys

import numpy as np

DATA_DIR = "/data/input"
TRAIN_NPZ = os.path.join(DATA_DIR, "toapayoh_train_slice.npz")
TRAIN_JSON = os.path.join(DATA_DIR, "toapayoh_train_slice.json")
VOXEL_M = 2.0
Z_MAX_M = 60.0
WHEELCHAIR_BLOCK_M = 0.10


def _load_bundle():
    if os.path.exists(TRAIN_NPZ) and os.path.exists(TRAIN_JSON):
        z = np.load(TRAIN_NPZ)
        bundle = {
            "dem": z["dem"],
            "flood_depth": z["flood_depth"],
            "sdf": z["sdf"],
            "shadow": z["shadow"],
            "voxel_m": float(VOXEL_M),
            "z_max_m": float(Z_MAX_M),
        }
        with open(TRAIN_JSON, "r", encoding="utf-8") as f:
            side = json.load(f)
        task = side["task"]
        return bundle, task, side
    raise FileNotFoundError(
        "Staged slice not found at " + TRAIN_NPZ + ". "
        "Upload the train slice with cd_upload first (see README-CD.md)."
    )


bundle, task, sidecar = _load_bundle()
print("bundle dem", bundle["dem"].shape, "sdf", bundle["sdf"].shape,
      "flood max", float(bundle["flood_depth"].max()))
print("task", task)
'''
BRIEF_MD = """# AETHER-SG: offline planner for one cached 200 x 200 m estate slice

You are the solver author for a cached replay build. Heavy search already
ran in the cloud; the code you write here runs fully offline and is replayed
on stage. Do not add network use, do not add live search, do not claim the
stage replay is a live search. Data is a simulated Toa Payoh-like estate
(official datasets would replace it without changing the solver interface).

## Goals (qualitative - exact weights live in hidden scoring cells)

Produce two short, dry, collision-free paths:
- ground: a step-free ground route for a wheelchair profile that stays on
  dry cells and avoids steep cells;
- air: a flight path inside the 0-60 m envelope that stays clear of
  buildings and prefers clear-sky cells.

Prefer short paths, but staying dry and staying clear matters more than
saving a few metres. The score rewards short paths and penalises wet cells,
shadow cells and any hit on a building.

## Envelope and screening notes

- Height: solve only between 0 m and 60 m above the ground datum. The grid
  tops out at 60 m, so any path you emit in range is inside by construction.
- Flood: the depth tensor is a screening screen (water level minus ground,
  restricted to the drain-connected area), not a hydraulic model. Operational
  water levels belong to the water agency; here a ground cell whose depth is
  at or above 0.10 m must be treated as blocked for the wheelchair profile.
- Ground also blocks on steep cells (slope above the barrier-free limit in
  the baseline) and on building footprints.

## Data you read (staged under DATA_DIR)

- `bundle["dem"]` shape (100, 100), float32: ground height at 2 m cells.
- `bundle["flood_depth"]` shape (100, 100), float32: screening depth (m).
- `bundle["sdf"]` shape (30, 100, 100), float32: signed distance (m),
  negative inside a building box, positive in free air. Height levels are
  0, 2, ..., 60 m.
- `bundle["shadow"]` shape (30, 100, 100), uint8 0/1: building-occlusion
  proxy for sky visibility. Advisory only.
- `task` dict: `ground_start_xy`, `ground_goal_xy` (metres),
  `drone_start_xyz`, `drone_goal_xyz` (metres, z above datum).

## Contract

Implement `solve_candidate(bundle, task, params=None)` returning a dict with:
- `"ground_path"`: list of [x, y] metre pairs, inclusive, 4-neighbour steps;
- `"drone_path"`: list of [x, y, z] metre triples, inclusive, 6-neighbour
  steps, every z in [0, 60].

Rules:
- Read start/goal only from `task`. Never bake in place names or fixed
  map positions; the same solver must work on a different slice.
- Keep the returned paths inside the 200 x 200 m extent with exact
  endpoints from `task`.
- Keep dependencies to stdlib + numpy. Deterministic output only.
"""
MUTABLE_CODE = r'''# AETHER-SG mutable solver (baseline heuristic).
# A* over grids. Parameters shape the heuristic; the search harness may
# retune them. Reads start/goal only from `task` (no fixed positions).
import heapq
import math

import numpy as np

BASELINE_PARAMS = {
    "ground_w_slope": 8.0,
    "ground_curve": 0.4,
    "drone_w_shadow": 1.5,
    "drone_w_alt": 0.6,
    "drone_w_clear": 0.3,
    "cruise_lo": 16.0,
    "cruise_hi": 32.0,
    "sdf_margin": 0.5,
    "slope_max": 0.0833,
}

_NB4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_NB6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _to_cell_xy(x, y, voxel_m, ny, nx):
    ix = int(min(max(int(math.floor(x / voxel_m)), 0), nx - 1))
    iy = int(min(max(int(math.floor(y / voxel_m)), 0), ny - 1))
    return ix, iy


def _to_cell_xyz(x, y, z, voxel_m, nz, ny, nx):
    ix = int(min(max(int(math.floor(x / voxel_m)), 0), nx - 1))
    iy = int(min(max(int(math.floor(y / voxel_m)), 0), ny - 1))
    iz = int(min(max(int(round(z / voxel_m)), 0), nz - 1))
    return ix, iy, iz


def plan_ground_route(dem, flood_depth, sdf_ground, start_xy, goal_xy,
                      params=None, voxel_m=2.0, block_m=0.10):
    """A* on the 2-D grid. Blocked: flood >= block_m, steep, footprint."""
    p = dict(BASELINE_PARAMS)
    if params:
        p.update(params)
    ny, nx = dem.shape
    six, siy = _to_cell_xy(start_xy[0], start_xy[1], voxel_m, ny, nx)
    gix, giy = _to_cell_xy(goal_xy[0], goal_xy[1], voxel_m, ny, nx)
    blocked = (flood_depth >= block_m) | (sdf_ground < 0.0)
    # steep cells: neighbour height jump above the barrier-free limit
    dzx = np.zeros_like(dem)
    dzy = np.zeros_like(dem)
    dzx[:, 1:] = np.abs(dem[:, 1:] - dem[:, :-1]) / voxel_m
    dzy[1:, :] = np.abs(dem[1:, :] - dem[:-1, :]) / voxel_m
    steep = np.maximum(dzx, dzy) > float(p["slope_max"])
    blocked = blocked | steep
    blocked[siy, six] = False
    blocked[giy, gix] = False

    def heur(c):
        return (abs(c[0] - gix) + abs(c[1] - giy)) * voxel_m

    open_heap = [(heur((six, siy)), 0.0, (six, siy))]
    came = {}
    gscore = {(six, siy): 0.0}
    closed = set()
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == (gix, giy):
            break
        cx, cy = cur
        for dx, dy in _NB4:
            n = (cx + dx, cy + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or blocked[n[1], n[0]]:
                continue
            jump = max(abs(float(dem[n[1], n[0]]) - float(dem[cy, cx])) / voxel_m, 0.0)
            over = max(jump - float(p["slope_max"]), 0.0)
            step = voxel_m + float(p["ground_w_slope"]) * over * 10.0
            step += 0.0  # curve term reserved; baseline keeps it near zero
            ng = g + step
            if n not in gscore or ng < gscore[n] - 1e-9:
                gscore[n] = ng
                came[n] = cur
                heapq.heappush(open_heap, (ng + heur(n), ng, n))
    if (gix, giy) not in closed:
        return None
    cells = [(gix, giy)]
    while cells[-1] != (six, siy):
        cells.append(came[cells[-1]])
    cells.reverse()
    return [[(ix + 0.5) * voxel_m, (iy + 0.5) * voxel_m] for ix, iy in cells]


def plan_drone_route(sdf, shadow, start_xyz, goal_xyz, params=None,
                     voxel_m=2.0):
    """A* on the 3-D grid. Blocked: sdf below margin. 0-60 m only."""
    p = dict(BASELINE_PARAMS)
    if params:
        p.update(params)
    nz, ny, nx = sdf.shape
    six, siy, siz = _to_cell_xyz(start_xyz[0], start_xyz[1], start_xyz[2],
                                voxel_m, nz, ny, nx)
    gix, giy, giz = _to_cell_xyz(goal_xyz[0], goal_xyz[1], goal_xyz[2],
                                voxel_m, nz, ny, nx)
    margin = float(p["sdf_margin"])
    passable = sdf >= margin
    # nearest free cell if start/goal sits on/inside a box
    def nearest_free(c):
        if passable[c[2], c[1], c[0]]:
            return c
        best, bs = c, -1e18
        for dz in range(-2, 3):
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    n = (c[0] + dx, c[1] + dy, c[2] + dz)
                    if 0 <= n[0] < nx and 0 <= n[1] < ny and 0 <= n[2] < nz:
                        s = float(sdf[n[2], n[1], n[0]])
                        if s > bs:
                            bs, best = s, n
        return best

    six, siy, siz = nearest_free((six, siy, siz))
    gix, giy, giz = nearest_free((gix, giy, giz))

    def heur(c):
        return (abs(c[0] - gix) + abs(c[1] - giy) + abs(c[2] - giz)) * voxel_m

    def step_cost(n):
        z = n[2] * voxel_m
        sh = float(shadow[n[2], n[1], n[0]])
        if z < float(p["cruise_lo"]):
            alt = (float(p["cruise_lo"]) - z) / 40.0
        elif z > float(p["cruise_hi"]):
            alt = (z - float(p["cruise_hi"])) / 40.0
        else:
            alt = 0.0
        clear = max(float(p["sdf_margin"]) - float(sdf[n[2], n[1], n[0]]), 0.0)
        return voxel_m * (1.0 + float(p["drone_w_shadow"]) * sh
                          + float(p["drone_w_alt"]) * alt
                          + float(p["drone_w_clear"]) * clear * 4.0)

    open_heap = [(heur((six, siy, siz)), 0.0, (six, siy, siz))]
    came = {}
    gscore = {(six, siy, siz): 0.0}
    closed = set()
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == (gix, giy, giz):
            break
        cx, cy, cz = cur
        for dx, dy, dz in _NB6:
            n = (cx + dx, cy + dy, cz + dz)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny and 0 <= n[2] < nz):
                continue
            if not passable[n[2], n[1], n[0]]:
                continue
            ng = g + step_cost(n)
            if n not in gscore or ng < gscore[n] - 1e-9:
                gscore[n] = ng
                came[n] = cur
                heapq.heappush(open_heap, (ng + heur(n), ng, n))
    if (gix, giy, giz) not in closed:
        return None
    cells = [(gix, giy, giz)]
    while cells[-1] != (six, siy, siz):
        cells.append(came[cells[-1]])
    cells.reverse()
    return [[(ix + 0.5) * voxel_m, (iy + 0.5) * voxel_m, iz * voxel_m]
            for ix, iy, iz in cells]


def solve_candidate(bundle, task, params=None):
    """Entry point for the harness. Returns ground + drone paths."""
    dem = bundle["dem"]
    flood = bundle["flood_depth"]
    sdf = bundle["sdf"]
    shadow = bundle["shadow"]
    voxel_m = float(bundle.get("voxel_m", 2.0))
    sdf_ground = sdf[0, :, :]
    gpath = plan_ground_route(dem, flood, sdf_ground,
                              task["ground_start_xy"], task["ground_goal_xy"],
                              params=params, voxel_m=voxel_m)
    dpath = plan_drone_route(sdf, shadow, task["drone_start_xyz"],
                             task["drone_goal_xyz"], params=params,
                             voxel_m=voxel_m)
    return {"ground_path": gpath, "drone_path": dpath}
'''
SCORING_CODE = r'''# [exclude_from_prompt]
# AETHER-SG scoring (fixed, hidden from the solver prompt). Single scalar.
import numpy as np

VOXEL_M = 2.0
Z_MAX_M = 60.0
Z_MIN_M = 0.0
BLOCK_M = 0.10
EVAC_M = 1.00
DRONE_SPEED = 12.0
GROUND_SPEED = 1.1
ENERGY_PER_M = 0.30
FLOOD_W = 100.0
SHADOW_W = 20.0


def _sample_sdf(sdf, pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    v = VOXEL_M
    ix = np.clip((pts[:, 0] / v).astype(np.int64), 0, sdf.shape[2] - 1)
    iy = np.clip((pts[:, 1] / v).astype(np.int64), 0, sdf.shape[1] - 1)
    iz = np.clip((pts[:, 2] / v).astype(np.int64), 0, sdf.shape[0] - 1)
    return sdf[iz, iy, ix]


def _sample_shadow(sh, pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    v = VOXEL_M
    ix = np.clip((pts[:, 0] / v).astype(np.int64), 0, sh.shape[2] - 1)
    iy = np.clip((pts[:, 1] / v).astype(np.int64), 0, sh.shape[1] - 1)
    iz = np.clip((pts[:, 2] / v).astype(np.int64), 0, sh.shape[0] - 1)
    return sh[iz, iy, ix].astype(np.float32)


def _score_one(pts, sdf, sh, flood, ground):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0 or pts.ndim != 2 or pts.shape[0] < 2:
        return float("-inf"), {}
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    if ground and pts.shape[1] == 3:
        seg = np.hypot(seg, np.diff(pts[:, 2]))
    length = float(seg.sum())
    if length <= 0:
        return float("-inf"), {}
    n_dense = int(min(max(int(length / 1.0), len(pts)), 4000))
    t = np.linspace(0, 1, n_dense)
    dense = np.column_stack([np.interp(t, np.linspace(0, 1, len(pts)),
                                       pts[:, k]) for k in range(pts.shape[1])])
    if bool((dense[:, 2] < Z_MIN_M).any()) or bool((dense[:, 2] > Z_MAX_M).any()):
        return float("-inf"), {"why": "height outside 0-60 m"}
    sdf_vals = _sample_sdf(sdf, dense)
    collisions = int((sdf_vals < 0).sum())
    if collisions > 0:
        return float("-inf"), {"why": "building hit", "collisions": collisions}
    speed = GROUND_SPEED if ground else DRONE_SPEED
    time_s = length / max(speed, 1e-6)
    energy = ENERGY_PER_M * length
    shadow_mean = float(_sample_shadow(sh, dense).mean())
    flood_excess = 0.0
    max_flood = 0.0
    if ground and flood is not None:
        H, W = flood.shape
        ix = np.clip(np.round(dense[:, 0] / 2.0).astype(np.int64), 0, W - 1)
        iy = np.clip(np.round(dense[:, 1] / 2.0).astype(np.int64), 0, H - 1)
        vals = flood[iy, ix].astype(np.float64)
        max_flood = float(vals.max()) if vals.size else 0.0
        if bool((vals > EVAC_M).any()):
            return float("-inf"), {"why": "flood above evacuate level",
                                   "max_flood": max_flood}
        excess = np.clip(vals - BLOCK_M, 0.0, None)
        flood_excess = float(excess.mean()) if excess.size else 0.0
    score = (-(time_s + energy) - 1000.0 * collisions
             - FLOOD_W * flood_excess - SHADOW_W * shadow_mean)
    return score, {"length": length, "time_s": time_s, "energy": energy,
                   "shadow_mean": shadow_mean, "flood_excess": flood_excess,
                   "max_flood": max_flood}


def _check_steps(path, step):
    p = np.asarray(path, dtype=np.float64)
    if p.size == 0:
        return False
    d = np.abs(np.diff(p, axis=0))
    if step == "4":
        ok = ((d.sum(axis=1) == 2.0) | (d.sum(axis=1) == 0.0))
        return bool(ok.all())
    return bool((d.sum(axis=1) <= 2.0001).all())


def evaluate(candidate, bundle, task):
    sdf = bundle["sdf"]
    sh = bundle["shadow"]
    flood = bundle["flood_depth"]
    dem = bundle["dem"]
    g = candidate.get("ground_path")
    d = candidate.get("drone_path")
    if g is None or d is None:
        return float("-inf"), {"why": "missing path"}
    if not _check_steps(g, "4"):
        return float("-inf"), {"why": "ground steps not 4-neighbour"}
    garr = np.asarray(g, dtype=np.float64)
    darr = np.asarray(d, dtype=np.float64)
    if darr.ndim != 2 or darr.shape[1] != 3:
        return float("-inf"), {"why": "drone path shape"}
    gz = np.array([float(dem[int(np.clip(round(p[1] / 2.0), 0, dem.shape[0] - 1)),
                             int(np.clip(round(p[0] / 2.0), 0, dem.shape[1] - 1))])
                   for p in garr])
    g3 = np.column_stack([garr, gz])
    gs, gd = _score_one(g3, sdf, sh, flood, True)
    ds, dd = _score_one(darr, sdf, sh, None, False)
    if gs == float("-inf") or ds == float("-inf"):
        return float("-inf"), {"ground": gd, "drone": dd}
    return gs + ds, {"ground": gd, "drone": dd}


candidate = solve_candidate(bundle, task)
score, detail = evaluate(candidate, bundle, task)
print(detail["ground"].get("length"), detail["drone"].get("length"))
print("Candidate metric to be maximized: " + format(score, ".6f"))
'''


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def build_notebook() -> dict:
    cells = [
        _md(BRIEF_MD.rstrip("\n")),
        _code(SETUP_CODE.rstrip("\n")),
        _md("# Begin mutable cells"),
        _code(MUTABLE_CODE.rstrip("\n")),
        _md("# End mutable cells"),
        _code(SCORING_CODE.rstrip("\n")),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "aether": {"data_dir": DATA_DIR, "z_max_m": Z_MAX_M,
                       "block_m": WHEELCHAIR_BLOCK_M,
                       "train_npz": TRAIN_NPZ_NAME,
                       "note": "cached replay; solver runs offline"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save_notebook(path: Path | str | None = None) -> Path:
    out = Path(path) if path is not None else _HERE / NOTEBOOK_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


def _resample_mean(arr, fx=2, fy=2):
    h, w = arr.shape
    hh = h // fy * fy
    ww = w // fx * fx
    a = arr[:hh, :ww].reshape(hh // fy, fy, ww // fx, fx)
    return a.mean(axis=(1, 3)).astype(np.float32)


def _resample_max(arr, fx=2, fy=2):
    h, w = arr.shape
    hh = h // fy * fy
    ww = w // fx * fx
    a = arr[:hh, :ww].reshape(hh // fy, fy, ww // fx, fx)
    return a.max(axis=(1, 3)).astype(np.float32)


def _fallback_slice(seed: int):
    """Standalone synthetic slice with the same keys when sibling core missing."""
    rng = np.random.default_rng(int(seed))
    ny = nx = 100
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    dem = (1.5 + 0.002 * (xx - 50) + 0.001 * (yy - 50)
           - 1.2 * np.exp(-((xx - 50) / 4.0) ** 2) * np.exp(-((yy - 50) / 12.0) ** 2)
           + rng.normal(0, 0.03, size=(ny, nx)))
    dem = np.clip(dem - dem.min(), 0.0, None).astype(np.float32)
    boxes = [(30, 45, 40, 60, 24.0), (60, 75, 30, 50, 36.0),
             (52, 55, 70, 73, 6.0)]
    sdf = np.full((30, ny, nx), 60.0, dtype=np.float32)
    xc = (np.arange(nx) + 0.5) * 2.0
    yc = (np.arange(ny) + 0.5) * 2.0
    zc = np.arange(30) * 2.0
    X, Y = np.meshgrid(xc, yc)
    for (x0, x1, y0, y1, h) in boxes:
        cx, cy = (x0 + x1) * 1.0, (y0 + y1) * 1.0
        qx = np.abs(X - cx) - (x1 - x0) * 1.0
        qy = np.abs(Y - cy) - (y1 - y0) * 1.0
        for iz, z in enumerate(zc):
            qz = z - h
            ox = np.maximum(qx, 0.0)
            oy = np.maximum(qy, 0.0)
            oz = max(qz, 0.0)
            out = np.sqrt(ox**2 + oy**2 + oz**2)
            inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
            box = (out + inside).astype(np.float32)
            np.minimum(sdf[iz], box, out=sdf[iz])
    flood = np.clip(0.6 - dem, 0.0, None).astype(np.float32)
    shadow = np.zeros((30, ny, nx), dtype=bool)
    for (x0, x1, y0, y1, h) in boxes:
        ix0 = max(0, int(x0 // 2))
        ix1 = min(nx, int(x1 // 2) + 1)
        iy0 = max(0, int(y0 // 2))
        iy1 = min(ny, int(y1 // 2) + 1)
        iz1 = min(30, int(h // 2) + 1)
        shadow[:iz1, iy0:iy1, ix0:ix1] = True
    shadow = shadow.astype(np.uint8)
    return {"dem": dem, "flood_depth": flood, "sdf": sdf,
            "shadow": shadow, "buildings": [],
            "task": {"ground_start_xy": [25.0, 75.0],
                     "ground_goal_xy": [175.0, 175.0],
                     "drone_start_xyz": [20.0, 20.0, 20.0],
                     "drone_goal_xyz": [130.0, 140.0, 5.0]},
            "water_level_m": 0.6, "source": "fallback"}


def build_staged_bundle(seed: int = TRAIN_SEED, water_level_m=None):
    """Build (bundle_arrays, sidecar) for staging.

    Prefers the sibling ``src/aether`` core so staged tensors match the
    viewer/routing schema. Falls back to a local synthetic set when the
    sibling import is missing. Bundle arrays live on the 2 m grid:
    dem/flood (100, 100), sdf/shadow (30, 100, 100).
    """
    side = {}
    if _HAS_AETHER:
        sl = build_slice(seed=int(seed))
        dem200 = np.asarray(sl["dem"], dtype=np.float32)
        sdf_b = _eval_build_sdf(sl["buildings"])
        sh_b = _eval_gnss(sl["buildings"])
        wl = float(dem200.min()) + 0.5 if water_level_m is None else float(water_level_m)
        fl = _eval_flood(dem200, water_level_m=wl)
        flood200 = np.asarray(fl["depth"], dtype=np.float32)
        nodes = sl.get("nodes", [])
        juncs = [n for n in nodes if n.get("kind") == "junction"
                 and 0.0 <= float(n["x"]) <= 200.0 and 0.0 <= float(n["y"]) <= 200.0]
        juncs = sorted(juncs, key=lambda n: int(n["id"]))
        if len(juncs) >= 2:
            gs = [float(juncs[0]["x"]), float(juncs[0]["y"])]
            gg = [float(juncs[-1]["x"]), float(juncs[-1]["y"])]
        else:
            gs, gg = [25.0, 75.0], [175.0, 175.0]
        task = {"ground_start_xy": gs, "ground_goal_xy": gg,
                "drone_start_xyz": [20.0, 20.0, 20.0],
                "drone_goal_xyz": [130.0, 140.0, 5.0]}
        bundle = {
            "dem": _resample_mean(dem200),
            "flood_depth": _resample_max(flood200),
            "sdf": np.asarray(sdf_b["sdf"], dtype=np.float32),
            "shadow": np.asarray(sh_b["shadow"], dtype=np.uint8),
        }
        side = {"task": task, "buildings": sl["buildings"],
                "nodes": sl["nodes"], "edges": sl["edges"],
                "meta": sl["meta"], "water_level_m": wl,
                "voxel_m": VOXEL_M, "z_max_m": Z_MAX_M, "source": "aether"}
        return bundle, side
    fb = _fallback_slice(seed)
    bundle = {"dem": fb["dem"], "flood_depth": fb["flood_depth"],
              "sdf": fb["sdf"], "shadow": fb["shadow"]}
    side = {"task": fb["task"], "water_level_m": fb["water_level_m"],
            "voxel_m": VOXEL_M, "z_max_m": Z_MAX_M, "source": "fallback",
            "meta": {"seed": int(seed)}}
    return bundle, side


def save_staged_slice(out_dir, npz_name, json_name, seed=TRAIN_SEED,
                      water_level_m=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle, side = build_staged_bundle(seed=seed, water_level_m=water_level_m)
    npz_path = out_dir / npz_name
    json_path = out_dir / json_name
    np.savez_compressed(npz_path, dem=bundle["dem"],
                        flood_depth=bundle["flood_depth"],
                        sdf=bundle["sdf"], shadow=bundle["shadow"])
    json_path.write_text(json.dumps(side, indent=1), encoding="utf-8")
    return npz_path, json_path


def _notebook_helpers():
    """Exec mutable + scoring defs into a namespace (no bottom driver)."""
    ns = {"__name__": "aether_nb_helpers"}
    exec(MUTABLE_CODE, ns)
    scoring_defs = SCORING_CODE.split("candidate = solve_candidate")[0]
    exec(scoring_defs, ns)
    return ns


def evolve_weights(bundle, task, generations=8, seed=0):
    """Offline random search over solver params on the train slice.

    Keeps an incumbent so the returned score never drops below Gen0.
    Mirrors the cloud search in miniature for local checks; it never
    bakes in place positions.
    """
    ns = _notebook_helpers()
    solve = ns["solve_candidate"]
    evaluate = ns["evaluate"]
    baseline = dict(ns["BASELINE_PARAMS"])
    keys = ["ground_w_slope", "drone_w_shadow", "drone_w_alt",
            "drone_w_clear"]

    def _score(params):
        cand = solve(bundle, task, params=params)
        s, _ = evaluate(cand, bundle, task)
        return float(s)

    best_params = dict(baseline)
    best_score = _score(best_params)
    history = [best_score]
    rng = random.Random(int(seed))
    for _ in range(max(int(generations) - 1, 0)):
        cand_params = dict(best_params)
        for k in keys:
            cand_params[k] = float(min(max(cand_params[k] * rng.uniform(0.6, 1.6), 0.0), 5.0))
        s = _score(cand_params)
        if s > best_score + 1e-9:
            best_score, best_params = s, cand_params
        history.append(best_score)
    return best_params, best_score, history


def main():
    nb_path = save_notebook()
    data_dir = _HERE / "data"
    tr_npz, _ = save_staged_slice(data_dir, TRAIN_NPZ_NAME, TRAIN_JSON_NAME, TRAIN_SEED)
    ho_npz, _ = save_staged_slice(data_dir, HOLDOUT_NPZ_NAME, HOLDOUT_JSON_NAME, HOLDOUT_SEED)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    print("notebook:", nb_path)
    print("cells:", len(nb["cells"]), "nbformat:", nb["nbformat"])
    print("train npz:", tr_npz)
    print("holdout npz:", ho_npz)
    bundle, side = build_staged_bundle(seed=TRAIN_SEED)
    task = side["task"]
    params, score, hist = evolve_weights(bundle, task, generations=5, seed=0)
    print("local preview evolve:", len(hist), "steps, best:", round(score, 3))
    print("Gen0 vs best delta:", round(score - hist[0], 3))


if __name__ == "__main__":
    main()

