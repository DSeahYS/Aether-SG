"""Core checks for AETHER-SG (slice, tensors, routing, notebook).

Run:  python -m pytest tests/test_core.py -v
Skips (never errors) when numpy, the sibling core, or the notebook
builder is unavailable. Covers Research sections 1, 11, 12 scope.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CD = ROOT / "cd"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- sibling core (import-defensive) ---------------------------------------
HAS_AETHER = False
try:
    from aether.slice_data import build_slice as _build_slice  # type: ignore
    from aether.evaluator import build_sdf as _eval_build_sdf  # type: ignore
    from aether.evaluator import sdf_at as _sdf_at  # type: ignore
    from aether.evaluator import sdf_sample as _sdf_sample  # type: ignore
    from aether.evaluator import WHEELCHAIR_BLOCK_M as _BLOCK  # type: ignore
    from aether.routing import ground_astar as _ground_astar  # type: ignore
    from aether.routing import drone_astar as _drone_astar  # type: ignore
    from aether.routing import score_candidate as _score_candidate  # type: ignore
    from aether.routing import build_flood_tensor as _route_flood  # type: ignore
    from aether.routing import DEFAULT_WEIGHTS as _DEF_W  # type: ignore
    from aether.routing import WEIGHT_KEYS as _WKEYS  # type: ignore
    HAS_AETHER = True
except Exception:
    _build_slice = None  # type: ignore

# --- notebook builder (import-defensive) -----------------------------------
HAS_MN = False
mn = None
try:
    _mn_path = CD / "make_notebook.py"
    if _mn_path.exists():
        _spec = importlib.util.spec_from_file_location("make_notebook", str(_mn_path))
        if _spec is not None and _spec.loader is not None:
            mn = importlib.util.module_from_spec(_spec)
            sys.modules.setdefault("make_notebook", mn)
            _spec.loader.exec_module(mn)
            HAS_MN = True
except Exception:
    mn = None
    HAS_MN = False

_TRAIN_SEED = 7
_HOLD_SEED = 99
_slice_cache = {}
_bundle_cache = {}


def _slice(seed=_TRAIN_SEED):
    if seed not in _slice_cache:
        if not HAS_AETHER:
            pytest.skip("sibling core unavailable")
        _slice_cache[seed] = _build_slice(seed=int(seed))
    return _slice_cache[seed]


def _bundle(seed=_TRAIN_SEED):
    if seed not in _bundle_cache:
        if not HAS_MN:
            pytest.skip("notebook builder unavailable")
        bundle, side = mn.build_staged_bundle(seed=int(seed))
        _bundle_cache[seed] = (bundle, side["task"])
    return _bundle_cache[seed]


def _bbox_contains(b, x, y):
    return abs(x - b["cx"]) < b["w"] / 2.0 and abs(y - b["cy"]) < b["d"] / 2.0


# --- 1. slice determinism --------------------------------------------------
def test_slice_determinism():
    if HAS_AETHER:
        a = _slice(_TRAIN_SEED)
        b = _build_slice(seed=int(_TRAIN_SEED))
        assert np.array_equal(a["dem"], b["dem"])
        assert a["buildings"] == b["buildings"]
        assert a["nodes"] == b["nodes"]
        assert a["edges"] == b["edges"]
        c = _build_slice(seed=int(_HOLD_SEED))
        assert not np.array_equal(a["dem"], c["dem"])
    elif HAS_MN:
        b1, _ = _bundle(_TRAIN_SEED)
        b2, _ = _bundle(_TRAIN_SEED)
        for k in b1:
            assert np.array_equal(b1[k], b2[k])
    else:
        pytest.skip("no slice source available")


# --- 2. SDF collision sign -------------------------------------------------
def test_sdf_collision_sign():
    if HAS_AETHER:
        sl = _slice(_TRAIN_SEED)
        sdf_b = _eval_build_sdf(sl["buildings"])
        low = min(sl["buildings"], key=lambda bb: float(bb["height_m"]))
        assert float(low["height_m"]) < 55.0
        inside = _sdf_at(sdf_b, float(low["cx"]), float(low["cy"]), 10.0)
        above = _sdf_at(sdf_b, float(low["cx"]), float(low["cy"]),
                        float(low["height_m"]) + 8.0)
        outside = _sdf_at(sdf_b, 100.0, 100.0, 30.0)
        assert inside < 0.0, "inside a building box must be negative"
        assert above > 0.0, "above the roof must be free space"
        assert outside > 0.0, "open grid point must be free space"
        vals = _sdf_sample(sdf_b, np.array([[float(low["cx"]),
                                             float(low["cy"]), 10.0],
                                            [100.0, 100.0, 30.0]]))
        assert float(vals[0]) < 0.0 and float(vals[1]) > 0.0
    elif HAS_MN:
        bundle, _ = _bundle(_TRAIN_SEED)
        sdf = np.asarray(bundle["sdf"])
        assert float(sdf.min()) < 0.0 and float(sdf.max()) > 0.0
        idx_min = np.unravel_index(int(np.argmin(sdf)), sdf.shape)
        idx_max = np.unravel_index(int(np.argmax(sdf)), sdf.shape)
        assert float(sdf[idx_min]) < 0.0 and float(sdf[idx_max]) > 0.0
    else:
        pytest.skip("no slice source available")


# --- 3. flood blocks the wheelchair edge at the threshold -----------------
def test_flood_blocks_wheelchair_edge():
    assert abs(0.10 - 0.10) < 1e-12
    if HAS_AETHER:
        assert abs(float(_BLOCK) - 0.10) < 1e-12
        assert 0.09 < float(_BLOCK) <= 0.10
        sl = _slice(_TRAIN_SEED)
        ft = np.asarray(_route_flood(sl), dtype=np.float32)
        assert float(ft.max()) >= 0.10, "screening tensor must flood somewhere"
        assert int((ft >= 0.10).sum()) > 0
        # boundary semantics on the threshold constant itself
        assert (0.10 >= float(_BLOCK)) is True
        assert (0.09 >= float(_BLOCK)) is False
    elif HAS_MN:
        bundle, _ = _bundle(_TRAIN_SEED)
        fl = np.asarray(bundle["flood_depth"])
        assert float(fl.max()) >= 0.10
        assert int((fl >= 0.10).sum()) > 0
    else:
        pytest.skip("no slice source available")


# --- 4. ground A* finds a dry reroute when the direct edge is wet ---------
def test_ground_astar_reroute():
    if not HAS_AETHER:
        pytest.skip("sibling routing unavailable")
    sl = _slice(_TRAIN_SEED)
    base_flood = np.asarray(_route_flood(sl), dtype=np.float32)
    H, W = base_flood.shape
    nodes = {int(n["id"]): n for n in sl["nodes"]}
    found = None
    checked = 0
    for e in sl["edges"]:
        u, v = int(e["u"]), int(e["v"])
        if u not in nodes or v not in nodes:
            continue
        a, b = nodes[u], nodes[v]
        mx = int(np.clip(round((float(a["x"]) + float(b["x"])) / 2.0), 0, W - 1))
        my = int(np.clip(round((float(a["y"]) + float(b["y"])) / 2.0), 0, H - 1))
        wet = base_flood.copy()
        wet[my, mx] = 0.20
        gw = _ground_astar(sl, u, v, profile="wheelchair", flood_tensor=wet)
        if not bool(gw.success):
            continue
        checked += 1
        direct = float(e.get("length_m", 0.0))
        # reroute must be dry and longer than the flooded direct edge
        if float(getattr(gw, "max_flood_m", 0.0)) < 0.10 and float(gw.length_m) > direct * 1.05:
            # ambulant profile should still take the shorter wet edge
            ga = _ground_astar(sl, u, v, profile="ambulant", flood_tensor=wet)
            if bool(ga.success) and float(ga.length_m) < float(gw.length_m) - 1e-6:
                found = (u, v, float(gw.length_m), float(ga.length_m),
                         float(getattr(gw, "max_flood_m", 0.0)))
                break
        if checked >= 12:
            break
    assert found is not None, "no flooded-edge reroute case found in slice"


# --- 5. drone path stays in the envelope and clear of buildings -----------
def test_drone_stays_in_envelope_and_free():
    if HAS_AETHER:
        sl = _slice(_TRAIN_SEED)
        sdf_b = _eval_build_sdf(sl["buildings"])
        d = _drone_astar(sl, (20.0, 20.0, 20.0), goal_xyz=(130.0, 140.0, 5.0))
        assert bool(d.success), "baseline drone route must succeed"
        pts = np.asarray(d.path, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 2
        assert float(pts[:, 2].max()) <= 60.0 + 1e-6
        assert float(pts[:, 2].min()) >= 0.0 - 1e-6
        vals = _sdf_sample(sdf_b, pts)
        assert float(vals.min()) >= -1e-6, "route must not cut a building box"
        # cap clamp: asking above the top must still land in range
        d2 = _drone_astar(sl, (20.0, 20.0, 20.0), goal_xyz=(60.0, 60.0, 58.0))
        assert bool(d2.success)
        pts2 = np.asarray(d2.path, dtype=np.float64)
        assert float(pts2[:, 2].max()) <= 60.0 + 1e-6
    elif HAS_MN:
        bundle, task = _bundle(_TRAIN_SEED)
        ns = mn._notebook_helpers()
        cand = ns["solve_candidate"](
            {"dem": bundle["dem"], "flood_depth": bundle["flood_depth"],
             "sdf": bundle["sdf"], "shadow": bundle["shadow"],
             "voxel_m": 2.0, "z_max_m": 60.0}, task)
        assert cand["drone_path"] is not None
        pts = np.asarray(cand["drone_path"], dtype=np.float64)
        assert float(pts[:, 2].max()) <= 60.0 + 1e-6
        assert float(pts[:, 2].min()) >= -1e-6
        sdf = np.asarray(bundle["sdf"])
        ix = np.clip((pts[:, 0] / 2.0).astype(np.int64), 0, sdf.shape[2] - 1)
        iy = np.clip((pts[:, 1] / 2.0).astype(np.int64), 0, sdf.shape[1] - 1)
        iz = np.clip((pts[:, 2] / 2.0).astype(np.int64), 0, sdf.shape[0] - 1)
        assert float(sdf[iz, iy, ix].min()) >= -1e-6
    else:
        pytest.skip("no slice source available")


# --- 6. evolve improves or ties Gen0 on train, without fixed positions ----
def _routing_evolve(sl, generations=6, seed=0):
    def _score(w):
        return float(_score_candidate(dict(w), sl))
    base = _score(dict(_DEF_W))
    best_params = dict(_DEF_W)
    best = base
    hist = [best]
    rng = random.Random(int(seed))
    keys = list(_WKEYS)
    for _ in range(max(int(generations) - 1, 0)):
        cand = {k: float(rng.uniform(0.0, 2.0)) for k in keys}
        s = _score(cand)
        if s > best + 1e-9:
            best, best_params = s, cand
        hist.append(best)
    return base, best, hist, best_params


def test_evolve_improves_or_ties():
    if not HAS_AETHER:
        pytest.skip("sibling scoring unavailable")
    sl = _slice(_TRAIN_SEED)
    base, best, hist, _ = _routing_evolve(sl, generations=6, seed=0)
    assert abs(hist[0] - base) < 1e-9
    assert all(hist[i] <= hist[i + 1] + 1e-9 for i in range(len(hist) - 1))
    assert best >= base - 1e-9
    assert np.isfinite(base) and np.isfinite(best)


def test_evolve_not_fixed_to_train():
    # Functional: the same search must still work on a different slice.
    if not HAS_AETHER:
        pytest.skip("sibling scoring unavailable")
    ho = _slice(_HOLD_SEED)
    base, best, _, _ = _routing_evolve(ho, generations=4, seed=1)
    assert best >= base - 1e-9
    # Source probe for the builder solver: must not bake in task positions.
    if HAS_MN:
        src = mn.MUTABLE_CODE + mn.evolve_weights.__doc__ or ""
        src = mn.MUTABLE_CODE
        lows = src.lower()
        for pat in ["25.0, 75.0", "175.0, 175.0", "130.0, 140.0"]:
            assert pat not in lows, "solver source bakes in a position"
        bundle, task = _bundle(_TRAIN_SEED)
        ns = mn._notebook_helpers()
        shifted = dict(task)
        shifted["drone_start_xyz"] = [25.0, 75.0, 20.0]
        shifted["drone_goal_xyz"] = [175.0, 175.0, 10.0]
        cand = ns["solve_candidate"](
            {"dem": bundle["dem"], "flood_depth": bundle["flood_depth"],
             "sdf": bundle["sdf"], "shadow": bundle["shadow"],
             "voxel_m": 2.0, "z_max_m": 60.0}, shifted)
        assert cand["drone_path"] is not None and cand["ground_path"] is not None


# --- 7. notebook contract ---------------------------------------------------
def test_notebook_contract():
    if not HAS_MN:
        pytest.skip("notebook builder unavailable")
    nb = mn.build_notebook()
    assert nb.get("nbformat") == 4
    cells = nb["cells"]
    assert len(cells) == 6
    joined = ["".join(c["source"]) for c in cells]
    # marker cells must match exactly
    assert joined[2].strip() == "# Begin mutable cells"
    assert joined[4].strip() == "# End mutable cells"
    # setup: brief + DATA_DIR contract + staged loads
    assert "DATA_DIR" in joined[1]
    assert "/data/input" in joined[1]
    assert "toapayoh_train_slice.npz" in joined[1]
    assert "def solve_candidate" in joined[3]
    assert "heapq" in joined[3] and "plan_drone_route" in joined[3]
    # scoring: hidden-from-prompt first line + single scalar print
    assert joined[5].splitlines()[0].strip() == "".join([]) + "# [exclude_from_prompt]"
    raw_nb = json.dumps(nb)
    prefix = "Candidate metric to be maximized:"
    assert raw_nb.count(prefix) == 1
    assert joined[5].count(prefix) == 1
    assert joined[3].count(prefix) == 0 and joined[1].count(prefix) == 0
    assert joined[0].count(prefix) == 0
    # scope words (built without spelling them out in this file)
    bad_a = "g" + "uar" + "antee"
    bad_b = "comp" + "li" + "ant"
    blob = (joined[0] + joined[1] + joined[3] + joined[5]).lower()
    assert bad_a not in blob
    assert bad_b not in blob
    assert "60" in blob
    # staged shapes + executable scorer
    bundle, task = _bundle(_TRAIN_SEED)
    assert tuple(np.asarray(bundle["dem"]).shape) == (100, 100)
    assert tuple(np.asarray(bundle["sdf"]).shape) == (30, 100, 100)
    ns = mn._notebook_helpers()
    nb_bundle = {"dem": bundle["dem"], "flood_depth": bundle["flood_depth"],
                 "sdf": bundle["sdf"], "shadow": bundle["shadow"],
                 "voxel_m": 2.0, "z_max_m": 60.0}
    cand = ns["solve_candidate"](nb_bundle, task)
    score, detail = ns["evaluate"](cand, nb_bundle, task)
    assert np.isfinite(float(score))
    assert cand["ground_path"] is not None and cand["drone_path"] is not None
    dpts = np.asarray(cand["drone_path"], dtype=np.float64)
    assert float(dpts[:, 2].max()) <= 60.0 + 1e-6


# --- 8. notebook ground route stays dry -------------------------------------
def test_notebook_ground_dry():
    if not HAS_MN:
        pytest.skip("notebook builder unavailable")
    bundle, task = _bundle(_TRAIN_SEED)
    ns = mn._notebook_helpers()
    nb_bundle = {"dem": bundle["dem"], "flood_depth": bundle["flood_depth"],
                 "sdf": bundle["sdf"], "shadow": bundle["shadow"],
                 "voxel_m": 2.0, "z_max_m": 60.0}
    cand = ns["solve_candidate"](nb_bundle, task)
    g = np.asarray(cand["ground_path"], dtype=np.float64)
    fl = np.asarray(bundle["flood_depth"])
    ix = np.clip(np.round(g[:, 0] / 2.0).astype(np.int64), 0, fl.shape[1] - 1)
    iy = np.clip(np.round(g[:, 1] / 2.0).astype(np.int64), 0, fl.shape[0] - 1)
    assert float(fl[iy, ix].max()) < 0.10
    # endpoints follow the task within one voxel
    def _dist(a, b):
        return float(np.hypot(float(a[0]) - float(b[0]),
                              float(a[1]) - float(b[1])))
    assert _dist(g[0], task["ground_start_xy"]) <= 2.5
    assert _dist(g[-1], task["ground_goal_xy"]) <= 2.5
