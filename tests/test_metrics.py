"""Unit tests for the AETHER-SG metrics module (backend slice coords 0..200).

Covers: line-of-sight (clear vs blocked + ``blocked_by``), swath radius
(``alt * tan(30deg)``), coverage monotonicity in [0, 100], flight metrics
(length / time / climb) on a known 3-point path, and environment-flood
blocked-cell counting on a tiny DEM.

Import-defensive: the ``metrics`` implementation is produced by a sibling
agent and may not exist yet.  Discovery probes several plausible module
locations and function names; every test skips with an explicit reason when
its capability is unavailable instead of erroring.  The environment-flood
test additionally falls back to ``evaluator.flood_tensor`` (the existing
bathtub+connectivity implementation of the same contract) so that part of
the suite verifies real behaviour even before ``metrics`` lands.

Coordinate contract (backend slice, as in ``slice_data`` / ``evaluator``):
metres, x east / y north / z up AGL, extent 0..200.  Building boxes are
``{"cx", "cy", "w", "d", "height_m"}``.  (The viewer shows the same data
scene-centred; viewer value = backend value - 100.)
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (SRC, ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Import-defensive discovery of the metrics implementation.
# ---------------------------------------------------------------------------
_METRICS_FILE_CANDIDATES = (
    SRC / "aether" / "metrics.py",
    SRC / "metrics.py",
    ROOT / "metrics.py",
    ROOT / "scripts" / "metrics.py",
)


def _load_metrics():
    """Return (module, source-description) or (None, reason-tried)."""
    for name in ("aether.metrics", "metrics"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if mod is not None:
            return mod, f"import:{name}"
    for path in _METRICS_FILE_CANDIDATES:
        if not path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                "aether_metrics_probe", str(path))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("aether_metrics_probe", mod)
            spec.loader.exec_module(mod)
            return mod, f"file:{path.name}"
        except Exception:
            continue
    tried = ", ".join(str(p) for p in _METRICS_FILE_CANDIDATES)
    return None, f"metrics module not found (tried: {tried})"


METRICS_MOD, METRICS_SRC = _load_metrics()

LO_NAMES = ("los_swath_frames", "line_of_sight", "los", "los_check",
            "check_los", "segment_blocked", "occlusion_check", "has_los",
            "ray_blocked")
SWATH_NAMES = ("los_swath_frames", "swath_radius", "swath",
               "footprint_radius", "camera_footprint_radius", "footprint",
               "coverage_radius")
COV_NAMES = ("los_swath_frames", "coverage", "coverage_percent",
             "coverage_pct", "covered_fraction", "map_coverage",
             "route_coverage")
FLIGHT_NAMES = ("flight_metrics", "path_metrics", "flight_stats",
                "drone_metrics", "route_metrics", "path_stats")
FLOOD_NAMES = ("environment_metrics", "environment_flood", "flood_cells",
               "flood_blocked_cells", "blocked_cells", "environment",
               "flood_summary", "flood_blocked")


def _looks_like_los_swath(fn) -> bool:
    """True when ``fn`` is the los_swath_frames-style frames API."""
    if getattr(fn, "__name__", "") == "los_swath_frames":
        return True
    try:
        names = [p.name.lower() for p in inspect.signature(fn).parameters
                 .values()]
    except (TypeError, ValueError):
        return False
    return (names and ("target" in names[0] or "path" in names[0])) \
        and any("target" in n for n in names)


def _find_fn(mod, names):
    """First callable attribute among candidate names, else (None, None)."""
    if mod is None:
        return None, None
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return name, fn
    return None, None


def _get_los_fn():
    name, fn = _find_fn(METRICS_MOD, LO_NAMES)
    if fn is None:
        pytest.skip(f"no line-of-sight function among {list(LO_NAMES)} "
                    f"(module: {METRICS_SRC})")
    return name, fn


def _get_swath_fn():
    name, fn = _find_fn(METRICS_MOD, SWATH_NAMES)
    if fn is None:
        pytest.skip(f"no swath function among {list(SWATH_NAMES)} "
                    f"(module: {METRICS_SRC})")
    return name, fn


def _get_coverage_fn():
    name, fn = _find_fn(METRICS_MOD, COV_NAMES)
    if fn is None:
        pytest.skip(f"no coverage function among {list(COV_NAMES)} "
                    f"(module: {METRICS_SRC})")
    return name, fn


def _get_flight_fn():
    name, fn = _find_fn(METRICS_MOD, FLIGHT_NAMES)
    if fn is None:
        pytest.skip(f"no flight-metrics function among {list(FLIGHT_NAMES)} "
                    f"(module: {METRICS_SRC})")
    return name, fn


# ---------------------------------------------------------------------------
# Hand-built boxes / points (backend slice frame, x east / y north / z AGL).
# ---------------------------------------------------------------------------
def _box(id_, cx, cy, w, d, h):
    return {"id": id_, "cx": float(cx), "cy": float(cy), "w": float(w),
            "d": float(d), "height_m": float(h)}


# One box straddling the x=100 / y=100 lines, x 90..110, y 90..110, z 0..30.
BOX_A = _box(0, 100, 100, 20, 20, 30)
# A second box far north of the test segments (must never block them).
BOX_B = _box(1, 100, 170, 20, 16, 45)


def _pick(mapping, *keys):
    """First present value among alias keys, else None."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _norm_los(result):
    """Normalise any plausible LOS return to (clear: bool, blocked_by).

    Accepts bool / (bool, blocker) / dict with clear-or-blocked style keys,
    and objects with ``blocked`` / ``blocked_by`` attributes.
    """
    if isinstance(result, bool):
        return result, None
    if isinstance(result, (tuple, list)):
        if not result:
            raise AssertionError(f"empty LOS result: {result!r}")
        head = result[0]
        if isinstance(head, bool):
            return head, result[1] if len(result) > 1 else None
        # (blocked, blocker) ordering.
        return (not bool(head)), result[1] if len(result) > 1 else None
    if isinstance(result, dict):
        clear = _pick(result, "clear", "is_clear", "visible", "has_los",
                      "ok", "unblocked")
        blocked_by = _pick(result, "blocked_by", "blocker", "blocking",
                           "obstacle", "obstructed_by")
        if clear is not None:
            return bool(clear), blocked_by
        blocked = _pick(result, "blocked", "occluded", "is_blocked")
        if blocked is not None:
            return (not bool(blocked)), blocked_by
        raise AssertionError(f"LOS dict has no clear/blocked keys: {result!r}")
    # Object-style result.
    clear = getattr(result, "clear", getattr(result, "visible", None))
    blocked_by = getattr(result, "blocked_by",
                         getattr(result, "blocker", None))
    if clear is not None:
        return bool(clear), blocked_by
    blocked = getattr(result, "blocked", None)
    if blocked is not None:
        return (not bool(blocked)), blocked_by
    raise AssertionError(f"unrecognised LOS result: {result!r}")


def _call_los(fn, drone, target, boxes):
    """Call LOS with the most plausible positional/keyword conventions.

    Preference order follows the documented backend convention
    ``los(drone, target, buildings)``.  Parameter names (if they hint at
    box/drone arguments) reorder the positional attempts.  A TypeError
    raised inside the implementation itself is re-raised, not masked.
    """
    import re

    if _looks_like_los_swath(fn):
        # los_swath_frames(path, buildings, target_xy, dem): the path is
        # the drone sequence (duplicated here), the target height comes
        # from the DEM (None -> ground 0).  Row 0 is the drone frame.
        tgt_xy = (float(target[0]), float(target[1]))
        rows = fn([list(drone), list(drone)], boxes, tgt_xy, None)
        row = rows[0] if isinstance(rows, (list, tuple)) else rows
        clear = bool(row["los_clear"]) if isinstance(row, dict) \
            else bool(row[0])
        blocked_by = (row["los_blocked_by"] if isinstance(row, dict)
                      else (row[1] if len(row) > 1 else None))
        return clear, blocked_by

    def _looks_like_sig_mismatch(exc: TypeError) -> bool:
        msg = str(exc)
        return bool(re.search(
            r"positional argument|missing .* argument|unexpected keyword|"
            r"takes from \d+ to \d+|takes \d+ positional", msg))

    def _param_hints():
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return []
        return [p.name.lower() for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY,
                              p.POSITIONAL_OR_KEYWORD)]

    names = _param_hints()
    box_words = ("box", "build", "obst", "occlu", "shadow", "solid")
    boxes_first = any(w in (names[0] if names else "") for w in box_words)
    keyword_names = [n for n in names
                     if any(w in n for w in box_words)]
    variants = []
    if boxes_first:
        variants.append(lambda: fn(boxes, drone, target))
    variants.append(lambda: fn(drone, target, boxes))
    if not boxes_first:
        variants.append(lambda: fn(boxes, drone, target))
    for kw in keyword_names:
        variants.append(
            (lambda kw=kw: fn(drone, target, **{kw: boxes})))
    for kw in ("buildings", "boxes", "obstacles"):
        if kw not in keyword_names:
            variants.append(
                (lambda kw=kw: fn(drone, target, **{kw: boxes})))
    last_err: Exception | None = None
    for call in variants:
        try:
            return _norm_los(call())
        except TypeError as exc:
            if not _looks_like_sig_mismatch(exc):
                raise
            last_err = exc
            continue
    raise AssertionError(
        f"LOS callable {fn.__name__} rejected all call shapes: {last_err}")


def _assert_blocked_by(blocked_by, boxes):
    assert blocked_by is not None, "blocked LOS must set blocked_by"
    ids = {b.get("id") for b in boxes}
    if isinstance(blocked_by, bool):
        pytest.fail("blocked_by must name the blocking box, not a bool")
    if isinstance(blocked_by, int):
        assert 0 <= blocked_by < len(boxes), (
            f"blocked_by index {blocked_by} out of range")
        assert blocked_by == 0, (
            f"expected blocker index 0, got {blocked_by}")
        return
    if isinstance(blocked_by, dict):
        bid = blocked_by.get("id")
        assert bid == 0, f"expected blocker id 0, got {bid}"
        return
    # Name / id scalar (or a list containing one).
    vals = blocked_by if isinstance(blocked_by, (list, tuple)) else [blocked_by]
    flat = [v.get("id") if isinstance(v, dict) else v for v in
            (vals if not isinstance(vals, (str, bytes)) else [vals])]
    assert 0 in flat or "0" in [str(v) for v in flat], (
        f"expected blocker to name box 0, got {blocked_by!r}")
    assert not (set(ids) - {0} == set() and 0 not in flat)


# ---------------------------------------------------------------------------
# Line of sight.
# ---------------------------------------------------------------------------
def test_los_clear_when_no_box_between():
    _, fn = _get_los_fn()
    boxes = [BOX_A, BOX_B]
    # Segment runs east at y=150: passes 40 m north of BOX_A (y 90..110).
    clear, _ = _call_los(fn, (20.0, 150.0, 40.0), (180.0, 150.0, 0.0), boxes)
    assert clear, "LOS must be clear when no box crosses the segment"
    # Same geometry but identifiably below the roofline along a clear lane.
    clear2, _ = _call_los(fn, (20.0, 150.0, 10.0), (180.0, 150.0, 0.0), boxes)
    assert clear2, "low segment on a clear lane must also be clear"
    # Segment ends before the box: the box beyond the target cannot block.
    clear3, _ = _call_los(fn, (20.0, 100.0, 10.0), (80.0, 100.0, 0.0), boxes)
    assert clear3, "a box beyond the target must not report blocked"
    # No boxes at all: always clear.
    clear4, _ = _call_los(fn, (10.0, 10.0, 10.0), (190.0, 190.0, 0.0), [])
    assert clear4, "empty box list must report clear"


def test_los_blocked_and_blocked_by_when_box_crosses():
    _, fn = _get_los_fn()
    boxes = [BOX_A, BOX_B]
    # Segment along y=100 from z=10 down to the ground target passes
    # through BOX_A (x 90..110, y 90..110, z 0..30): blocked, blocker
    # must be BOX_A.
    clear, blocked_by = _call_los(
        fn, (20.0, 100.0, 10.0), (180.0, 100.0, 0.0), boxes)
    assert not clear, "LOS must report blocked when a box crosses the segment"
    _assert_blocked_by(blocked_by, boxes)
    # Drone directly above the roof (z=40 > height 30), descending to a
    # ground target further east: the pitched segment stays over the
    # roof while crossing x 90..110 (z 32..40 there) and must clear.
    clear2, _ = _call_los(
        fn, (100.0, 100.0, 40.0), (150.0, 100.0, 0.0), boxes)
    assert clear2, "segment passing over the roof must be clear"
    # Same start, but the target is too close: the segment clips the
    # east face (z=20 at x=110) and must report blocked.
    clear2b, blocked2b = _call_los(
        fn, (100.0, 100.0, 40.0), (120.0, 100.0, 0.0), boxes)
    assert not clear2b, (
        "pitched segment clipping the east face must report blocked")
    _assert_blocked_by(blocked2b, boxes)
    # Low drone above the roof centre but aimed back through the box
    # core: the segment descends into the box and must report blocked.
    clear3, blocked_by3 = _call_los(
        fn, (100.0, 100.0, 8.0), (180.0, 100.0, 0.0), boxes)
    assert not clear3, (
        "descending segment into the box core must report blocked")
    _assert_blocked_by(blocked_by3, boxes)


# ---------------------------------------------------------------------------
# Swath radius: alt * tan(30deg).
# ---------------------------------------------------------------------------
def _norm_radius(result):
    val = result
    if isinstance(result, dict):
        val = _pick(result, "radius", "swath_radius", "radius_m", "r")
        if val is None:
            nums = [v for v in result.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(nums) == 1:
                val = nums[0]
    if isinstance(val, (tuple, list)) and len(val) == 1:
        val = val[0]
    assert isinstance(val, (int, float)) and not isinstance(val, bool), (
        f"swath result is not numeric: {result!r}")
    return float(val)


def _call_swath(fn, alt):
    if _looks_like_los_swath(fn):
        # los_swath_frames(path=[(x,y,alt),(x,y,alt)], boxes=[], target,
        # dem=None): default fov_deg=60 -> radius = alt * tan(30deg)
        # exactly as the contract states.
        rows = fn([[170.0, 170.0, float(alt)], [170.0, 170.0, float(alt)]],
                  [], (0.0, 0.0), None)
        row = rows[0] if isinstance(rows, (list, tuple)) else rows
        assert isinstance(row, dict) and "swath_radius_m" in row, (
            f"los_swath frame has no swath_radius_m: {row!r}")
        return float(row["swath_radius_m"])
    try:
        return _norm_radius(fn(float(alt)))
    except TypeError:
        # Two-argument forms: (alt, half_fov_deg=30) or (alt, fov=60).
        for extra in (30.0, 60.0):
            try:
                return _norm_radius(fn(float(alt), extra))
            except TypeError:
                continue
        raise


def test_swath_radius_matches_alt_tan30():
    _, fn = _get_swath_fn()
    tan30 = math.tan(math.radians(30.0))  # ~0.57735026919
    for alt in (10.0, 30.0, 60.0):
        got = _call_swath(fn, alt)
        want = alt * tan30
        assert math.isclose(got, want, rel_tol=1e-3, abs_tol=1e-6), (
            f"swath radius at alt {alt}: got {got!r}, want {want!r} "
            f"(alt*tan(30deg))")
    got0 = _call_swath(fn, 0.0)
    assert math.isclose(got0, 0.0, abs_tol=1e-9), (
        f"swath at alt 0 must be 0, got {got0!r}")


# ---------------------------------------------------------------------------
# Coverage: monotonically non-decreasing, bounded to [0, 100].
# ---------------------------------------------------------------------------
def _norm_coverage(result):
    """Extract the raw numeric coverage value (percent or fraction)."""
    val = result
    if isinstance(result, dict):
        val = _pick(result, "coverage", "coverage_percent", "coverage_pct",
                    "percent", "pct", "fraction", "value")
        if val is None:
            vals = [v for v in result.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(vals) == 1:
                val = vals[0]
    if isinstance(val, (tuple, list)) and len(val) == 1:
        val = val[0]
    assert isinstance(val, (int, float)) and not isinstance(val, bool), (
        f"coverage result is not numeric: {result!r}")
    return float(val)


def _call_coverage(fn, path, boxes):
    """Call coverage for one path; returns the final cumulative %."""
    if _looks_like_los_swath(fn):
        # los_swath_frames: the last frame's coverage_pct is the union
        # over the whole path prefix (needs >= 2 points: _as_xyz rule).
        if len(path) < 2:
            raise AssertionError("los_swath coverage needs a >= 2-point "
                                 "path prefix")
        rows = fn([[float(p[0]), float(p[1]), float(p[2])] for p in
                   path], boxes, (0.0, 0.0), None)
        last = rows[-1] if isinstance(rows, (list, tuple)) else rows
        assert isinstance(last, dict) and "coverage_pct" in last, (
            f"frames have no coverage_pct: {last!r}")
        return _norm_coverage(last["coverage_pct"])
    grid = [(x, y, 0.0) for x in (10.0, 50.0, 100.0, 150.0, 190.0)
            for y in (10.0, 50.0, 100.0, 150.0, 190.0)]
    variants = (
        lambda: fn(path),
        lambda: fn(path, boxes),
        lambda: fn(path, boxes, grid),
        lambda: fn(path, grid),
        lambda: fn(path, buildings=boxes),
    )
    last_err: Exception | None = None
    for call in variants:
        try:
            return _norm_coverage(call())
        except TypeError as exc:
            last_err = exc
            continue
    raise AssertionError(
        f"coverage callable {fn.__name__} rejected all call shapes: {last_err}")


def test_coverage_monotonic_and_bounded():
    _, fn = _get_coverage_fn()
    boxes = [BOX_A, BOX_B]
    # Sweep a path eastward across the slice, one prefix at a time.
    # Prefixes start at 2 points (3-D path coercion elsewhere requires
    # >= 2 points; monotonicity over 2..5 prefixes still pins the law).
    pts = [(10.0, 100.0, 30.0), (50.0, 100.0, 30.0), (100.0, 100.0, 30.0),
           (150.0, 100.0, 30.0), (190.0, 100.0, 30.0)]
    raws = []
    for k in range(2, len(pts) + 1):
        raws.append(_call_coverage(fn, [list(p) for p in pts[:k]], boxes))
    # Fraction form (0..1) vs percent form (0..100) is decided globally so
    # the rescale cannot distort monotonicity across mixed magnitudes.
    if raws and max(raws) <= 1.0 + 1e-12:
        values = [v * 100.0 for v in raws]
    else:
        values = raws
    for k, pct in enumerate(values, start=1):
        assert 0.0 - 1e-9 <= pct <= 100.0 + 1e-9, (
            f"coverage {pct} outside [0, 100] for prefix of {k} points")
    for prev, cur in zip(values, values[1:]):
        assert cur >= prev - 1e-9, (
            f"coverage must be monotonically non-decreasing, "
            f"got {values}")
    assert values[-1] > 0.0, "a full sweep must cover something"
    assert values[-1] >= values[0], "coverage cannot shrink as path grows"


# ---------------------------------------------------------------------------
# Flight metrics: length / time / climb on a known 3-point path.
# ---------------------------------------------------------------------------
def _norm_flight(result):
    """Normalise to a dict with length_m / time_s / climb_m / max_alt_m."""
    if isinstance(result, dict):
        length = _pick(result, "length_m", "length", "path_length_m",
                       "path_length", "distance_m", "distance")
        time_ = _pick(result, "time_s", "time", "duration_s", "duration",
                      "flight_time_s", "flight_time")
        climb = _pick(result, "climb_m", "climb", "total_climb_m",
                      "alt_gain_m", "altitude_gain_m", "gain_m",
                      "ascent_m")
        max_alt = _pick(result, "max_alt_m", "max_altitude_m",
                        "altitude_max_m", "altitude_peak_m", "peak_m",
                        "max_z_m")
        assert length is not None, (
            f"flight result has no length key: {result!r}")
        return {"length_m": float(length),
                "time_s": (float(time_) if time_ is not None else None),
                "climb_m": float(climb) if climb is not None else None,
                "max_alt_m": float(max_alt)
                if max_alt is not None else None,
                "raw": result}
    if isinstance(result, (tuple, list)):
        # Positional (length, time, climb) is the conventional order.
        assert len(result) >= 2, f"flight tuple too short: {result!r}"
        return {"length_m": float(result[0]),
                "time_s": float(result[1]),
                "climb_m": float(result[2]) if len(result) > 2 else None,
                "max_alt_m": None, "raw": result}
    # Object-style result.
    length = getattr(result, "length_m", getattr(result, "length", None))
    assert length is not None, f"flight result has no length: {result!r}"
    return {"length_m": float(length),
            "time_s": getattr(result, "time_s",
                              getattr(result, "time", None)),
            "climb_m": getattr(result, "climb_m",
                               getattr(result, "climb", None)),
            "max_alt_m": getattr(result, "max_alt_m", None),
            "raw": result}


def _call_flight(fn, path, speed, boxes):
    """Call flight metrics across plausible signatures.

    Returns (metrics, speed_used) where ``speed_used`` is the speed value
    actually handed to the callable (None if it was not parameterised).
    """
    path_list = [[float(v) for v in p] for p in path]
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        params = [p.name.lower() for p in sig.parameters.values()
                  if p.name != "self"]
        sp = next((n for n in params if "speed" in n), None)
        builders_first = any(w in (params[0] if params else "")
                             for w in ("build", "box", "sector"))
        if len(params) >= 4 and not builders_first:
            # Documented rich shape:
            # (path, buildings, sdf_bundle, shadow_bundle, dem) with
            # None bundles selecting the analytic box SDF / no shadow.
            if sp is not None:
                try:
                    return _norm_flight(fn(path_list, boxes, None, None,
                                            None, **{sp: speed})), speed
                except TypeError:
                    pass
            try:
                return _norm_flight(
                    fn(path_list, boxes, None, None, None)), None
            except TypeError:
                pass
        if sp is not None and len(params) <= 2:
            return _norm_flight(fn(path_list, **{sp: speed})), speed
        if len(params) >= 2:
            try:
                return _norm_flight(fn(path_list, speed)), speed
            except TypeError:
                return _norm_flight(fn(path_list)), None
        return _norm_flight(fn(path_list)), None
    # No signature available: rich shape, path-only, then (path, speed).
    try:
        return _norm_flight(fn(path_list, boxes, None, None, None)), None
    except TypeError:
        pass
    try:
        return _norm_flight(fn(path_list)), None
    except TypeError:
        return _norm_flight(fn(path_list, speed)), speed


def test_flight_metrics_known_three_point_path():
    _, fn = _get_flight_fn()
    # Horizontal 30 m leg, then 40 m north + 20 m climb.
    path = [(0.0, 0.0, 10.0), (30.0, 0.0, 10.0), (30.0, 40.0, 30.0)]
    seg1 = math.dist(path[0], path[1])            # 30.0 exactly
    seg2 = math.dist(path[1], path[2])            # sqrt(40^2 + 20^2)
    want_length = seg1 + seg2                      # ~74.72135954999579
    want_climb = 20.0      # total altitude gain 10 -> 10 -> 30
    want_max_alt = 30.0
    speed = 10.0
    boxes = [BOX_A, BOX_B]
    got, speed_used = _call_flight(fn, path, speed, boxes)
    assert math.isclose(got["length_m"], want_length, rel_tol=1e-6), (
        f"flight length: got {got['length_m']!r}, want {want_length!r}")
    if got["climb_m"] is not None:
        assert math.isclose(got["climb_m"], want_climb, abs_tol=1e-6), (
            f"flight climb: got {got['climb_m']!r}, want {want_climb!r}")
    if got["max_alt_m"] is not None:
        assert math.isclose(got["max_alt_m"], want_max_alt, abs_tol=1e-6), (
            f"flight max altitude: got {got['max_alt_m']!r}, "
            f"want {want_max_alt!r}")
    assert got["time_s"] is not None and got["time_s"] > 0.0, (
        f"flight time must be positive, got {got['time_s']!r}")
    if speed_used is not None:
        want_time = want_length / float(speed_used)
        assert math.isclose(got["time_s"], want_time, rel_tol=1e-6), (
            f"flight time {got['time_s']!r} != length/speed "
            f"({want_time!r}) at passed speed {speed_used!r}")
    else:
        if isinstance(got["raw"], dict):
            raw_speed = _pick(got["raw"], "speed_mps", "speed",
                              "cruise_mps")
        else:
            raw_speed = None
        module_speed = getattr(METRICS_MOD, "DRONE_SPEED_MPS", None) \
            if METRICS_MOD is not None else None
        if raw_speed is None and module_speed is not None:
            raw_speed = module_speed
        if raw_speed is not None:
            assert math.isclose(got["time_s"],
                                want_length / float(raw_speed),
                                rel_tol=1e-6), (
                f"flight time {got['time_s']!r} != length/speed "
                f"({want_length / float(raw_speed)!r})")
        else:
            # No stated speed: time must at least be consistent with a
            # plausible drone speed (0.5..60 m/s) over the known length.
            implied = want_length / got["time_s"]
            assert 0.5 <= implied <= 60.0, (
                f"implied cruise speed {implied!r} m/s implausible for "
                f"length {want_length!r} and time {got['time_s']!r}")


# ---------------------------------------------------------------------------
# Environment flood: blocked-cell count on a tiny DEM.
# ---------------------------------------------------------------------------
def _tiny_dem():
    """20x5 DEM: south drain channel + one isolated low cell inland.

    Rows 0..11 form the south seed band in ``evaluator.flood_tensor``;
    the isolated cell at row 15 is surrounded by 10 m ground, so under
    bathtub+connectivity semantics it must never be wet.
    """
    dem = np.full((20, 5), 10.0, dtype=np.float64)
    dem[0, :] = 0.0     # drain channel along the south edge
    dem[15, 2] = 5.0    # disconnected depression (below a 6 m waterline)
    return dem


def _norm_flood(result):
    """Normalise to (blocked_count, wet_mask_or_None, max_depth_or_None)."""
    if isinstance(result, dict):
        wet = result.get("wet")
        meta = result.get("meta") or {}
        count = _pick(result, "blocked_cells", "wet_cells",
                      "blocked", "wet_count", "count")
        if count is None:
            count = _pick(meta, "wet_cells", "blocked_cells", "wet_count",
                          "count")
        if count is None and wet is not None:
            count = int(np.count_nonzero(np.asarray(wet, dtype=bool)))
        depth = result.get("depth")
        max_depth = _pick(meta, "max_depth_m", "max_depth")
        if max_depth is None and depth is not None:
            arr = np.asarray(depth, dtype=np.float64)
            max_depth = float(arr.max()) if arr.size else 0.0
        assert count is not None, (
            f"flood result has no cell count or wet mask: {result!r}")
        return int(count), wet, (None if max_depth is None
                                 else float(max_depth))
    if isinstance(result, (tuple, list)):
        # Candidate shapes: (depth, count), (wet, ...), (count, ...).
        for item in result:
            if isinstance(item, (int, np.integer)) and not isinstance(
                    item, bool):
                count = int(item)
                wet = None
                for other in result:
                    if isinstance(other, np.ndarray) or (
                            other is not item and hasattr(other, "__len__")
                            and not isinstance(
                                other, (str, bytes))):
                        if isinstance(other, np.ndarray):
                            wet = other
                return count, wet, None
        raise AssertionError(f"unrecognised flood tuple: {result!r}")
    if isinstance(result, (int, np.integer)) and not isinstance(
            result, bool):
        return int(result), None, None
    raise AssertionError(f"unrecognised flood result: {result!r}")


def _call_flood(fn, dem, water):
    """Call an environment-flood callable across plausible signatures."""
    dem_arr = np.asarray(dem, dtype=np.float64)
    if getattr(fn, "__name__", "") == "environment_metrics":
        # environment_metrics(slice_dict, flood_depth): flood_depth is a
        # depth tensor.  Reuse the evaluator flood screening tensor (the
        # connectivity-aware implementation) as the depth input.
        from aether import evaluator as _ev  # noqa: E402
        tensor = _ev.flood_tensor(dem_arr, float(water))
        result = fn({"dem": dem_arr, "buildings": []}, tensor["depth"])
        assert isinstance(result, dict), (
            f"environment_metrics must return a dict: {result!r}")
        blocked = _pick(result, "flood_blocked_cells", "blocked_cells",
                        "blocked", "flood_blocked")
        wet_count = _pick(result, "flood_wet_cells", "wet_cells", "wet",
                          "wet_count")
        assert blocked is not None, (
            f"environment_metrics has no blocked-cell count: {result!r}")
        return int(blocked), None, None
    variants = (
        lambda: fn(dem_arr, water),
        lambda: fn(dem_arr, water_level_m=water),
        lambda: fn(dem=dem_arr, water_level=water),
        lambda: fn(dem_arr, float(water), 1.0),
    )
    last_err: Exception | None = None
    for call in variants:
        try:
            return _norm_flood(call())
        except TypeError as exc:
            last_err = exc
            continue
    raise AssertionError(
        f"flood callable {fn.__name__} rejected all call shapes: {last_err}")


def test_environment_flood_blocked_cells_tiny_dem():
    """Blocked/wet cell count on a tiny DEM (bathtub + connectivity).

    Prefers the metrics environment API; falls back to the existing
    ``evaluator.flood_tensor`` implementation of the same contract.
    Skips only when neither is importable.
    """
    name, fn = _find_fn(METRICS_MOD, FLOOD_NAMES)
    provenance = f"metrics.{name}" if fn is not None else None
    if fn is None:
        try:
            from aether import evaluator as _ev  # noqa: E402
            fn = _ev.flood_tensor
            provenance = "evaluator.flood_tensor"
        except Exception:
            pytest.skip("no environment-flood function in metrics "
                        f"({METRICS_SRC}) and evaluator unavailable")
    dem = _tiny_dem()
    # Water at 3 m: only the 5 channel cells are below the waterline.
    count1, wet1, _ = _call_flood(fn, dem, 3.0)
    assert count1 == 5, (
        f"[{provenance}] 3 m water: expected 5 blocked cells, "
        f"got {count1}")
    # Water at 6 m: the inland depression (5 m) is also below the
    # waterline but disconnected from the south channel, so the count
    # must still be exactly 5 (connectivity, not naive bathtub).
    count2, wet2, max_depth = _call_flood(fn, dem, 6.0)
    assert count2 == 5, (
        f"[{provenance}] 6 m water: disconnected low cell must stay dry; "
        f"expected 5 blocked cells, got {count2}")
    if wet2 is not None:
        mask = np.asarray(wet2, dtype=bool)
        assert mask.shape == (20, 5), (
            f"[{provenance}] wet mask shape {mask.shape}, want (20, 5)")
        assert int(mask.sum()) == 5, (
            f"[{provenance}] wet mask sum {int(mask.sum())}, want 5")
        assert bool(mask[0, 2]), (
            f"[{provenance}] south channel cell must be wet")
        assert not bool(mask[15, 2]), (
            f"[{provenance}] isolated low cell must stay dry")
    if max_depth is not None:
        assert math.isclose(max_depth, 6.0, abs_tol=1e-6), (
            f"[{provenance}] max depth at 6 m water over a 0 m channel "
            f"must be 6 m, got {max_depth!r}")
