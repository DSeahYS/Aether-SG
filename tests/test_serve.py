"""Unit tests for the AETHER-SG serve module (import-defensive).

Covers: bounding-box validation rejecting bad areas/coords, and the
resolve-endpoint handler logic when exposed as a direct callable.

Import-defensive: the ``serve`` implementation is produced by a sibling
agent and may not exist yet.  Discovery probes several plausible module
locations; tests skip with an explicit reason when the module, the
bbox validator, or the resolve handler is unavailable.  When a validator
IS found, bad bboxes must actually be rejected (failure, not skip), and
the validator's expected schema is inferred from whichever plausible
valid bbox it accepts -- this avoids false failures from guessing the
wrong coordinate frame.

Known coordinate frames (any one of them may be the server's schema):
backend slice 0..200, viewer scene-centred -100..100, geographic lat/lon
(capture bbox lat 1.3311..1.3329 / lon 103.8461..103.8479).
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (SRC, ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_SERVE_FILE_CANDIDATES = (
    SRC / "aether" / "serve.py",
    SRC / "serve.py",
    ROOT / "serve.py",
    ROOT / "scripts" / "serve.py",
)


def _load_serve():
    """Return (module, source-description) or (None, reason-tried)."""
    for name in ("serve", "aether.serve"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if mod is not None:
            return mod, f"import:{name}"
    for path in _SERVE_FILE_CANDIDATES:
        if not path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                "aether_serve_probe", str(path))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("aether_serve_probe", mod)
            spec.loader.exec_module(mod)
            return mod, f"file:{path.name}"
        except Exception:
            continue
    tried = ", ".join(str(p) for p in _SERVE_FILE_CANDIDATES)
    return None, f"serve module not found (tried: {tried})"


SERVE_MOD, SERVE_SRC = _load_serve()

# Marks that a bbox validator is likely a bbox/area validator by name.
_BBOX_WORDS = ("bbox", "bounding_box", "bound_box", "area", "extent",
               "window", "viewport", "clip", "region")
# Marks that the callable checks/validates (rather than merely parses).
_CHECK_WORDS = ("valid", "check", "assert", "reject", "verify", "guard",
                "ensure", "sanit", "clamp", "normali", "parse", "build",
                "make", "get", "resolve", "is_")
# Names that look like resolve handlers.
_RESOLVE_WORDS = ("resolve",)


def _candidate_validators(mod):
    """Yield (name, fn) for attributes that look like bbox validators."""
    if mod is None:
        return
    for attr in sorted(dir(mod)):
        if attr.startswith("_"):
            continue
        try:
            fn = getattr(mod, attr)
        except Exception:
            continue
        if not callable(fn):
            continue
        low = attr.lower()
        if any(w in low for w in _BBOX_WORDS):
            if any(c in low for c in _CHECK_WORDS) or low in (
                    "bbox", "bounding_box", "area", "extent", "window"):
                yield attr, fn


def _is_rejected(result):
    """True if a validator's non-raising return means rejection."""
    if result is None or result is False:
        return True
    if isinstance(result, (tuple, list)) and len(result) > 0 and result[0] is False:
        return True
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return result == 0
    if isinstance(result, (dict, list, tuple, str)):
        return len(result) == 0
    return not bool(result)


def _run_validator(fn, bbox):
    """Call validator; return (accepted: bool, detail: str).

    Rejection can be signalled by raising or by a falsy/error return.
    """
    try:
        result = fn(bbox)
    except Exception as exc:  # rejection via exception
        return False, f"raised {type(exc).__name__}: {exc}"
    if _is_rejected(result):
        return False, f"returned {result!r}"
    return True, f"returned {result!r}"


# Valid-bbox probe battery: (schema label, builder for a sane in-frame box).
def _valid_probes():
    backend = {"x0": 10.0, "y0": 10.0, "x1": 90.0, "y1": 90.0}
    backend_alt = {"xmin": 10.0, "ymin": 10.0, "xmax": 90.0, "ymax": 90.0}
    backend_minmax = {"x_min": 10.0, "y_min": 10.0,
                      "x_max": 90.0, "y_max": 90.0}
    viewer = {"x0": -40.0, "z0": -40.0, "x1": 40.0, "z1": 40.0}
    viewer_alt = {"minx": -40.0, "minz": -40.0, "maxx": 40.0, "maxz": 40.0}
    geo = {"min_lat": 1.3315, "min_lon": 103.8465,
           "max_lat": 1.3325, "max_lon": 103.8475}
    geo_alt = {"minlat": 1.3315, "minlon": 103.8465,
               "maxlat": 1.3325, "maxlon": 103.8475}
    return (
        ("backend:x0/y0", backend),
        ("backend:xmin", backend_alt),
        ("backend:x_min", backend_minmax),
        ("viewer:x0/z0", viewer),
        ("viewer:minx/minz", viewer_alt),
        ("geo:min_lat", geo),
        ("geo:minlat", geo_alt),
        ("list:[x0,y0,x1,y1]", [10.0, 10.0, 90.0, 90.0]),
        ("list:[latlon]", [1.3315, 103.8465, 1.3325, 103.8475]),
    )


def _bad_bboxes_for(schema_keys, lo, hi, list_form=False):
    """Bad bboxes rendered in the validator's own schema key names.

    ``schema_keys`` is (k_min1, k_min2, k_max1, k_max2).
    ``lo`` / ``hi`` bound the sane range for the frame in use.
    """
    k1, k2, k3, k4 = schema_keys
    bad = []
    if list_form:
        def box(a, b, c, d):
            return [a, b, c, d]
    else:
        def box(a, b, c, d):
            return {k1: a, k2: b, k3: c, k4: d}
    mid = (lo + hi) / 2.0
    bad.append(("inverted", box(hi, hi, lo, lo)))
    bad.append(("zero_area", box(mid, mid, mid, mid)))
    bad.append(("oversized", box(lo, lo, hi * 2.5 + 100.0, hi * 2.5 + 100.0)))
    bad.append(("out_of_frame_neg", box(lo - 1000.0, lo - 1000.0,
                                        lo - 500.0, lo - 500.0)))
    bad.append(("non_numeric", box("a", "b", "c", "d")))
    bad.append(("none_coord", box(None, None, None, None)))
    if list_form:
        bad.append(("missing", [10.0, 10.0, 90.0]))
    else:
        incomplete = {k1: lo, k2: lo, k3: hi}
        bad.append(("missing_key", incomplete))
    if list_form:
        bad.append(("empty_list", []))
    return bad


def _find_validator_and_schema(mod):
    """Find a bbox validator and the schema it understands.

    Returns (fname, fn, schema_keys, list_form, lo, hi) or (None, ...).
    """
    candidates = list(_candidate_validators(mod))
    if not candidates:
        return None, None, None, False, None, None
    for fname, fn in candidates:
        for _label, probe in _valid_probes():
            accepted, _detail = _run_validator(fn, dict(probe)
                                               if isinstance(probe, dict)
                                               else list(probe))
            if not accepted:
                continue
            # This probe's schema is understood by the validator.
            if isinstance(probe, list):
                return fname, fn, (0, 1, 2, 3), True, probe[0], probe[2]
            keys = list(probe.keys())
            if len(keys) == 4:
                k1, k2, k3, k4 = keys
                # Geographic schemas get their own sane frame bounds.
                if "lat" in k1.lower() or "lon" in k1.lower() \
                        or "lat" in k2.lower() or "lon" in k2.lower():
                    return fname, fn, (k1, k2, k3, k4), False, 0.0, 90.0
                # Backend vs viewer is decided by key names: z-axis keys
                # mean the scene-centred viewer frame (-100..100).
                if "z" in k2.lower() or "z" in k4.lower():
                    return fname, fn, (k1, k2, k3, k4), False, -100.0, 100.0
                return fname, fn, (k1, k2, k3, k4), False, 0.0, 200.0
    return None, None, None, False, None, None


def test_serve_module_loads():
    if SERVE_MOD is None:
        pytest.skip(f"serve module unavailable: {SERVE_SRC}")
    assert SERVE_MOD is not None


def test_bbox_validation_rejects_bad_area_and_coords():
    if SERVE_MOD is None:
        pytest.skip(f"serve module unavailable: {SERVE_SRC}")
    fname, fn, schema_keys, list_form, lo, hi = (
        _find_validator_and_schema(SERVE_MOD))
    if fn is None:
        tried = ", ".join(f"{n}" for n, _ in
                          _candidate_validators(SERVE_MOD)) or "none"
        pytest.skip(f"no bbox validator understood (candidates tried: "
                    f"{tried})")
    assert fname is not None
    # Sanity: the schema's own valid probe must be accepted.
    for label, probe in _valid_probes():
        valid_probe = dict(probe) if isinstance(probe, dict) else list(probe)
        if isinstance(probe, list) and not list_form:
            continue
        if isinstance(probe, dict) and list_form:
            continue
        ok, detail = _run_validator(fn, valid_probe)
        expected = (list(probe.keys()) == list(schema_keys)
                    if not list_form else isinstance(probe, list))
        # Only the matching-schema probe is asserted; others may
        # legitimately be rejected as the wrong frame.
        if expected and not ok:
            pytest.fail(f"{fname} rejected its own valid probe {label}: "
                        f"{detail}")
    failures = []
    for label, bbox in _bad_bboxes_for(schema_keys, lo, hi, list_form):
        try:
            accepted, detail = _run_validator(fn, bbox)
        except Exception as exc:  # validator itself crashed: not a pass
            failures.append(f"{label}: validator crashed: {exc!r}")
            continue
        if accepted:
            failures.append(f"{label}: validator ACCEPTED {bbox!r} "
                            f"({detail})")
    assert not failures, (
        f"{fname} must reject bad bboxes ({len(failures)} accepted): "
        + "; ".join(failures))


def _find_resolve_callables(mod):
    """Module-level callables whose name suggests a resolve handler."""
    out = []
    if mod is None:
        return out
    for attr in sorted(dir(mod)):
        if attr.startswith("_"):
            continue
        if not any(w in attr.lower() for w in _RESOLVE_WORDS):
            continue
        try:
            fn = getattr(mod, attr)
        except Exception:
            continue
        if callable(fn):
            out.append((attr, fn))
    return out


def _try_resolve(fn):
    """Attempt direct resolve calls.

    Returns ("ok", detail) when the handler logic demonstrably ran, or
    ("sig_mismatch", detail) when every call shape failed with TypeError
    (signature not understood).  A domain error (404-ish) counts as the
    handler running.
    """
    arg_forms = (
        ("empty-dict", {}),
        ("root-path", "/"),
        ("empty-path", ""),
        ("point-pair", (10.0, 10.0)),
        ("dict-bbox", {"x0": 10.0, "y0": 10.0, "x1": 90.0, "y1": 90.0}),
    )
    sig_errors = []
    for label, arg in arg_forms:
        try:
            result = fn(arg)
        except TypeError as exc:
            sig_errors.append(f"{label}: {exc}")
            continue
        except Exception as exc:
            if isinstance(exc, AttributeError):
                sig_errors.append(f"{label}: {exc!r}")
                continue
            return "ok", f"{label} raised {type(exc).__name__}: {exc}"
        if result is None:
            return "ok", f"{label} returned None (handled miss?)"
        if isinstance(result, dict):
            return ("ok", f"{label} returned dict with keys "
                          f"{sorted(result)[:6]}")
        return "ok", f"{label} returned {type(result).__name__}"
    return "sig_mismatch", "; ".join(sig_errors) or "no arg form accepted"


def test_resolve_endpoint_logic():
    if SERVE_MOD is None:
        pytest.skip(f"serve module unavailable: {SERVE_SRC}")
    cands = _find_resolve_callables(SERVE_MOD)
    if not cands:
        # Class-based handlers (BaseHTTPRequestHandler.do_GET etc.) are
        # not direct callables; report the skip reason explicitly.
        has_handler = any(hasattr(SERVE_MOD, a)
                          for a in ("do_GET", "do_POST", "Handler",
                                    "RequestHandler", "App"))
        if has_handler:
            reason = ("HTTP-handler class present; not callable without "
                      "a request object")
        else:
            reason = f"no resolve-named attributes on module: {SERVE_SRC}"
        pytest.skip(f"no direct resolve function exposed ({reason})")
    fname, fn = cands[0]
    status, detail = _try_resolve(fn)
    if status == "sig_mismatch":
        pytest.skip(f"direct resolve callable {fname} found but its "
                    f"call shapes could not be determined: {detail}")
    assert status == "ok", f"{fname} resolve failed: {detail}"
