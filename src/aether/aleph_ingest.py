"""Subprocess wrapper around the `alephgeo` CLI (research-only).

Research-AETHER-SG.md sec.5: Google Maps Platform Terms prohibit bulk
scrape / cache / 3D derivation of Google layers. Everything here that
touches Street View, satellite imagery, or the 3D mesh is RESEARCH-ONLY
- never redistribute and never production. OSM (ODbL) and AWS Terrain
are open with attribution. Label research captures honestly in any demo.

Each public function runs one `alephgeo` invocation, always appends
`--json`, returns the parsed stdout dict, and raises AlephError when
alephgeo reports status == "error" (or emits no JSON / exits non-zero).
Statuses other than error (complete / partial / cancelled) are returned
as dicts for the caller to inspect. stdin is always detached, so no
call can block on alephgeo's interactive capture prompt.

Executable lookup (find_alephgeo): $ALEPHGEO_EXE, then the project .venv
(Scripts/alephgeo.exe on Windows, bin/alephgeo on POSIX), then alephgeo
on PATH, then [sys.executable, "-m", "alephgeo"].

The _build_*_argv helpers are pure (no subprocess): they return the argv
list for one invocation, so flags can be unit-checked offline without
network downloads.

`launch_dashboard` is the deliberate exception to the --json dict
contract: `alephgeo dashboard` has no --json flag and serves until
stopped, so it returns the live subprocess.Popen handle instead.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = [
    "AlephError",
    "CAPTURE_SOURCES",
    "STREET_VIEWS",
    "capture_create",
    "capture_export",
    "capture_resume",
    "dashboard",
    "dashboard_url",
    "find_alephgeo",
    "launch_dashboard",
    "resolve",
    "satellite",
    "streetview_single",
    "streetview_street",
]

#: The five sources `alephgeo capture create --sources` accepts.
CAPTURE_SOURCES = ("streetview", "satellite", "osm", "terrain", "mesh")

#: Values `alephgeo streetview --view` accepts.
STREET_VIEWS = ("forward", "backward", "left", "right", "both")


class AlephError(Exception):
    """An alephgeo failure: machine code + human message + details.

    `code` mirrors alephgeo's --json error code (for example
    "invalid_arguments", "interrupted", "request_failed", or a
    RequestError code from the alephgeo source). `details` carries any
    extra error payload (candidates, gaps, folder, ...). `exit_code` is
    the process exit code (0/1/2/130 in alephgeo) when one exists.
    """

    def __init__(self, code: str, message: str, *,
                 exit_code: int | None = None,
                 details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        """Return alephgeo's --json error shape for this failure."""
        return {"status": "error",
                "error": {"code": self.code, "message": self.message,
                          **self.details}}


_exe_cache: list[str] | None = None


def find_alephgeo() -> list[str]:
    """Return the argv prefix that runs alephgeo.

    Lookup order: $ALEPHGEO_EXE, project .venv exe, PATH, then
    [sys.executable, "-m", "alephgeo"]. The result is cached.
    """
    global _exe_cache
    if _exe_cache is not None:
        return list(_exe_cache)
    env = os.environ.get("ALEPHGEO_EXE")
    if env:
        _exe_cache = [env]
        return list(_exe_cache)
    root = Path(__file__).resolve().parents[2]  # src/aether/<this> -> root
    argv: list[str] | None = None
    for cand in (root / ".venv" / "Scripts" / "alephgeo.exe",
                 root / ".venv" / "bin" / "alephgeo"):
        if cand.is_file():
            argv = [str(cand)]
            break
    if argv is None:
        found = shutil.which("alephgeo")
        argv = [found] if found else [sys.executable, "-m", "alephgeo"]
    _exe_cache = argv
    return list(argv)


def _run(argv: list[str], *, timeout: float | None = None) -> dict:
    """Run alephgeo; parse the single --json object; raise AlephError."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, timeout=timeout)
    except FileNotFoundError as exc:
        raise AlephError("alephgeo_not_found",
                         f"cannot start {argv[0]!r}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AlephError("timeout",
                         f"alephgeo exceeded timeout={timeout}s") from exc
    out = (proc.stdout or "").strip()
    payload = None
    if out:
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict) and payload.get("status") == "error":
        err = payload.get("error") or {}
        if isinstance(err, dict):
            code = str(err.get("code") or "request_failed")
            message = str(err.get("message") or "alephgeo reported an error")
            details = {k: v for k, v in err.items()
                       if k not in ("code", "message")}
            if payload.get("folder") and "folder" not in details:
                details["folder"] = payload["folder"]
        else:
            code, message, details = "request_failed", str(err), {}
        raise AlephError(code, message, exit_code=proc.returncode,
                         details=details)
    tail = (proc.stderr or "").strip()[-2000:]
    if payload is None:
        raise AlephError(f"exit_{proc.returncode}",
                         f"alephgeo exited {proc.returncode} without a "
                         f"usable --json result on stdout. stderr: {tail}")
    if proc.returncode != 0:
        raise AlephError("request_failed",
                         f"alephgeo exited {proc.returncode} without "
                         f"status=error. stderr: {tail}")
    return payload


def _network_argv(*, geocoder=None, delay=None, cache_dir=None,
                  refresh=False) -> list[str]:
    argv: list[str] = []
    if geocoder is not None:
        argv += ["--geocoder", str(geocoder)]
    if delay is not None:
        argv += ["--delay", repr(float(delay))]
    if cache_dir is not None:
        argv += ["--cache-dir", str(cache_dir)]
    if refresh:
        argv.append("--refresh")
    return argv


def _pair(pair, label: str) -> tuple[float, float]:
    if pair is None or len(pair) != 2:
        raise ValueError(f"{label} takes exactly two numbers (LAT LON).")
    try:
        return float(pair[0]), float(pair[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} takes numeric LAT LON, got {pair!r}") from exc


def _int_in(value, lo: int, hi: int, flag: str) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag} must be an integer, got {value!r}") from exc
    if not lo <= iv <= hi:
        raise ValueError(f"{flag} must be in {lo}..{hi}, got {iv}")
    return iv


def _build_resolve_argv(query=None, at=None, *, streets=False, nearby=False,
                        radius=None, limit=None, geocoder=None, delay=None,
                        cache_dir=None, refresh=False) -> list[str]:
    """Pure argv for `alephgeo resolve` (validates alephgeo's rules)."""
    if (query is None) == (at is None):
        raise ValueError("resolve needs exactly one of query / at.")
    if streets and query is None:
        raise ValueError("--streets requires a place-name query.")
    if nearby and at is None:
        raise ValueError("--nearby requires --at LAT LON.")
    if radius is not None and not nearby:
        raise ValueError("--radius requires --nearby.")
    if limit is not None:
        limit = _int_in(limit, 1, 50, "--limit")
    argv = find_alephgeo() + ["resolve"]
    if query is not None:
        argv.append(str(query))
    else:
        lat, lon = _pair(at, "--at")
        argv += ["--at", repr(lat), repr(lon)]
    if streets:
        argv.append("--streets")
    if nearby:
        argv.append("--nearby")
    if radius is not None:
        argv += ["--radius", repr(float(radius))]
    if limit is not None:
        argv += ["--limit", str(limit)]
    argv += _network_argv(geocoder=geocoder, delay=delay,
                          cache_dir=cache_dir, refresh=refresh)
    argv.append("--json")
    return argv


def _build_streetview_single_argv(at=None, place=None, pano_id=None, *,
                                  heading=None, look_at=None, pitch=None,
                                  fov=None, full_sphere=False,
                                  sphere_zoom=None, radius=None, match=None,
                                  best_match=False, geocoder=None, delay=None,
                                  cache_dir=None, refresh=False, workers=None,
                                  streetview_format=None,
                                  output=None) -> list[str]:
    """Pure argv for a single-view `alephgeo streetview`."""
    given = [item for item in (at, place, pano_id) if item is not None]
    if len(given) != 1:
        raise ValueError("streetview_single needs exactly one of "
                         "at / place / pano_id.")
    if heading is not None and look_at is not None:
        raise ValueError("--heading and --look-at are mutually exclusive.")
    if radius is not None and pano_id is not None:
        raise ValueError("--radius applies to coordinates or a place name.")
    if match is not None and best_match:
        raise ValueError("--match and --best-match are mutually exclusive.")
    if (match is not None or best_match) and place is None:
        raise ValueError("--match / --best-match require place.")
    if fov is not None:
        fov = _int_in(fov, 5, 175, "--fov")
    if sphere_zoom is not None:
        sphere_zoom = _int_in(sphere_zoom, 0, 5, "--sphere-zoom")
    if workers is not None:
        workers = _int_in(workers, 1, 64, "--workers")
    if streetview_format is not None and streetview_format not in ("jpg", "png"):
        raise ValueError("--streetview-format must be jpg or png.")
    argv = find_alephgeo() + ["streetview"]
    if pano_id is not None:
        argv += ["--pano-id", str(pano_id)]
    elif place is not None:
        argv += ["--place", str(place)]
    else:
        lat, lon = _pair(at, "--at")
        argv += ["--at", repr(lat), repr(lon)]
    if radius is not None:
        argv += ["--radius", repr(float(radius))]
    if heading is not None:
        argv += ["--heading", repr(float(heading))]
    if look_at is not None:
        lat, lon = _pair(look_at, "--look-at")
        argv += ["--look-at", repr(lat), repr(lon)]
    if match is not None:
        argv += ["--match", str(match)]
    if best_match:
        argv.append("--best-match")
    argv += _camera_argv(pitch=pitch, fov=fov, full_sphere=full_sphere,
                         sphere_zoom=sphere_zoom)
    argv += _network_argv(geocoder=geocoder, delay=delay,
                          cache_dir=cache_dir, refresh=refresh)
    if workers is not None:
        argv += ["--workers", str(workers)]
    if streetview_format is not None:
        argv += ["--streetview-format", streetview_format]
    if output is not None:
        argv += ["-o", str(output)]
    argv.append("--json")
    return argv


def _camera_argv(*, pitch=None, fov=None, full_sphere=False,
                 sphere_zoom=None) -> list[str]:
    argv: list[str] = []
    if pitch is not None:
        argv += ["--pitch", repr(float(pitch))]
    if fov is not None:
        argv += ["--fov", str(_int_in(fov, 5, 175, "--fov"))]
    if full_sphere:
        argv.append("--full-sphere")
    if sphere_zoom is not None:
        argv += ["--sphere-zoom", str(_int_in(sphere_zoom, 0, 5, "--sphere-zoom"))]
    return argv


def _build_streetview_street_argv(street, *, stops=None, step=None, view=None,
                                  route=None, reverse=False, pitch=None,
                                  fov=None, full_sphere=False,
                                  sphere_zoom=None, match=None,
                                  best_match=False, geocoder=None, delay=None,
                                  cache_dir=None, refresh=False, workers=None,
                                  streetview_format=None,
                                  output=None) -> list[str]:
    """Pure argv for a street-sequence `alephgeo streetview --street`."""
    if not street:
        raise ValueError("streetview_street needs a street name.")
    if (stops is None) == (step is None):
        raise ValueError("streetview_street needs exactly one of stops / step.")
    if view is not None and view not in STREET_VIEWS:
        raise ValueError(f"--view must be one of {', '.join(STREET_VIEWS)}.")
    if stops is not None:
        stops = _int_in(stops, 1, 10 ** 9, "--stops")
    if match is not None and best_match:
        raise ValueError("--match and --best-match are mutually exclusive.")
    if workers is not None:
        workers = _int_in(workers, 1, 64, "--workers")
    if streetview_format is not None and streetview_format not in ("jpg", "png"):
        raise ValueError("--streetview-format must be jpg or png.")
    argv = find_alephgeo() + ["streetview", "--street", str(street)]
    if stops is not None:
        argv += ["--stops", str(stops)]
    if step is not None:
        argv += ["--step", repr(float(step))]
    if view is not None:
        argv += ["--view", view]
    if route is not None:
        argv += ["--route", str(_int_in(route, 0, 10 ** 9, "--route"))]
    if reverse:
        argv.append("--reverse")
    if match is not None:
        argv += ["--match", str(match)]
    if best_match:
        argv.append("--best-match")
    argv += _camera_argv(pitch=pitch, fov=fov, full_sphere=full_sphere,
                         sphere_zoom=sphere_zoom)
    argv += _network_argv(geocoder=geocoder, delay=delay,
                          cache_dir=cache_dir, refresh=refresh)
    if workers is not None:
        argv += ["--workers", str(workers)]
    if streetview_format is not None:
        argv += ["--streetview-format", streetview_format]
    if output is not None:
        argv += ["-o", str(output)]
    argv.append("--json")
    return argv


def _normalise_tile(tile) -> str:
    """Return canonical "Z/X/Y" for a tile string or (z, x, y) triple."""
    if isinstance(tile, str):
        parts = tile.split("/")
        if len(parts) != 3:
            raise ValueError(f"--tile must be Z/X/Y, got {tile!r}")
        try:
            z, x, y = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"--tile must be integer Z/X/Y, got {tile!r}") from exc
    else:
        if len(tile) != 3:
            raise ValueError(f"--tile takes Z/X/Y or (z, x, y), got {tile!r}")
        z, x, y = (int(part) for part in tile)
    return f"{z}/{x}/{y}"


def _build_satellite_argv(at=None, place=None, bbox=None, tile=None, *,
                          zoom=None, size=None, match=None, best_match=False,
                          geocoder=None, delay=None, cache_dir=None,
                          refresh=False, workers=None, tile_format=None,
                          output=None) -> list[str]:
    """Pure argv for `alephgeo satellite` (validates alephgeo's rules)."""
    given = [item for item in (at, place, bbox, tile) if item is not None]
    if len(given) != 1:
        raise ValueError("satellite needs exactly one of at / place / bbox / tile.")
    if size is not None and (bbox is not None or tile is not None):
        raise ValueError("--size applies to coordinates or a place name.")
    if tile is not None and zoom is not None:
        raise ValueError("--tile already includes its zoom; omit --zoom.")
    if zoom is not None:
        zoom = _int_in(zoom, 1, 21, "--zoom")
    if workers is not None:
        workers = _int_in(workers, 1, 64, "--workers")
    if tile_format is not None and tile_format not in ("jpg", "png"):
        raise ValueError("--tile-format must be jpg or png.")
    if match is not None and best_match:
        raise ValueError("--match and --best-match are mutually exclusive.")
    if (match is not None or best_match) and place is None:
        raise ValueError("--match / --best-match require place.")
    argv = find_alephgeo() + ["satellite"]
    if place is not None:
        argv += ["--place", str(place)]
    elif bbox is not None:
        if len(bbox) != 4:
            raise ValueError("--bbox takes S W N E (south, west, north, east).")
        argv += ["--bbox"] + [repr(float(v)) for v in bbox]
    elif tile is not None:
        argv += ["--tile", _normalise_tile(tile)]
    else:
        lat, lon = _pair(at, "--at")
        argv += ["--at", repr(lat), repr(lon)]
    if size is not None:
        argv += ["--size", repr(float(size))]
    if zoom is not None:
        argv += ["--zoom", str(zoom)]
    if match is not None:
        argv += ["--match", str(match)]
    if best_match:
        argv.append("--best-match")
    argv += _network_argv(geocoder=geocoder, delay=delay,
                          cache_dir=cache_dir, refresh=refresh)
    if workers is not None:
        argv += ["--workers", str(workers)]
    if tile_format is not None:
        argv += ["--tile-format", tile_format]
    if output is not None:
        argv += ["-o", str(output)]
    argv.append("--json")
    return argv


def _build_capture_create_argv(bbox=None, polygon=None, *, sources=None,
                               plan=False, yes=True, depth=None, step=None,
                               fov=None, streetview_format=None,
                               full_sphere=False, sphere_zoom=None,
                               satellite_zoom=None, tile_format=None,
                               terrain_zoom=None, mesh_level=None, delay=None,
                               workers=None, cache_dir=None, refresh=False,
                               output=None) -> list[str]:
    """Pure argv for `alephgeo capture create` (non-interactive only)."""
    if (bbox is None) == (polygon is None):
        raise ValueError("capture_create needs exactly one of bbox / polygon.")
    if plan and yes:
        raise ValueError("--plan and --yes are mutually exclusive.")
    if not plan and not yes:
        raise ValueError("capture_create is non-interactive: pass yes=True "
                         "(download) or plan=True (plan only).")
    if sources is not None:
        sources = list(sources)
        if not sources:
            raise ValueError("--sources must name at least one source.")
        bad = [s for s in sources if s not in CAPTURE_SOURCES]
        if bad:
            raise ValueError(f"invalid sources {bad!r}; "
                             f"choose from {', '.join(CAPTURE_SOURCES)}.")
        sources = list(dict.fromkeys(sources))  # dedupe, keep order
    if depth is not None and depth not in ("main", "roads", "all"):
        raise ValueError("--depth must be main, roads, or all.")
    if fov is not None:
        fov = _int_in(fov, 5, 175, "--fov")
    if sphere_zoom is not None:
        sphere_zoom = _int_in(sphere_zoom, 0, 5, "--sphere-zoom")
    if satellite_zoom is not None:
        satellite_zoom = _int_in(satellite_zoom, 1, 21, "--satellite-zoom")
    if terrain_zoom is not None:
        terrain_zoom = _int_in(terrain_zoom, 1, 14, "--terrain-zoom")
    if mesh_level is not None:
        mesh_level = _int_in(mesh_level, 1, 22, "--mesh-level")
    if workers is not None:
        workers = _int_in(workers, 1, 64, "--workers")
    if streetview_format is not None and streetview_format not in ("jpg", "png"):
        raise ValueError("--streetview-format must be jpg or png.")
    if tile_format is not None and tile_format not in ("jpg", "png"):
        raise ValueError("--tile-format must be jpg or png.")
    argv = find_alephgeo() + ["capture", "create"]
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("--bbox takes S W N E (south, west, north, east).")
        argv += ["--bbox"] + [repr(float(v)) for v in bbox]
    else:
        if len(polygon) < 3:
            raise ValueError("--polygon needs at least three LAT LON pairs.")
        flat: list[str] = []
        for vertex in polygon:
            lat, lon = _pair(vertex, "--polygon")
            flat += [repr(lat), repr(lon)]
        argv += ["--polygon"] + flat
    if sources:
        argv += ["--sources"] + sources
    argv.append("--plan" if plan else "--yes")
    if depth is not None:
        argv += ["--depth", depth]
    if step is not None:
        argv += ["--step", repr(float(step))]
    if fov is not None:
        argv += ["--fov", str(fov)]
    if streetview_format is not None:
        argv += ["--streetview-format", streetview_format]
    argv += _camera_argv(fov=None, full_sphere=full_sphere,
                         sphere_zoom=sphere_zoom)
    if satellite_zoom is not None:
        argv += ["--satellite-zoom", str(satellite_zoom)]
    if tile_format is not None:
        argv += ["--tile-format", tile_format]
    if terrain_zoom is not None:
        argv += ["--terrain-zoom", str(terrain_zoom)]
    if mesh_level is not None:
        argv += ["--mesh-level", str(mesh_level)]
    if delay is not None:
        argv += ["--delay", repr(float(delay))]
    if workers is not None:
        for flag in ("--streetview-workers", "--satellite-workers",
                     "--terrain-workers", "--mesh-workers"):
            argv += [flag, str(workers)]
    if cache_dir is not None:
        argv += ["--cache-dir", str(cache_dir)]
    if refresh:
        argv.append("--refresh")
    if output is not None:
        argv += ["-o", str(output)]
    argv.append("--json")
    return argv


def _build_capture_resume_argv(folder, *, yes=True, cache_dir=None,
                               refresh=False) -> list[str]:
    """Pure argv for `alephgeo capture resume` (non-interactive only)."""
    if not folder:
        raise ValueError("capture_resume needs a run folder.")
    if not yes:
        raise ValueError("capture_resume is non-interactive: pass yes=True.")
    argv = find_alephgeo() + ["capture", "resume", str(folder), "--yes"]
    if cache_dir is not None:
        argv += ["--cache-dir", str(cache_dir)]
    if refresh:
        argv.append("--refresh")
    argv.append("--json")
    return argv


def _build_capture_export_argv(folder) -> list[str]:
    """Pure argv for `alephgeo capture export` (offline rebuild)."""
    if not folder:
        raise ValueError("capture_export needs a run folder.")
    return find_alephgeo() + ["capture", "export", str(folder), "--json"]


def _build_dashboard_argv(port=8100, captures_dir=None, no_browser=False,
                          cache_dir=None, refresh=False) -> list[str]:
    """Pure argv for `alephgeo dashboard` (no --json exists there)."""
    port = _int_in(port, 1, 65535, "--port")
    argv = find_alephgeo() + ["dashboard", "--port", str(port)]
    if captures_dir is not None:
        argv += ["-o", str(captures_dir)]
    if no_browser:
        argv.append("--no-browser")
    if cache_dir is not None:
        argv += ["--cache-dir", str(cache_dir)]
    if refresh:
        argv.append("--refresh")
    return argv


def resolve(query=None, at=None, *, streets=False, nearby=False, radius=None,
            limit=None, geocoder=None, delay=None, cache_dir=None,
            refresh=False, timeout=None) -> dict:
    """Geocode a place name or reverse-geocode / list nearby POIs.

    Pass exactly one of `query` (place name) or `at` ((lat, lon)).
    `streets=True` restricts a name search to streets; `nearby=True`
    (needs `at`) lists nearby POIs instead of reverse geocoding, and
    `radius` (metres) then applies. Returns alephgeo's --json dict
    (keys: command, mode, status, results, cache). Raises
    ValueError on invalid combinations and AlephError when alephgeo
    reports status == "error".

    No Google layers here - Photon/OSM geocoding only (OSM ODbL,
    attribution inside the response). Downstream captures that use
    Google layers remain research-only under Google Maps Platform
    Terms - never redistribute, never production
    (Admin/Research-AETHER-SG.md sec.5).
    """
    argv = _build_resolve_argv(
        query, at, streets=streets, nearby=nearby, radius=radius,
        limit=limit, geocoder=geocoder, delay=delay, cache_dir=cache_dir,
        refresh=refresh)
    return _run(argv, timeout=timeout)


def streetview_single(at=None, place=None, pano_id=None, *, heading=None,
                      look_at=None, pitch=None, fov=None, full_sphere=False,
                      sphere_zoom=None, radius=None, match=None,
                      best_match=False, geocoder=None, delay=None,
                      cache_dir=None, refresh=False, workers=None,
                      streetview_format=None, output=None,
                      timeout=None) -> dict:
    """Download one Street View view (single panorama position).

    Pass exactly one of `at` ((lat, lon), needs `radius` search),
    `place` (name, optionally with `match`/`best_match` selection), or
    `pano_id` (exact panorama). `heading` (degrees clockwise from
    north) and `look_at` ((lat, lon)) are mutually exclusive; `pitch`
    is degrees up/down; `fov` is 5..175; `full_sphere=True` saves a
    360-degree panorama per stop (camera direction and fov then do not
    apply) and `sphere_zoom` is 0..5. Returns alephgeo's --json dict
    (status + folder / photos / gaps). Raises AlephError on
    status == "error".

    Research-only: Street View is Google-derived data under Google
    Maps Platform Terms - research use only, never redistribute,
    never production (Admin/Research-AETHER-SG.md sec.5).
    """
    argv = _build_streetview_single_argv(
        at, place, pano_id, heading=heading, look_at=look_at, pitch=pitch,
        fov=fov, full_sphere=full_sphere, sphere_zoom=sphere_zoom,
        radius=radius, match=match, best_match=best_match,
        geocoder=geocoder, delay=delay, cache_dir=cache_dir,
        refresh=refresh, workers=workers,
        streetview_format=streetview_format, output=output)
    return _run(argv, timeout=timeout)


def streetview_street(street, *, stops=None, step=None, view=None,
                      route=None, reverse=False, pitch=None, fov=None,
                      full_sphere=False, sphere_zoom=None, match=None,
                      best_match=False, geocoder=None, delay=None,
                      cache_dir=None, refresh=False, workers=None,
                      streetview_format=None, output=None,
                      timeout=None) -> dict:
    """Follow one named street end to end, saving ordered captures.

    Pass exactly one of `stops` (total positions across the selected
    street sections) or `step` (target spacing in metres). `view` is
    one of forward / backward / left / right / both; `route` limits
    to one numbered street section; `reverse=True` reverses traversal
    within each section. Camera kwargs behave as in
    streetview_single. Returns alephgeo's --json dict (status may be
    "partial" with saved_stops / requested_stops and a gaps list).
    Raises AlephError on status == "error".

    Research-only: Street View is Google-derived data under Google
    Maps Platform Terms - research use only, never redistribute,
    never production (Admin/Research-AETHER-SG.md sec.5).
    """
    argv = _build_streetview_street_argv(
        street, stops=stops, step=step, view=view, route=route,
        reverse=reverse, pitch=pitch, fov=fov, full_sphere=full_sphere,
        sphere_zoom=sphere_zoom, match=match, best_match=best_match,
        geocoder=geocoder, delay=delay, cache_dir=cache_dir,
        refresh=refresh, workers=workers,
        streetview_format=streetview_format, output=output)
    return _run(argv, timeout=timeout)


def satellite(at=None, place=None, bbox=None, tile=None, *, zoom=None,
              size=None, match=None, best_match=False, geocoder=None,
              delay=None, cache_dir=None, refresh=False, workers=None,
              tile_format=None, output=None, timeout=None) -> dict:
    """Download a Google satellite patch or one exact Web-Mercator tile.

    Pass exactly one of `at` ((lat, lon) with `size` in metres),
    `place` (name, optionally with `match`/`best_match`), `bbox`
    ((south, west, north, east)), or `tile` ("Z/X/Y" or (z, x, y)).
    `zoom` is 1..21 (default 19) and is forbidden with `tile`; `size`
    is forbidden with `bbox`/`tile`. Returns alephgeo's --json dict
    (status + folder / path / width / height). Raises AlephError on
    status == "error".

    Research-only: satellite imagery is Google-derived data under
    Google Maps Platform Terms - research use only, never
    redistribute, never production (Admin/Research-AETHER-SG.md sec.5).
    """
    argv = _build_satellite_argv(
        at, place, bbox, tile, zoom=zoom, size=size, match=match,
        best_match=best_match, geocoder=geocoder, delay=delay,
        cache_dir=cache_dir, refresh=refresh, workers=workers,
        tile_format=tile_format, output=output)
    return _run(argv, timeout=timeout)


def capture_create(bbox=None, polygon=None, *, sources=None, plan=False,
                   yes=True, depth=None, step=None, fov=None,
                   streetview_format=None, full_sphere=False,
                   sphere_zoom=None, satellite_zoom=None, tile_format=None,
                   terrain_zoom=None, mesh_level=None, delay=None,
                   workers=None, cache_dir=None, refresh=False, output=None,
                   timeout=None) -> dict:
    """Plan and download one area capture (multi-source).

    Pass exactly one of `bbox` ((south, west, north, east)) or
    `polygon` (at least three (lat, lon) vertices, in order).
    `sources` is a subset of streetview / satellite / osm / terrain /
    mesh (default in alephgeo: streetview, satellite, osm, terrain).
    `workers` (1..64) is fanned out to all four per-source worker
    flags. This wrapper is non-interactive by construction: `yes`
    defaults to True (skip the confirmation prompt); pass
    `plan=True` instead to save a plan without downloading. Returns
    alephgeo's --json dict (keys: command, action, status, folder,
    manifest, plus estimate when planned). Raises AlephError on
    status == "error".

    Research-only: Google layers in the capture (Street View /
    satellite / 3D mesh) are under Google Maps Platform Terms -
    research use only, never redistribute, never production. OSM
    (ODbL) and AWS Terrain are open with attribution. Label research
    captures honestly in any demo (Admin/Research-AETHER-SG.md
    sec.5).
    """
    argv = _build_capture_create_argv(
        bbox, polygon, sources=sources, plan=plan, yes=yes, depth=depth,
        step=step, fov=fov, streetview_format=streetview_format,
        full_sphere=full_sphere, sphere_zoom=sphere_zoom,
        satellite_zoom=satellite_zoom, tile_format=tile_format,
        terrain_zoom=terrain_zoom, mesh_level=mesh_level, delay=delay,
        workers=workers, cache_dir=cache_dir, refresh=refresh,
        output=output)
    return _run(argv, timeout=timeout)


def capture_resume(folder, *, yes=True, cache_dir=None, refresh=False,
                   timeout=None) -> dict:
    """Continue a saved or planned capture run in `folder`.

    `folder` is the timestamped run directory containing manifest.json.
    `yes` defaults to True so the call never blocks on the interactive
    confirmation prompt (stdin is detached regardless). Returns
    alephgeo's --json dict (keys: command, action, status, folder,
    manifest). Raises AlephError on status == "error".

    Research-only: resuming continues a research capture whose
    Google-derived outputs (Street View / satellite / 3D mesh) stay
    under Google Maps Platform Terms - never redistribute, never
    production (Admin/Research-AETHER-SG.md sec.5).
    """
    argv = _build_capture_resume_argv(folder, yes=yes, cache_dir=cache_dir,
                                      refresh=refresh)
    return _run(argv, timeout=timeout)


def capture_export(folder, *, timeout=None) -> dict:
    """Rebuild images and metadata from saved files, fully offline.

    `folder` is the timestamped run directory containing manifest.json.
    No network access happens here, so this is safe in the AETHER-SG
    airgap phase. Returns alephgeo's --json dict (keys: command,
    action="export", status, folder, manifest). Raises AlephError on
    status == "error".

    Research-only: the rebuilt files are research captures whose
    Google-derived layers (Street View / satellite / 3D mesh) stay
    under Google Maps Platform Terms - never redistribute, never
    production (Admin/Research-AETHER-SG.md sec.5).
    """
    argv = _build_capture_export_argv(folder)
    return _run(argv, timeout=timeout)


def launch_dashboard(port=8100, *, captures_dir=None, no_browser=False,
                     cache_dir=None, refresh=False) -> subprocess.Popen:
    """Launch `alephgeo dashboard` as a child process; return its handle.

    This is the one deliberate exception to the --json dict contract:
    alephgeo's dashboard command has no --json flag and serves a local
    HTTP UI until stopped, so this launcher returns subprocess.Popen
    instead of a parsed dict. `captures_dir` maps to alephgeo's -o
    folder-holding-captures. The child inherits no stdin and logs
    nothing to this process; keep the Popen (`proc.poll()` /
    `proc.terminate()`) for the lifetime you want the server. Raises
    AlephError("alephgeo_not_found", ...) if the executable cannot be
    started.

    Research-only: the dashboard serves previously captured research
    data locally. Publishing or redistributing its Google-derived
    layers (Street View / satellite / 3D mesh) would breach Google
    Maps Platform Terms - research use only, never redistribute,
    never production (Admin/Research-AETHER-SG.md sec.5).
    """
    argv = find_alephgeo() + _build_dashboard_argv(
        port, captures_dir, no_browser, cache_dir, refresh)
    try:
        return subprocess.Popen(
            argv, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise AlephError("alephgeo_not_found",
                         f"cannot start {argv[0]!r}: {exc}") from exc


def dashboard_url(port=8100) -> str:
    """Return the local URL of a dashboard on `port` (127.0.0.1)."""
    return f"http://127.0.0.1:{int(port)}/"


#: Convenience alias for launch_dashboard.
dashboard = launch_dashboard

# Public aliases for the argv builders / runner (stable API for tests,
# CLI and MCP wrapping; the underscore forms are the implementations).
build_resolve_argv = _build_resolve_argv
build_streetview_single_argv = _build_streetview_single_argv
build_streetview_street_argv = _build_streetview_street_argv
build_satellite_argv = _build_satellite_argv
build_capture_create_argv = _build_capture_create_argv
build_capture_argv = _build_capture_create_argv
build_capture_resume_argv = _build_capture_resume_argv
build_capture_export_argv = _build_capture_export_argv
build_dashboard_argv = _build_dashboard_argv
run_aleph = _run
