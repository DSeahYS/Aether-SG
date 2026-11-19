"""Command-line frontend for aether.aleph_ingest (research-only).

Mirrors every wrapper function as a subcommand and prints one JSON
object to stdout. AlephError failures print alephgeo's
{"status": "error", "error": {...}} shape and exit 1; invalid
arguments print the same shape with code "invalid_arguments" and
exit 2. The `dashboard` subcommand prints a {"status": "launched",
"url": ..., "pid": ...} object, then stays attached until Ctrl+C
(exit 130), like `alephgeo dashboard` itself.

Research-only: Google layers (Street View / satellite / 3D mesh) are
under Google Maps Platform Terms - research use only, never
redistribute, never production (Admin/Research-AETHER-SG.md sec.5).
OSM (ODbL) and AWS Terrain are open with attribution; label research
captures honestly in any demo or paper.

Run with the project interpreter from the project root:
  C:\\VSCode Folder\\Ather-SG\\.venv\\Scripts\\python.exe scripts\\aleph.py ...
(This script adds ../src to sys.path itself, like run_demo.py.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aether.aleph_ingest import (  # noqa: E402
    CAPTURE_SOURCES,
    STREET_VIEWS,
    AlephError,
    capture_create,
    capture_export,
    capture_resume,
    dashboard_url,
    launch_dashboard,
    resolve,
    satellite,
    streetview_single,
    streetview_street,
)


def _pair(values):
    if values is None:
        return None
    return (values[0], values[1])


def _network(p):
    p.add_argument("--geocoder", default=None, metavar="URL")
    p.add_argument("--delay", type=float, default=None)
    p.add_argument("--cache-dir", default=None, metavar="DIR")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--timeout", type=float, default=None,
                   help="subprocess timeout in seconds (default: none)")


def _output(p):
    p.add_argument("-o", "--output", default=None, metavar="DIR",
                   help="parent output directory")


def _camera(p, *, single_only=False):
    if single_only:
        g = p.add_mutually_exclusive_group()
        g.add_argument("--heading", type=float)
        g.add_argument("--look-at", nargs=2, type=float, metavar=("LAT", "LON"))
    p.add_argument("--pitch", type=float, default=None)
    p.add_argument("--fov", type=int, default=None)
    p.add_argument("--full-sphere", action="store_true")
    p.add_argument("--sphere-zoom", type=int, default=None)


def _selection(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--match", default=None, metavar="TYPE/ID",
                   help="choose an OSM ID from an ambiguous-name response")
    g.add_argument("--best-match", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser (one subcommand per wrapper function)."""
    root = argparse.ArgumentParser(
        prog="aleph.py",
        description="JSON CLI over aether.aleph_ingest "
                    "(subprocess wrapper around alephgeo). "
                    "Google layers are research-only; see module docstring.")
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("resolve", help="find places / reverse geocode / nearby POIs")
    p.add_argument("query", nargs="?", default=None,
                   help="place name, including city or country")
    p.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"))
    p.add_argument("--streets", action="store_true")
    p.add_argument("--nearby", action="store_true")
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    _network(p)

    p = sub.add_parser("streetview-single", help="one Street View view")
    p.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"))
    p.add_argument("--place", default=None)
    p.add_argument("--pano-id", default=None, dest="pano_id")
    _camera(p, single_only=True)
    p.add_argument("--radius", type=float, default=None)
    _selection(p)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--streetview-format", choices=("jpg", "png"), default=None)
    _output(p)
    _network(p)

    p = sub.add_parser("streetview-street", help="an ordered street sequence")
    p.add_argument("street")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--stops", type=int, default=None)
    g.add_argument("--step", type=float, default=None)
    p.add_argument("--view", choices=STREET_VIEWS, default=None)
    p.add_argument("--route", type=int, default=None)
    p.add_argument("--reverse", action="store_true")
    _camera(p)
    _selection(p)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--streetview-format", choices=("jpg", "png"), default=None)
    _output(p)
    _network(p)

    p = sub.add_parser("satellite", help="a satellite patch or one exact tile")
    p.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"))
    p.add_argument("--place", default=None)
    p.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    p.add_argument("--tile", default=None, metavar="Z/X/Y")
    p.add_argument("--zoom", type=int, default=None)
    p.add_argument("--size", type=float, default=None)
    _selection(p)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--tile-format", choices=("jpg", "png"), default=None)
    _output(p)
    _network(p)

    p = sub.add_parser("capture-create", help="plan and download an area")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    p.add_argument("--polygon", nargs="+", type=float, metavar="LAT LON",
                   help="flat LAT LON list, at least three pairs")
    p.add_argument("--sources", nargs="+", choices=CAPTURE_SOURCES, default=None)
    p.add_argument("--plan", action="store_true",
                   help="save a plan without downloading")
    p.add_argument("--yes", action=argparse.BooleanOptionalAction, default=True,
                   help="skip the confirmation prompt (default: yes)")
    p.add_argument("--depth", choices=("main", "roads", "all"), default=None)
    p.add_argument("--step", type=float, default=None)
    p.add_argument("--fov", type=int, default=None)
    p.add_argument("--streetview-format", choices=("jpg", "png"), default=None)
    p.add_argument("--full-sphere", action="store_true")
    p.add_argument("--sphere-zoom", type=int, default=None)
    p.add_argument("--satellite-zoom", type=int, default=None)
    p.add_argument("--tile-format", choices=("jpg", "png"), default=None)
    p.add_argument("--terrain-zoom", type=int, default=None)
    p.add_argument("--mesh-level", type=int, default=None)
    p.add_argument("--delay", type=float, default=None)
    p.add_argument("--workers", type=int, default=None,
                   help="applied to every capture source (1..64)")
    p.add_argument("--cache-dir", default=None, metavar="DIR")
    p.add_argument("--refresh", action="store_true")
    _output(p)
    p.add_argument("--timeout", type=float, default=None)

    p = sub.add_parser("capture-resume", help="continue a saved or planned run")
    p.add_argument("folder", metavar="FOLDER")
    p.add_argument("--yes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache-dir", default=None, metavar="DIR")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--timeout", type=float, default=None)

    p = sub.add_parser("capture-export", help="rebuild from saved files (offline)")
    p.add_argument("folder", metavar="FOLDER")
    p.add_argument("--timeout", type=float, default=None)

    p = sub.add_parser("dashboard",
                       help="serve the captures UI (blocks until Ctrl+C)")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("-o", "--output", default=None, metavar="DIR",
                   help="folder holding captures")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--cache-dir", default=None, metavar="DIR")
    p.add_argument("--refresh", action="store_true")
    return root


def _polygon_pairs(flat):
    if flat is None:
        return None
    if len(flat) % 2:
        raise ValueError("--polygon takes LAT LON pairs (even count).")
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def main(argv=None) -> int:
    """Dispatch one subcommand; print one JSON object; return exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "dashboard":
            proc = launch_dashboard(
                args.port, captures_dir=args.output,
                no_browser=args.no_browser, cache_dir=args.cache_dir,
                refresh=args.refresh)
            print(json.dumps({"status": "launched",
                              "url": dashboard_url(args.port),
                              "pid": proc.pid}, ensure_ascii=False),
                  flush=True)
            try:
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                return 130
        if args.command == "resolve":
            result = resolve(
                args.query, at=_pair(args.at), streets=args.streets,
                nearby=args.nearby, radius=args.radius, limit=args.limit,
                geocoder=args.geocoder, delay=args.delay,
                cache_dir=args.cache_dir, refresh=args.refresh,
                timeout=args.timeout)
        elif args.command == "streetview-single":
            result = streetview_single(
                _pair(args.at), args.place, args.pano_id,
                heading=args.heading, look_at=_pair(args.look_at),
                pitch=args.pitch, fov=args.fov, full_sphere=args.full_sphere,
                sphere_zoom=args.sphere_zoom, radius=args.radius,
                match=args.match, best_match=args.best_match,
                geocoder=args.geocoder, delay=args.delay,
                cache_dir=args.cache_dir, refresh=args.refresh,
                workers=args.workers,
                streetview_format=args.streetview_format, output=args.output,
                timeout=args.timeout)
        elif args.command == "streetview-street":
            result = streetview_street(
                args.street, stops=args.stops, step=args.step, view=args.view,
                route=args.route, reverse=args.reverse, pitch=args.pitch,
                fov=args.fov, full_sphere=args.full_sphere,
                sphere_zoom=args.sphere_zoom, match=args.match,
                best_match=args.best_match, geocoder=args.geocoder,
                delay=args.delay, cache_dir=args.cache_dir,
                refresh=args.refresh, workers=args.workers,
                streetview_format=args.streetview_format, output=args.output,
                timeout=args.timeout)
        elif args.command == "satellite":
            result = satellite(
                _pair(args.at), args.place, tuple(args.bbox) if args.bbox else None,
                args.tile, zoom=args.zoom, size=args.size, match=args.match,
                best_match=args.best_match, geocoder=args.geocoder,
                delay=args.delay, cache_dir=args.cache_dir,
                refresh=args.refresh, workers=args.workers,
                tile_format=args.tile_format, output=args.output,
                timeout=args.timeout)
        elif args.command == "capture-create":
            result = capture_create(
                tuple(args.bbox) if args.bbox else None,
                _polygon_pairs(args.polygon), sources=args.sources,
                plan=args.plan, yes=args.yes, depth=args.depth,
                step=args.step, fov=args.fov,
                streetview_format=args.streetview_format,
                full_sphere=args.full_sphere, sphere_zoom=args.sphere_zoom,
                satellite_zoom=args.satellite_zoom,
                tile_format=args.tile_format,
                terrain_zoom=args.terrain_zoom, mesh_level=args.mesh_level,
                delay=args.delay, workers=args.workers,
                cache_dir=args.cache_dir, refresh=args.refresh,
                output=args.output, timeout=args.timeout)
        elif args.command == "capture-resume":
            result = capture_resume(
                args.folder, yes=args.yes, cache_dir=args.cache_dir,
                refresh=args.refresh, timeout=args.timeout)
        else:  # capture-export
            result = capture_export(args.folder, timeout=args.timeout)
    except AlephError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False))
        return 1
    except ValueError as exc:
        print(json.dumps({"status": "error",
                          "error": {"code": "invalid_arguments",
                                    "message": str(exc)}},
                         ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
