"""Convert a real Aleph capture directory into viewer assets.

Aleph captures contain ``mesh.glb``, ``satellite.png``/``satellite.tif``,
``map.osm``, ``terrain.tif``, ``manifest.json`` and
``streetview/photos.geojson`` (+ ``streetview/photos/`` jpgs).

Licence note (Research-AETHER-SG.md sec.5): Google Maps Platform Terms
(prohibit bulk scrape / cache / 3D-derivation) and Maps Additional Terms
mean Google-derived layers are research-only and must never be claimed
redistributable or production-ready. OSM data is under the Open Database
Licence (ODbL) and AWS Terrain Tiles derive from open DEMs. This module
copies the heavy binaries into the viewer folder and writes a compact
``capture.json`` index. It never uploads anything and never rewrites
capture sources.

Deps: stdlib + Pillow + numpy only.
"""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

LABEL = "REAL Aleph capture \u2014 research only, Google layers not redistributable"


def _bbox_from_polygon(polygon: List[List[float]]) -> Optional[Dict[str, float]]:
    """Polygon coords are [lat, lon] pairs (Aleph manifest order)."""
    if not polygon:
        return None
    try:
        lats = [float(p[0]) for p in polygon]
        lons = [float(p[1]) for p in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    return {"min_lat": min(lats), "min_lon": min(lons),
            "max_lat": max(lats), "max_lon": max(lons)}


def _load_manifest(capture_dir: Path) -> Optional[Dict[str, Any]]:
    path = capture_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _manifest_summary(manifest: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not manifest:
        return None
    stages: Dict[str, Any] = {}
    for stage in manifest.get("stages", []) or []:
        mode = stage.get("mode") or "unknown"
        entry: Dict[str, Any] = {"mode": mode}
        for key in ("samples", "results"):
            if key in stage:
                try:
                    entry["count"] = len(stage[key])
                except TypeError:
                    pass
        for key in ("grid", "level", "radius", "full_sphere", "length"):
            if key in stage:
                entry[key] = stage[key]
        stages[mode] = entry
    opts = manifest.get("options", {}) or {}
    return {
        "format": manifest.get("format"),
        "version": manifest.get("version"),
        "state": manifest.get("state"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "sources": opts.get("include"),
        "options": {k: opts[k] for k in (
            "step", "fov", "satellite_zoom", "terrain_zoom", "mesh_level",
            "full_sphere", "streetview_format", "satellite_format") if k in opts},
        "stages": stages,
        "exports_saved": manifest.get("exports_saved"),
    }


def _copy_if_present(src: Path, out_dir: Path, out_name: str) -> bool:
    if not src.is_file():
        return False
    shutil.copyfile(src, out_dir / out_name)
    return True


def _file_info(path: Path) -> Dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {"file": path.name, "size_bytes": size}


def _read_photos(capture_dir: Path) -> List[Dict[str, Any]]:
    """GeoJSON first (has coords), photos/ dir as fallback."""
    geojson_path = capture_dir / "streetview" / "photos.geojson"
    if geojson_path.is_file():
        try:
            data = json.loads(geojson_path.read_text(encoding="utf-8"))
            photos: List[Dict[str, Any]] = []
            for feat in data.get("features", []) or []:
                props = feat.get("properties", {}) or {}
                geom = feat.get("geometry", {}) or {}
                coords = geom.get("coordinates") or [None, None]
                photo: Dict[str, Any] = {
                    "file": Path(props.get("filename", "")).name or None,
                    "lat": props.get("lat"),
                    "lon": props.get("lon"),
                    "pano_id": props.get("pano_id"),
                    "path_name": props.get("path_name"),
                }
                if photo["lat"] is None and len(coords) == 2:
                    photo["lon"], photo["lat"] = coords
                if props.get("heading") is not None:
                    photo["heading"] = props["heading"]
                if props.get("imagery_date") is not None:
                    photo["date"] = props["imagery_date"]
                if props.get("side") is not None:
                    photo["side"] = props["side"]
                photos.append(photo)
            if photos:
                return photos
        except (OSError, json.JSONDecodeError):
            pass
    photos_dir = capture_dir / "streetview" / "photos"
    if not photos_dir.is_dir():
        photos_dir = capture_dir / "photos"
    if photos_dir.is_dir():
        return [{"file": p.name, "lat": None, "lon": None}
                for p in sorted(photos_dir.iterdir()) if p.is_file()]
    return []


def _read_osm(capture_dir: Path) -> Dict[str, Any]:
    """Minimal OSM stats: counts via regex + bounds via XML/`<bounds>`.

    Regex counting avoids a full DOM parse of the multi-MB OSM file;
    ElementTree is used only for bounds when regex finds nothing.
    """
    osm_path = capture_dir / "map.osm"
    if not osm_path.is_file():
        return {"available": False}
    try:
        text = osm_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"available": False}
    info: Dict[str, Any] = {"available": True, "file": "map.osm"}
    try:
        info["size_bytes"] = osm_path.stat().st_size
    except OSError:
        pass
    info["nodes"] = len(re.findall(r"<node\s", text))
    info["ways"] = len(re.findall(r"<way\s", text))
    info["relations"] = len(re.findall(r"<relation\s", text))
    info["buildings"] = len(re.findall(r'k="building"', text))
    info["highways"] = len(re.findall(r'k="highway"', text))
    m = re.search(
        r'<bounds\s+minlat="([^"]+)"\s+minlon="([^"]+)"\s+'
        r'maxlat="([^"]+)"\s+maxlon="([^"]+)"', text)
    if m:
        info["bounds"] = {"min_lat": float(m.group(1)), "min_lon": float(m.group(2)),
                          "max_lat": float(m.group(3)), "max_lon": float(m.group(4))}
    else:
        try:
            root = ET.fromstring(text.encode("utf-8")[:2_000_000])
            bounds = root.find("bounds")
            if bounds is not None:
                info["bounds"] = {k: float(bounds.get(k))
                                  for k in ("minlat", "minlon", "maxlat", "maxlon")}  # type: ignore[arg-type]
        except ET.ParseError:
            pass
    return info


def _read_terrain(capture_dir: Path) -> Dict[str, Any]:
    """Elevation stats via Pillow + numpy (float32 GeoTIFF, metres)."""
    terrain_path = capture_dir / "terrain.tif"
    if not terrain_path.is_file():
        return {"available": False}
    try:
        from PIL import Image
        import numpy as np

        with Image.open(terrain_path) as im:
            width, height = im.size
            mode = im.mode
            arr = np.array(im)
        info: Dict[str, Any] = {
            "available": True, "file": "terrain.tif",
            "width": width, "height": height, "mode": mode,
            "min_elevation_m": float(arr.min()),
            "max_elevation_m": float(arr.max()),
            "mean_elevation_m": float(arr.mean()),
        }
        try:
            info["size_bytes"] = terrain_path.stat().st_size
        except OSError:
            pass
        return info
    except Exception:  # unreadable raster: report presence, not crash
        return {"available": True, "file": "terrain.tif", "stats": "unreadable"}


def convert_capture(capture_dir: str, out_dir: str) -> Dict[str, Any]:
    """Convert an Aleph capture dir into viewer assets; return capture dict."""
    capture_dir = Path(capture_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(capture_dir)
    summary = _manifest_summary(manifest)

    bbox = _bbox_from_polygon((manifest or {}).get("polygon", []) or [])
    if bbox is None:
        osm = _read_osm(capture_dir)
        bbox = osm.get("bounds")
        if bbox is not None:
            bbox = {"min_lat": bbox["min_lat"], "min_lon": bbox["min_lon"],
                    "max_lat": bbox["max_lat"], "max_lon": bbox["max_lon"]}

    mesh: Optional[Dict[str, Any]] = None
    if _copy_if_present(capture_dir / "mesh.glb", out_dir, "mesh.glb"):
        mesh = _file_info(out_dir / "mesh.glb")

    satellite: Optional[Dict[str, Any]] = None
    if _copy_if_present(capture_dir / "satellite.png", out_dir, "satellite.png"):
        satellite = _file_info(out_dir / "satellite.png")

    capture: Dict[str, Any] = {
        "label": LABEL,
        "capture": capture_dir.name,
        "bbox": bbox,
        "mesh": "mesh.glb" if mesh else None,
        "satellite": "satellite.png" if satellite else None,
        "photos": _read_photos(capture_dir),
        "osm": _read_osm(capture_dir),
        "terrain": _read_terrain(capture_dir),
        "manifest": summary,
        "files": {"mesh": mesh, "satellite": satellite},
    }
    (out_dir / "capture.json").write_text(
        json.dumps(capture, indent=2), encoding="utf-8")
    return capture


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Aleph capture -> viewer assets")
    parser.add_argument("capture_dir", help="Aleph capture directory")
    parser.add_argument("out_dir", help="Viewer output directory")
    args = parser.parse_args(argv)
    capture = convert_capture(args.capture_dir, args.out_dir)
    print(f"photos: {len(capture['photos'])}  osm: {capture['osm']}  "
          f"terrain: {capture['terrain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
