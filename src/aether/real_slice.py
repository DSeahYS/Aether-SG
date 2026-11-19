"""AETHER-SG real slice from an Aleph capture directory.

Builds a routing-ready 200 x 200 m slice fused from the real capture
artefacts in ``capture_dir``:

- ``mesh.glb``      : 3-D mesh (research-only Google-derived layer).
  Vertex clouds are parsed directly from the GLB binary and rasterised
  into a 2 m heightfield. GLB frame here is metres from the bbox centre
  with x east, y up and z south; slice space is x east 0..W,
  y north 0..H (x = glb_x + W/2, y = H/2 - glb_z).
- ``terrain.tif``   : GeoTIFF elevation (open DEM tile). Cropped to the
  manifest bbox via its geotransform, resampled to 200 x 200 cells at
  1 m, smoothed, datum shifted so the lowest cell is 0 m.
- ``map.osm``       : OpenStreetMap extract (ODbL). Pedestrian ways
  (``highway`` in footway, path, pedestrian, residential, service,
  cycleway, steps, living_street) become the wheelchair ground graph.
- ``streetview/photos.geojson`` : audit photo locations.

Derived buildings come from the mesh heightfield (cells more than 3 m
above the low-percentile ground datum), never from OSM building tags.

Primary API
-----------
build_real_slice(capture_dir) -> dict
    Returns a dict compatible with routing.ground_astar /
    routing.drone_astar / evaluator:

      "dem"           : float32 ndarray (200, 200) @ 1 m, index [iy, ix]
                        with iy = north (row 0 = south edge)
      "dem_res_m"     : float 1.0
      "extent"        : float 200.0 (slice width/height in metres)
      "extent_m"      : float 200.0 (same value, slice_data naming)
      "buildings"     : list[dict] axis-aligned boxes
                        {id, cx, cy, w, d, height_m, storeys, void_deck,
                         linkways} in slice coords (x east, y north)
      "nodes"         : list[dict] {id, x, y, z, kind="junction"}
                        (OSM node id kept as id)
      "edges"         : list[dict] {u, v, length_m, slope, width_m,
                        sheltered, has_ramp}
      "ramps"         : list[dict] (empty; no ramp ways in this capture)
      "landing_pads"  : list[dict] rooftops of derived buildings plus
                        the largest open ground cell
      "streetview"    : list[dict] audit points {x, y, lat, lon, file,
                        date, verified=False}
      "audit_points"  : alias of the same list
      "meta"          : dict with real counts, bbox and provenance

Licence note (Research-AETHER-SG.md sec.5): Google Maps Platform Terms
(prohibit bulk scrape / cache / 3D-derivation) and Maps Additional Terms
mean Google-derived layers are research-only and must never be claimed
redistributable or production-ready. OSM data is under the Open Database
Licence (ODbL) and AWS Terrain Tiles derive from open DEMs. This module
only reads the local capture; it never uploads anything.
"""

from __future__ import annotations

import json
import math
import struct
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np

EXTENT_M = 200.0
DEM_RES_M = 1.0
DEM_N = int(round(EXTENT_M / DEM_RES_M))        # 200 cells per side
HF_RES_M = 2.0
HF_N = int(round(EXTENT_M / HF_RES_M))          # 100 cells per side
BUILDING_MIN_AGL_M = 3.0

PEDESTRIAN_HIGHWAYS = (
    "footway", "path", "pedestrian", "residential",
    "service", "cycleway", "steps", "living_street",
)

# Default carriageway widths (metres) used when OSM ways carry no width tag.
WIDTH_BY_HIGHWAY = {
    "footway": 2.0,
    "path": 2.0,
    "pedestrian": 3.0,
    "residential": 4.0,
    "service": 3.0,
    "cycleway": 2.0,
    "steps": 1.5,
    "living_street": 4.0,
}


# ---------------------------------------------------------------------------
# Manifest / bbox
# ---------------------------------------------------------------------------

def _read_manifest(capture_dir: Path) -> dict:
    path = capture_dir / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    polygon = data.get("polygon") or []
    lats = [float(p[0]) for p in polygon]
    lons = [float(p[1]) for p in polygon]
    bbox = {
        "min_lat": min(lats), "max_lat": max(lats),
        "min_lon": min(lons), "max_lon": max(lons),
        "polygon": [[float(a), float(b)] for a, b in polygon],
    }
    return {"manifest": data, "bbox": bbox,
            "name": capture_dir.name}


# ---------------------------------------------------------------------------
# GLB parsing (vertex clouds, numpy + stdlib only)
# ---------------------------------------------------------------------------

def _node_matrix(node: dict) -> np.ndarray:
    """4x4 world matrix for one glTF node (matrix or TRS form)."""
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    r = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    rx, ry, rz, rw = (float(v) for v in r)
    mat = np.eye(4, dtype=np.float64)
    # Rotation matrix from unit quaternion.
    mat[:3, :3] = np.array([
        [1 - 2 * (ry * ry + rz * rz), 2 * (rx * ry - rz * rw),
         2 * (rx * rz + ry * rw)],
        [2 * (rx * ry + rz * rw), 1 - 2 * (rx * rx + rz * rz),
         2 * (ry * rz - rx * rw)],
        [2 * (rx * rz - ry * rw), 2 * (ry * rz + rx * rw),
         1 - 2 * (rx * rx + ry * ry)],
    ])
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    mat[:3, :3] = mat[:3, :3] * s[None, :]
    mat[:3, 3] = t
    return mat


def _scene_node_matrices(gltf: dict) -> dict:
    """Map node index -> world matrix, following the scene graph."""
    nodes = gltf.get("nodes", [])
    out: dict[int, np.ndarray] = {}

    def walk(idx: int, parent: np.ndarray) -> None:
        world = parent @ _node_matrix(nodes[idx])
        out[idx] = world
        for child in nodes[idx].get("children", []) or []:
            walk(int(child), world)

    scenes = gltf.get("scenes", []) or [{}]
    for root in scenes[0].get("nodes", []) or []:
        walk(int(root), np.eye(4))
    return out


def _read_accessor(accessors, buffer_views, bin_chunk, acc_idx) -> np.ndarray:
    """Decode one accessor to a float array (VEC3 -> N x 3)."""
    acc = accessors[acc_idx]
    view = buffer_views[acc["bufferView"]]
    comp = acc["componentType"]
    n_comp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}.get(acc["type"], 1)
    dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32,
             5121: np.uint8, 5120: np.int8, 5122: np.int16}[comp]
    count = int(acc["count"])
    stride = int(view.get("byteStride") or (n_comp * np.dtype(dtype).itemsize))
    start = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    raw = np.frombuffer(bin_chunk, dtype=np.uint8)[start:start + stride * count]
    if raw.size < stride * count:
        raise ValueError(f"GLB accessor {acc_idx} truncated")
    if stride == n_comp * np.dtype(dtype).itemsize:
        return raw.view(dtype).reshape(count, n_comp).astype(np.float64)
    rows = raw.reshape(count, stride)[:, :n_comp * np.dtype(dtype).itemsize]
    return rows.reshape(count, n_comp).view(dtype).astype(np.float64)


def read_glb_vertices(path: Path) -> tuple[np.ndarray, dict]:
    """All POSITION vertices of a GLB in world coordinates."""
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError(f"{path} is not a GLB file")
    (_magic, _version, _total) = struct.unpack("<III", raw[:12])
    (json_len, json_type) = struct.unpack("<II", raw[12:20])
    gltf = json.loads(raw[20:20 + json_len].decode("utf-8"))
    off = 20 + json_len
    bin_chunk = b""
    while off + 8 <= len(raw):
        (chunk_len, chunk_type) = struct.unpack("<II", raw[off:off + 8])
        if chunk_type == 0x004E4942:  # BIN
            bin_chunk = raw[off + 8:off + 8 + chunk_len]
        off += 8 + chunk_len
        if off >= len(raw):
            break
    accessors = gltf.get("accessors", [])
    buffer_views = gltf.get("bufferViews", [])
    world = _scene_node_matrices(gltf)
    node_of_mesh: dict[int, int] = {}
    for idx, node in enumerate(gltf.get("nodes", [])):
        if "mesh" in node:
            node_of_mesh[int(node["mesh"])] = idx
    clouds: list[np.ndarray] = []
    n_prims = 0
    for mesh_idx, mesh in enumerate(gltf.get("meshes", [])):
        mat = world.get(node_of_mesh.get(mesh_idx, -1), np.eye(4))
        for prim in mesh.get("primitives", []):
            pos = _read_accessor(accessors, buffer_views, bin_chunk,
                                 prim["attributes"]["POSITION"])
            if not np.allclose(mat, np.eye(4)):
                pos = (mat[:3, :3] @ pos.T).T + mat[:3, 3]
            clouds.append(pos)
            n_prims += 1
    if not clouds:
        raise ValueError(f"{path} contains no POSITION data")
    verts = np.vstack(clouds)
    meta = {"n_vertices": int(verts.shape[0]), "n_primitives": n_prims,
            "n_meshes": len(gltf.get("meshes", []))}
    return verts, meta


# ---------------------------------------------------------------------------
# Heightfield -> buildings
# ---------------------------------------------------------------------------

def build_heightfield(verts: np.ndarray) -> np.ndarray:
    """Upper-envelope heightfield of GLB vertices, 2 m cells.

    GLB frame: x east from bbox centre, y up, z south. Slice frame:
    x east 0..W, y north 0..H. Vertex (x, y, z) maps to cell
    (iy = floor((H/2 - z)/res), ix = floor((x + W/2)/res)) and the
    cell height is the maximum vertex y landing in it.
    """
    hf = np.full((HF_N, HF_N), np.nan, dtype=np.float64)
    sx = verts[:, 0] + EXTENT_M / 2.0          # east 0..W
    sy = EXTENT_M / 2.0 - verts[:, 2]          # north: z is south
    ix = np.clip((sx / HF_RES_M).astype(np.int64), 0, HF_N - 1)
    iy = np.clip((sy / HF_RES_M).astype(np.int64), 0, HF_N - 1)
    flat = iy * HF_N + ix
    h = verts[:, 1]
    order = np.argsort(h)  # ascending: later writes overwrite with higher
    hf.flat[flat[order]] = h[order]
    return hf


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labelling (0 = background)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    q: deque[tuple[int, int]] = deque()
    for iy in range(mask.shape[0]):
        for ix in range(mask.shape[1]):
            if not mask[iy, ix] or labels[iy, ix] != 0:
                continue
            n += 1
            labels[iy, ix] = n
            q.append((iy, ix))
            while q:
                cy, cx = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                            and mask[ny, nx] and labels[ny, nx] == 0):
                        labels[ny, nx] = n
                        q.append((ny, nx))
    return labels, n


_ERODE_CELLS = 1


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    """One cell of 4-neighbour erosion (removes 2 m-wide bridges)."""
    up = np.zeros_like(mask)
    up[1:, :] = mask[:-1, :]
    down = np.zeros_like(mask)
    down[:-1, :] = mask[1:, :]
    left = np.zeros_like(mask)
    left[:, 1:] = mask[:, :-1]
    right = np.zeros_like(mask)
    right[:, :-1] = mask[:, 1:]
    return mask & up & down & left & right


def derive_buildings(hf: np.ndarray) -> tuple[list[dict], dict]:
    """Connected components above low-percentile ground become boxes.

    Mask cells more than 3 m above the low-percentile ground datum are
    eroded by one cell before labelling so narrow linkway bridges do not
    merge distinct structures into one component; boxes are then grown
    back by one cell. Components smaller than 4 cells are speckle.
    """
    finite = hf[~np.isnan(hf)]
    ground = float(np.percentile(finite, 10.0)) if finite.size else 0.0
    mask = hf > (ground + BUILDING_MIN_AGL_M)
    core = mask
    for _ in range(_ERODE_CELLS):
        core = _erode_mask(core)
    labels, n = connected_components(core)
    grow = _ERODE_CELLS * HF_RES_M
    buildings: list[dict] = []
    for comp in range(1, n + 1):
        ys, xs = np.nonzero(labels == comp)
        if ys.size < 4:  # speckle (tiny noise) is not a building
            continue
        heights = hf[ys, xs]
        heights = heights[~np.isnan(heights)]
        height_m = float(np.percentile(heights, 95.0) - ground)
        if height_m < BUILDING_MIN_AGL_M:
            continue
        cx = float((xs * HF_RES_M + HF_RES_M / 2.0).mean())
        cy = float((ys * HF_RES_M + HF_RES_M / 2.0).mean())
        w = float((xs.max() - xs.min() + 1) * HF_RES_M + 2.0 * grow)
        d = float((ys.max() - ys.min() + 1) * HF_RES_M + 2.0 * grow)
        buildings.append({
            "id": len(buildings),
            "cx": min(max(cx, 0.0), EXTENT_M),
            "cy": min(max(cy, 0.0), EXTENT_M),
            "w": float(min(w, EXTENT_M)), "d": float(min(d, EXTENT_M)),
            "height_m": max(height_m, 0.0),
            "storeys": None,
            "void_deck": False,
            "linkways": [],
        })
    info = {"ground_datum_m": ground, "n_components": n,
            "mask_cells": int(mask.sum()), "erode_cells": _ERODE_CELLS}
    return buildings, info


# ---------------------------------------------------------------------------
# DEM from terrain.tif
# ---------------------------------------------------------------------------

def _box_blur(dem: np.ndarray, radius: int = 2) -> np.ndarray:
    """Small square-kernel blur (edge-replicated), numpy only."""
    out = dem.astype(np.float64)
    for _ in range(radius):
        padded = np.pad(out, 1, mode="edge")
        out = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
               padded[1:-1, :-2] + padded[1:-1, 1:-1] +
               padded[1:-1, 2:] + padded[2:, :-2] + padded[2:, 1:-1] +
               padded[2:, 2:]) / 9.0
    return out


def load_terrain_dem(path: Path, bbox: dict) -> tuple[np.ndarray, dict]:
    """DEM cropped to the manifest bbox, resampled to 200 x 200.

    Uses the GeoTIFF geotransform (Web-Mercator tags 33550/33922) when
    present so the raster aligns with the OSM / mesh slice frame;
    otherwise the whole raster is resampled. Datum: lowest cell = 0 m.
    """
    from PIL import Image

    with Image.open(path) as im:
        arr = np.array(im).astype(np.float64)
        try:
            tags = im.tag_v2
            scale_vals = tags[33550]
            scale_x, scale_y = float(scale_vals[0]), float(scale_vals[1])
            tie_vals = tags[33922]
            (tie_x, tie_y) = (float(tie_vals[3]), float(tie_vals[4]))
        except Exception:
            tags = None
            scale_x = scale_y = tie_x = tie_y = 0.0
    if arr.ndim != 2:
        arr = arr[..., 0]
    meta = {"source_size": [int(arr.shape[1]), int(arr.shape[0])],
            "cropped": False}

    if tags is not None and scale_x > 0 and scale_y > 0:
        radius = 6378137.0

        def merc(lat: float, lon: float) -> tuple[float, float]:
            phi = math.radians(lat)
            return (radius * math.radians(lon),
                    radius * math.log(math.tan(math.pi / 4.0 + phi / 2.0)))

        west, south = merc(bbox["min_lat"], bbox["min_lon"])
        east, north = merc(bbox["max_lat"], bbox["max_lon"])
        h_px, w_px = arr.shape
        # Output cell centres, west->east and south->north.
        xs = (np.arange(DEM_N) + 0.5) / DEM_N * (east - west) + west
        ys = (np.arange(DEM_N) + 0.5) / DEM_N * (north - south) + south
        col = (xs - tie_x) / scale_x - 0.5
        row = (tie_y - ys) / scale_y - 0.5
        c0 = np.clip(np.floor(col).astype(np.int64), 0, w_px - 2)
        r0 = np.clip(np.floor(row).astype(np.int64), 0, h_px - 2)
        fx = np.clip(col - c0, 0.0, 1.0)
        fy = np.clip(row - r0, 0.0, 1.0)
        c1 = c0 + 1
        r1 = r0 + 1
        # Bilinear sample: dem[iy] is elevation at ys[iy], so row 0 is the
        # south edge, matching slice index [iy, ix] (iy = north).
        c0g = (1 - fx) * arr[np.ix_(r0, c0)] + fx * arr[np.ix_(r0, c1)]
        c1g = (1 - fx) * arr[np.ix_(r1, c0)] + fx * arr[np.ix_(r1, c1)]
        dem = (1 - fy[:, None]) * c0g + fy[:, None] * c1g
        meta["cropped"] = True
    else:
        h_px, w_px = arr.shape
        src_y = (np.arange(DEM_N) + 0.5) * h_px / DEM_N - 0.5
        src_x = (np.arange(DEM_N) + 0.5) * w_px / DEM_N - 0.5
        y0 = np.clip(np.floor(src_y).astype(np.int64), 0, h_px - 2)
        x0 = np.clip(np.floor(src_x).astype(np.int64), 0, w_px - 2)
        fy = (src_y - y0)[:, None]
        fx = (src_x - x0)[None, :]
        top = (1 - fx) * arr[np.ix_(y0, x0)] + fx * arr[np.ix_(y0, x0 + 1)]
        bot = (1 - fx) * arr[np.ix_(y0 + 1, x0)] + fx * arr[np.ix_(y0 + 1, x0 + 1)]
        dem = ((1 - fy) * top + fy * bot)[::-1, :]

    dem = _box_blur(dem, radius=2)
    dem = dem - dem.min()
    return dem.astype(np.float32), meta


# ---------------------------------------------------------------------------
# Ground graph from map.osm
# ---------------------------------------------------------------------------

def _dem_at(dem: np.ndarray, x: float, y: float) -> float:
    ix = int(np.clip(round(x), 0, dem.shape[1] - 1))
    iy = int(np.clip(round(y), 0, dem.shape[0] - 1))
    return float(dem[iy, ix])


def read_osm_graph(path: Path, bbox: dict,
                   dem: np.ndarray) -> tuple[list[dict], list[dict], dict]:
    """OSM pedestrian ways -> ground graph (nodes kind = junction)."""
    min_lat, max_lat = bbox["min_lat"], bbox["max_lat"]
    min_lon, max_lon = bbox["min_lon"], bbox["max_lon"]

    def to_local(lat: float, lon: float) -> tuple[float, float]:
        x = (lon - min_lon) / (max_lon - min_lon) * EXTENT_M
        y = (lat - min_lat) / (max_lat - min_lat) * EXTENT_M
        return x, y

    tree = ET.parse(str(path))
    root = tree.getroot()
    osm_nodes: dict[int, tuple[float, float]] = {}
    for node in root.iter("node"):
        try:
            osm_id = int(node.get("id"))
            lat = float(node.get("lat"))
            lon = float(node.get("lon"))
        except (TypeError, ValueError):
            continue
        x, y = to_local(lat, lon)
        if 0.0 <= x < EXTENT_M and 0.0 <= y < EXTENT_M:
            osm_nodes[osm_id] = (x, y)

    nodes: list[dict] = []
    for osm_id in sorted(osm_nodes):
        x, y = osm_nodes[osm_id]
        nodes.append({"id": osm_id, "x": float(x), "y": float(y),
                      "z": _dem_at(dem, x, y), "kind": "junction"})
    by_id = {n["id"]: n for n in nodes}

    edges: list[dict] = []
    seen: set[tuple[int, int]] = set()
    n_ways = 0
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        highway = tags.get("highway")
        if highway not in PEDESTRIAN_HIGHWAYS:
            continue
        n_ways += 1
        refs = []
        for nd in way.iter("nd"):
            try:
                refs.append(int(nd.get("ref")))
            except (TypeError, ValueError):
                continue
        width = WIDTH_BY_HIGHWAY.get(str(highway), 2.0)
        sheltered = str(tags.get("covered", "")).lower() == "yes"
        has_ramp = str(tags.get("ramp", "")).lower() == "yes"
        for u, v in zip(refs[:-1], refs[1:]):
            if u == v or u not in by_id or v not in by_id:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            a, b = by_id[u], by_id[v]
            length = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            if length <= 0:
                continue
            edges.append({"u": u, "v": v, "length_m": float(length),
                          "slope": float(abs(b["z"] - a["z"]) / length),
                          "width_m": float(width),
                          "sheltered": bool(sheltered),
                          "has_ramp": bool(has_ramp)})
    info = {"osm_nodes_in_bbox": len(osm_nodes), "osm_ways_used": n_ways}
    return nodes, edges, info


# ---------------------------------------------------------------------------
# Landing pads + streetview audit points
# ---------------------------------------------------------------------------

def largest_open_cell(buildings: list[dict], dem: np.ndarray) -> tuple[float, float]:
    """Ground cell farthest from building footprints (multi-source BFS)."""
    mask = np.zeros((HF_N, HF_N), dtype=bool)
    for b in buildings:
        x0 = int(np.clip((b["cx"] - b["w"] / 2.0) / HF_RES_M, 0, HF_N - 1))
        x1 = int(np.clip((b["cx"] + b["w"] / 2.0) / HF_RES_M, 0, HF_N - 1))
        y0 = int(np.clip((b["cy"] - b["d"] / 2.0) / HF_RES_M, 0, HF_N - 1))
        y1 = int(np.clip((b["cy"] + b["d"] / 2.0) / HF_RES_M, 0, HF_N - 1))
        mask[y0:y1 + 1, x0:x1 + 1] = True
    dist = np.where(mask, 0, -1).astype(np.int32)
    q: deque[tuple[int, int]] = deque()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        q.append((int(y), int(x)))
    best = (HF_N // 2, HF_N // 2)
    while q:
        cy, cx = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < HF_N and 0 <= nx < HF_N
                    and dist[ny, nx] < 0):
                dist[ny, nx] = dist[cy, cx] + 1
                q.append((ny, nx))
                if dist[ny, nx] > dist[best]:
                    best = (ny, nx)
    by, bx = best
    return (bx * HF_RES_M + HF_RES_M / 2.0, by * HF_RES_M + HF_RES_M / 2.0)


def build_landing_pads(buildings: list[dict],
                       dem: np.ndarray) -> list[dict]:
    """Rooftop pad per derived building + largest open ground cell."""
    pads: list[dict] = []
    pid = 0
    for b in buildings:
        pads.append({"id": pid, "x": float(b["cx"]), "y": float(b["cy"]),
                     "z": float(b["height_m"]), "kind": "rooftop",
                     "building_id": int(b["id"]), "size_m": 3.0})
        pid += 1
    gx, gy = largest_open_cell(buildings, dem)
    pads.append({"id": pid, "x": float(gx), "y": float(gy),
                 "z": _dem_at(dem, gx, gy), "kind": "ground",
                 "building_id": -1, "size_m": 4.0})
    return pads


def read_streetview(capture_dir: Path, bbox: dict) -> list[dict]:
    """Audit points from streetview/photos.geojson (slice coords)."""
    path = capture_dir / "streetview" / "photos.geojson"
    data = json.loads(path.read_text(encoding="utf-8"))
    min_lat, max_lat = bbox["min_lat"], bbox["max_lat"]
    min_lon, max_lon = bbox["min_lon"], bbox["max_lon"]
    points: list[dict] = []
    for feat in data.get("features", []) or []:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates") or [None, None]
        lon = props.get("lon", coords[0] if coords else None)
        lat = props.get("lat", coords[1] if len(coords) == 2 else None)
        if lat is None or lon is None:
            continue
        x = (float(lon) - min_lon) / (max_lon - min_lon) * EXTENT_M
        y = (float(lat) - min_lat) / (max_lat - min_lat) * EXTENT_M
        points.append({
            "x": float(np.clip(x, 0.0, EXTENT_M - 1e-6)),
            "y": float(np.clip(y, 0.0, EXTENT_M - 1e-6)),
            "lat": float(lat), "lon": float(lon),
            "file": props.get("filename"),
            "date": props.get("imagery_date"),
            "verified": False,
        })
    return points


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LABEL = ("REAL Aleph capture — research use; Google-derived layers "
         "(mesh, satellite, Street View) are research-only and not "
         "redistributable")


def build_real_slice(capture_dir: str | Path) -> dict:
    """Assemble the fused slice from one Aleph capture directory."""
    capture_dir = Path(capture_dir)
    info = _read_manifest(capture_dir)
    bbox = info["bbox"]

    verts, glb_meta = read_glb_vertices(capture_dir / "mesh.glb")
    hf = build_heightfield(verts)
    buildings, hf_info = derive_buildings(hf)

    dem, dem_info = load_terrain_dem(capture_dir / "terrain.tif", bbox)
    nodes, edges, osm_info = read_osm_graph(capture_dir / "map.osm", bbox, dem)
    pads = build_landing_pads(buildings, dem)
    audit = read_streetview(capture_dir, bbox)

    mid_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2.0
    width_m = (bbox["max_lon"] - bbox["min_lon"]) * 111320.0 * math.cos(
        math.radians(mid_lat))
    height_m = (bbox["max_lat"] - bbox["min_lat"]) * 110540.0

    meta = {
        "label": LABEL,
        "capture": info["name"],
        "bbox": {"min_lat": bbox["min_lat"], "max_lat": bbox["max_lat"],
                 "min_lon": bbox["min_lon"], "max_lon": bbox["max_lon"]},
        "bbox_m": {"width_east_m": float(width_m),
                   "height_north_m": float(height_m)},
        "extent_m": EXTENT_M,
        "dem_res_m": DEM_RES_M,
        "dem_shape": [int(dem.shape[0]), int(dem.shape[1])],
        "dem_min_m": float(dem.min()),
        "dem_max_m": float(dem.max()),
        "dem_mean_m": float(dem.mean()),
        "heightfield_res_m": HF_RES_M,
        "n_vertices": glb_meta["n_vertices"],
        "n_primitives": glb_meta["n_primitives"],
        "n_components": hf_info["n_components"],
        "n_buildings": len(buildings),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_pads": len(pads),
        "n_photos": len(audit),
        "osm_nodes_in_bbox": osm_info["osm_nodes_in_bbox"],
        "osm_ways_used": osm_info["osm_ways_used"],
        "sources": {
            "buildings": "mesh.glb heightfield (connected components)",
            "dem": "terrain.tif (bbox-cropped, resampled, smoothed)",
            "ground_graph": "map.osm pedestrian ways (footway, path, "
                            "pedestrian, residential, service, cycleway, "
                            "steps, living_street)",
            "audit_points": "streetview/photos.geojson",
        },
    }
    return {"dem": dem, "dem_res_m": DEM_RES_M,
            "extent": EXTENT_M, "extent_m": EXTENT_M,
            "buildings": buildings, "nodes": nodes, "edges": edges,
            "ramps": [], "landing_pads": pads,
            "streetview": audit, "audit_points": audit,
            "meta": meta}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build a real Aleph slice")
    ap.add_argument("capture_dir")
    args = ap.parse_args()
    sl = build_real_slice(args.capture_dir)
    print("slice meta:", sl["meta"])
    print("dem shape:", sl["dem"].shape,
          "min %.2f max %.2f" % (sl["dem"].min(), sl["dem"].max()))
    for b in sl["buildings"][:5]:
        print("building", b)
    print("nodes", len(sl["nodes"]), "edges", len(sl["edges"]),
          "pads", len(sl["landing_pads"]), "photos", len(sl["streetview"]))


