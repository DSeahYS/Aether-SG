"""AETHER-SG synthetic 200x200 m neighborhood slice (SIMULATED DATA).

This module generates a deterministic, seeded synthetic slice that mimics a
Toa Payoh-like HDB estate.  It is NOT real Aleph/OneMap data and must be
labelled simulated in any demo/UI.  See Research-AETHER-SG.md section 11:
the MVP ships as a pre-cached replay over synthetic geometry, never as
live captured data.

Coordinate frame
----------------
Metres, origin at the SW corner of the slice, x east, y north, z up AGL
(0 = ground DEM surface, not AMSL).  Slice extent is 200 x 200 m.

Primary API (for routing / evolve agents)
-----------------------------------------
build_slice(seed=7) -> dict
    Returns a dict with keys:
      "dem"           : float32 ndarray (H, W) @ 1 m, index [iy, ix]
      "dem_res_m"     : float 1.0
      "extent_m"      : float 200.0
      "buildings"     : list[dict] axis-aligned boxes
      "nodes"         : list[dict] ground-graph nodes
      "edges"         : list[dict] ground-graph edges
      "ramps"         : list[dict] ramp nodes (subset of nodes, kind="ramp")
      "landing_pads"  : list[dict] drone landing pads (rooftop + open court)
      "meta"          : dict with seed, labels, extents, counts
save_slice(slice_data, path)  -> None   (npz + sidecar json)
load_slice(path)              -> dict   (reads npz + sidecar json)

Ground-graph edge attributes: u, v (node ids), length_m, slope (rise/run),
width_m, sheltered (bool, linkway/covered), has_ramp (bool).

Building attributes: id, cx, cy, w, d (footprint centre/size m),
height_m, storeys, void_deck (bool), linkways (list of building ids).

Ramp attributes: node id, x, y, verified (bool, True = human-verified),
slope, width_m.  Unverified ramps must be treated as advisory only
(see Research section 7: VLM triage != verification).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXTENT_M = 200.0
DEM_RES_M = 1.0
N = int(EXTENT_M / DEM_RES_M)  # 200 cells per side

# Fixed jitter-free base layout for 14 HDB blocks in a 4x4 grid,
# one slot omitted (open court) and one slot as drain-side car park.
_BASE_BLOCKS = [
    # cx, cy, w, d, height_m, storeys
    (30, 30, 24, 12, 30, 10), (80, 30, 24, 12, 45, 15),
    (130, 30, 24, 12, 15, 5), (185, 30, 22, 11, 40, 13),
    (30, 85, 24, 12, 60, 20), (80, 85, 24, 12, 25, 8),
    (130, 85, 24, 12, 50, 16), (185, 85, 22, 11, 20, 7),
    (30, 140, 24, 12, 35, 12), (80, 140, 24, 12, 55, 18),
    # (130, 140) is the open court / landing-pad plaza (no building)
    (185, 140, 22, 11, 30, 10), (30, 185, 24, 12, 45, 15),
    (80, 185, 24, 12, 40, 13), (130, 185, 24, 12, 60, 20),
    (185, 185, 22, 11, 35, 12),
]


def _build_dem(rng: np.random.Generator) -> np.ndarray:
    """Gentle NW->SE slope + drain channel along the south edge."""
    ys, xs = np.mgrid[0:N, 0:N].astype(np.float64)
    # Gentle slope: higher in the NW, lower in the SE (metres over 200 m).
    dem = 2.0 - 0.006 * xs + 0.006 * (ys - (N - 1))
    # Drain channel along the south edge: Gaussian trough ~2 m deep.
    dem -= 2.0 * np.exp(-((ys - 4.0) ** 2) / (2 * 6.0**2))
    # Gentle dish in the open-court plaza (water ponds here first).
    dem -= 0.4 * np.exp(-(((xs - 130) ** 2 + (ys - 140) ** 2)) / (2 * 18.0**2))
    # Small deterministic survey noise (NOT a substitute for real DEM).
    dem += rng.normal(0, 0.05, size=dem.shape)
    # AGL datum: shift so the lowest cell is 0 m (ground z >= 0 everywhere,
    # keeping score_path's 0-60 m altitude reject valid for ground paths).
    dem -= dem.min()
    return dem.astype(np.float32)


def _build_buildings(rng: np.random.Generator) -> list[dict]:
    """14 axis-aligned HDB-like boxes with void decks + linkways."""
    buildings: list[dict] = []
    for i, (cx, cy, w, d, h, st) in enumerate(_BASE_BLOCKS):
        jx, jy = float(rng.normal(0, 1.0)), float(rng.normal(0, 1.0))
        buildings.append({
            "id": i,
            "cx": cx + jx, "cy": cy + jy,
            "w": float(w), "d": float(d),
            "height_m": float(h), "storeys": int(st),
            "void_deck": bool(i % 3 == 0),   # ground floor open void deck
            "linkways": [],                  # filled below
        })
    # Linkways: sheltered connectors between horizontally/vertically
    # adjacent blocks in the base grid (by centre proximity).
    for a in buildings:
        for b in buildings:
            if a["id"] >= b["id"]:
                continue
            dist = abs(a["cx"] - b["cx"]) + abs(a["cy"] - b["cy"])
            if dist <= 55.0:
                a["linkways"].append(b["id"])
                b["linkways"].append(a["id"])
    return buildings


def _nearest_dem(dem: np.ndarray, x: float, y: float) -> float:
    ix = int(np.clip(round(x), 0, N - 1))
    iy = int(np.clip(round(y), 0, N - 1))
    return float(dem[iy, ix])


def _build_graph(dem: np.ndarray, buildings: list[dict],
                 rng: np.random.Generator) -> tuple[list[dict], list[dict], list[dict]]:
    """Grid junctions + ~20 ramp nodes + building entrances, edges between."""
    nodes: list[dict] = []
    nid = 0
    # 5x5 junction lattice every 50 m (avoid building footprints).
    for jy in range(5):
        for ix in range(5):
            x, y = 25.0 + 50.0 * ix, 25.0 + 50.0 * jy
            inside = any(abs(x - b["cx"]) < b["w"] / 2 and abs(y - b["cy"]) < b["d"] / 2
                         for b in buildings)
            if inside:
                continue
            nodes.append({"id": nid, "x": x, "y": y,
                          "z": _nearest_dem(dem, x, y), "kind": "junction"})
            nid += 1
    junction_ids = [n["id"] for n in nodes]

    # Building entrance nodes (grade-level assembly, defend-in-place
    # exterior -- see Research section 7; no in-building refuge).
    for b in buildings:
        x, y = b["cx"], b["cy"] - b["d"] / 2 - 4.0
        nodes.append({"id": nid, "x": float(x), "y": float(y),
                      "z": _nearest_dem(dem, x, y), "kind": "entrance",
                      "building_id": b["id"]})
        nid += 1

    # ~20 ramp nodes along the lattice + at entrances. ~half verified.
    ramps: list[dict] = []
    ramp_spots = []
    for _ in range(20):
        base = nodes[int(rng.integers(0, len(junction_ids)))]
        x = float(np.clip(base["x"] + rng.uniform(-15, 15), 2, 198))
        y = float(np.clip(base["y"] + rng.uniform(-15, 15), 2, 198))
        ramp_spots.append((x, y))
    for k, (x, y) in enumerate(ramp_spots):
        rec = {"id": nid, "x": x, "y": y, "z": _nearest_dem(dem, x, y),
               "kind": "ramp",
               "verified": bool(k % 2 == 0),          # 10 verified / 10 advisory
               "slope": float(rng.uniform(0.03, 0.09)),  # ~1:12..1:33
               "width_m": float(rng.uniform(1.2, 1.8))}
        nodes.append(rec)
        ramps.append(rec)
        nid += 1

    # Edges: connect each junction to its nearest neighbours (grid-ish)
    # plus every ramp/entrance to its nearest junction.
    def nearest_node(x: float, y: float, pool: list[dict]) -> dict:
        return min(pool, key=lambda n: (n["x"] - x) ** 2 + (n["y"] - y) ** 2)

    jnodes = [n for n in nodes if n["kind"] == "junction"]
    edges: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for a in jnodes:
        nbrs = sorted((n for n in jnodes if n["id"] != a["id"]),
                      key=lambda n: (n["x"] - a["x"]) ** 2 + (n["y"] - a["y"]) ** 2)[:3]
        for b in nbrs:
            key = tuple(sorted((a["id"], b["id"])))
            if key in seen:
                continue
            seen.add(key)
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            length = float(np.hypot(dx, dy))
            edges.append({"u": a["id"], "v": b["id"], "length_m": length,
                          "slope": abs(b["z"] - a["z"]) / max(length, 1e-6),
                          "width_m": 2.0,
                          "sheltered": bool(abs(dy) < 5.0),  # E-W covered linkway-ish
                          "has_ramp": False})
    for n in nodes:
        if n["kind"] in ("ramp", "entrance"):
            b = nearest_node(n["x"], n["y"], jnodes)
            key = tuple(sorted((n["id"], b["id"])))
            if key in seen:
                continue
            seen.add(key)
            length = float(np.hypot(n["x"] - b["x"], n["y"] - b["y"]))
            edges.append({"u": n["id"], "v": b["id"], "length_m": length,
                          "slope": abs(n["z"] - b["z"]) / max(length, 1e-6),
                          "width_m": n.get("width_m", 1.5),
                          "sheltered": bool(n["kind"] == "ramp"),
                          "has_ramp": bool(n["kind"] == "ramp")})
    return nodes, edges, ramps


def _build_pads(buildings: list[dict]) -> list[dict]:
    """Rooftop pads (one per tall block) + open-court pads. Advisory only."""
    pads: list[dict] = []
    pid = 0
    for b in buildings:
        if b["height_m"] >= 25.0:
            pads.append({"id": pid, "x": b["cx"], "y": b["cy"],
                         "z": b["height_m"], "kind": "rooftop",
                         "building_id": b["id"], "size_m": 3.0})
            pid += 1
    for x, y in [(130.0, 140.0), (105.0, 140.0), (155.0, 140.0), (130.0, 120.0)]:
        pads.append({"id": pid, "x": x, "y": y, "z": 0.0,
                     "kind": "courtyard", "building_id": -1, "size_m": 4.0})
        pid += 1
    return pads


def build_slice(seed: int = 7) -> dict:
    """Build the deterministic synthetic slice. See module docstring."""
    rng = np.random.default_rng(int(seed))
    dem = _build_dem(rng)
    buildings = _build_buildings(rng)
    nodes, edges, ramps = _build_graph(dem, buildings, rng)
    pads = _build_pads(buildings)
    meta = {
        "seed": int(seed),
        "label": "SIMULATED (synthetic, not real Aleph/OneMap data)",
        "extent_m": EXTENT_M, "dem_res_m": DEM_RES_M,
        "n_buildings": len(buildings), "n_nodes": len(nodes),
        "n_edges": len(edges), "n_ramps": len(ramps),
        "n_ramps_verified": sum(1 for r in ramps if r["verified"]),
        "n_pads": len(pads),
        "z_max_m": 60.0,
    }
    return {"dem": dem, "dem_res_m": DEM_RES_M, "extent_m": EXTENT_M,
            "buildings": buildings, "nodes": nodes, "edges": edges,
            "ramps": ramps, "landing_pads": pads, "meta": meta}


def save_slice(slice_data: dict, path: str | Path) -> Path:
    """Persist DEM (+SDF-ready buildings) as npz and the rest as json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")
    np.savez_compressed(npz_path, dem=slice_data["dem"])
    sidecar = {k: v for k, v in slice_data.items() if k != "dem"}
    json_path.write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
    return npz_path


def load_slice(path: str | Path) -> dict:
    """Inverse of save_slice: npz DEM + json geometry/graph."""
    path = Path(path)
    z = np.load(path.with_suffix(".npz"))
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return {**sidecar, "dem": z["dem"]}


if __name__ == "__main__":
    s = build_slice(seed=7)
    print("slice meta:", s["meta"])
    print("dem shape:", s["dem"].shape, "min %.2f max %.2f" %
          (s["dem"].min(), s["dem"].max()))
