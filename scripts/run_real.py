"""AETHER-SG real-capture pipeline.

Rebuilds the viewer bundle from the REAL Aleph capture in
``captures/aleph-20261119T022946Z-1upsr2v9`` (mesh.glb, map.osm,
terrain.tif, satellite.png, streetview photos):

  real slice (real_slice.py) -> evaluator SDF + GNSS shadow + flood
  screening tensor -> local evolution over heuristic weights on the
  real slice (gens 5, pop 8) -> Gen0 / Gen25 / final drone routes
  (0-60 m) + wheelchair ground route SW->NE on the OSM graph ->
  viewer/capture/capture.json EXTENDED with slice-derived blocks.

The viewer bundle keeps the existing capture keys (name, polygon,
bbox, mesh, satellite, photos) and adds: buildings, dem grid,
gnss grid, dronePaths, groundPath, metrics, evolution, ground,
slice, stats.

Drone envelope 0-60 m. Flood is a screening tensor (PUB owns
operational hydrology). Routes are advisory, not for navigation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aether import evaluator, routing  # noqa: E402
from aether.evolve import (DEFAULT_WEIGHTS, crossover, mutate,  # noqa: E402
                           random_weights)
from aether.real_slice import build_real_slice  # noqa: E402

try:  # optional extended metrics; absence is fine (real-only pipeline)
    from aether import metrics as _aether_metrics  # type: ignore  # noqa: E402
except Exception:  # ModuleNotFoundError or import-time failure
    _aether_metrics = None  # type: ignore

CAPTURE_DIR = ROOT / "captures" / "aleph-20261119T022946Z-1upsr2v9"
OUT_PATH = ROOT / "viewer" / "capture" / "capture.json"

TRUE_TIME_W = 1.0
TRUE_ENERGY_W = 0.30
TRUE_SHADOW_W = 0.40
TRUE_WIND_W = 0.20
GUIDE_GENS = 5
GUIDE_POP = 8
GUIDE_SEED = 42

BLIND_WEIGHTS = {"w_time": 1.0, "w_energy": 0.0, "w_shadow": 0.0,
                 "w_wind": 0.0, "w_detour": 0.0}


def cx(v: float) -> float:
    """Slice coords 0..200 -> viewer coords -100..100."""
    return float(v) - 100.0


def downsample(grid: np.ndarray, n: int = 41) -> list:
    idx_y = np.linspace(0, grid.shape[0] - 1, n).astype(int)
    idx_x = np.linspace(0, grid.shape[1] - 1, n).astype(int)
    return grid[np.ix_(idx_y, idx_x)].astype(float).round(3).tolist()


def passable_components(nodes, edges) -> list[list[int]]:
    """Connected components using only wheelchair-passable edges
    (slope <= 1/12, width >= 1.2 m; screening flood checked at routing)."""
    adj: dict[int, list[int]] = {}
    for e in edges:
        if e["slope"] > 1.0 / 12.0 or e["width_m"] < 1.2:
            continue
        adj.setdefault(e["u"], []).append(e["v"])
        adj.setdefault(e["v"], []).append(e["u"])
    components: list[list[int]] = []
    seen: set[int] = set()
    for n in nodes:
        start = n["id"]
        if start in seen or start not in adj:
            continue
        seen.add(start)
        stack, comp = [start], [start]
        while stack:
            cur = stack.pop()
            for nb in adj.get(cur, []):
                if nb not in seen:
                    seen.add(nb)
                    comp.append(nb)
                    stack.append(nb)
        components.append(comp)
    return components


def pick_swne_pair(nodes, edges) -> tuple[int, int, list[int]]:
    """SW->NE endpoints: the passable component whose extreme nodes are
    jointly closest to the SW (20,20) / NE (180,180) slice corners."""
    by_id = {n["id"]: n for n in nodes}
    best = None
    for comp in passable_components(nodes, edges):
        if len(comp) < 2:
            continue
        pool = [by_id[i] for i in comp]
        a = min(pool, key=lambda n: (n["x"] - 20.0) ** 2 + (n["y"] - 20.0) ** 2)
        b = min(pool, key=lambda n: (n["x"] - 180.0) ** 2 + (n["y"] - 180.0) ** 2)
        total = math.hypot(a["x"] - 20.0, a["y"] - 20.0) + math.hypot(
            b["x"] - 180.0, b["y"] - 180.0)
        key = (total, -len(comp))
        if best is None or key < best[0]:
            best = (key, a, b, comp)
    if best is None:
        raise ValueError("no wheelchair-passable component with >= 2 nodes")
    _key, a, b, comp = best
    return int(a["id"]), int(b["id"]), comp


def nearest_node(nodes, tx: float, ty: float,
                 pool_ids: set[int]) -> dict:
    pool = [n for n in nodes if n["id"] in pool_ids]
    return min(pool, key=lambda n: (n["x"] - tx) ** 2 + (n["y"] - ty) ** 2)


def nearest_pad(pads, tx: float, ty: float) -> dict:
    return min(pads, key=lambda p: (p["x"] - tx) ** 2 + (p["y"] - ty) ** 2)


def score_on_slice(weights, s, layers, drone_start, drone_goal,
                   g_start, g_goal, flood_depth) -> float:
    """TRUE scalar on the real slice (maximise; heavily negative = infeasible)."""
    g = routing.ground_astar(s, g_start, g_goal, weights=weights,
                             flood_tensor=flood_depth)
    d = routing.drone_astar(s, drone_start, goal_xyz=drone_goal,
                            weights=weights, sdf=layers["sdf"],
                            gnss_shadow=layers["gnss"], wind=layers["wind"])
    score = 0.0
    if g.success:
        score -= float(g.time_s) * TRUE_TIME_W
        score -= float(g.length_m) * 0.01
    else:
        score -= 800.0
    if d.success:
        score -= float(d.time_s) * TRUE_TIME_W
        score -= float(d.energy) * TRUE_ENERGY_W
        score -= float(d.shadow_exposure) * TRUE_SHADOW_W
        score -= float(d.wind_exposure) * TRUE_WIND_W
    else:
        score -= 1000.0
    return float(score)


def evolve_on_slice(s, layers, drone_start, drone_goal, g_start, g_goal,
                    flood_depth, gens=GUIDE_GENS, pop=GUIDE_POP,
                    seed=GUIDE_SEED):
    """Tournament evolution of heuristic weights on the real slice."""
    rng = random.Random(seed)
    gen0 = score_on_slice(DEFAULT_WEIGHTS, s, layers, drone_start,
                          drone_goal, g_start, g_goal, flood_depth)

    def make(weights):
        return {"weights": dict(weights),
                "score": score_on_slice(weights, s, layers, drone_start,
                                        drone_goal, g_start, g_goal,
                                        flood_depth)}

    population = [make(DEFAULT_WEIGHTS)]
    for _ in range(pop - 1):
        population.append(make(random_weights(rng)))
    history = []

    def record(gen):
        best = max(population, key=lambda ind: ind["score"])
        history.append({"gen": gen, "best_score": float(best["score"]),
                        "mean_score": float(np.mean(
                            [ind["score"] for ind in population])),
                        "best_weights": dict(best["weights"])})
        return best

    record(0)
    for gen in range(1, gens + 1):
        ranked = sorted(population, key=lambda ind: ind["score"], reverse=True)
        elites = ranked[:max(1, int(pop * 0.25))]
        new_pop = [{"weights": dict(e["weights"]), "score": e["score"]}
                   for e in elites]

        def tournament():
            cand = [population[rng.randrange(pop)] for _ in range(3)]
            return max(cand, key=lambda ind: ind["score"])

        while len(new_pop) < pop:
            child = crossover(tournament()["weights"],
                              tournament()["weights"], rng)
            child = mutate(child, rng, sigma=0.25)
            new_pop.append(make(child))
        population = new_pop
        record(gen)
    best = max(population, key=lambda ind: ind["score"])
    return {"history": history, "gen0": float(gen0),
            "gen_intermediate": float(history[gens // 2]["best_score"]),
            "gen_final": float(best["score"]),
            "best_weights": dict(best["weights"])}


def route_metric(result, path_xyz, sdf_bundle, shadow_bundle, dem):
    score = -(result.time_s + 0.3 * result.energy
              + 0.4 * result.shadow_exposure
              + 0.2 * result.wind_exposure)
    ev_score = evaluator.score_path(np.asarray(path_xyz, dtype=float),
                                    sdf_bundle, shadow_bundle, dem,
                                    flood=None, ground=False)
    return {"score": float(score), "time": float(result.time_s),
            "collisions": 0 if result.min_sdf_m >= 0 else 1,
            "length_m": float(result.length_m),
            "energy": float(result.energy),
            "shadow_exposure": float(result.shadow_exposure),
            "wind_exposure": float(result.wind_exposure),
            "min_sdf_m": float(result.min_sdf_m),
            "evaluator_score": (None if not math.isfinite(ev_score)
                                else float(ev_score))}


# ---------------------------------------------------------------------------
# aether.metrics integration (defensive; real data only, no simulation).
#
# When ``aether.metrics`` is importable, its ``flight_metrics``,
# ``los_swath_frames``, environment and ground helpers are preferred.
# When it is absent, the ``_fallback_*`` helpers below derive the same
# capture.json blocks from the real slice / layers (altitude, AGL,
# wind and shadow along the flown path, flood screening, DEM stats).
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AETHER-SG real-capture pipeline")
    p.add_argument("--capture-dir", default=CAPTURE_DIR,
                   help="Aleph capture directory (default: %(default)s)")
    p.add_argument("--out", default=OUT_PATH,
                   help="viewer capture.json path (default: %(default)s)")
    return p.parse_args(argv)


def _to_viewer_point(x, y, z):
    """Slice coords (x east, y north, z alt) 0..200 -> viewer metres."""
    return [round(cx(float(x)), 1), round(float(z), 1),
            round(cx(float(y)), 1)]


def _viewer_path_from_slice(slice_path):
    return [_to_viewer_point(p[0], p[1], p[2]) for p in slice_path]


def _normalise_frames(raw):
    """Normalise los_swath_frames output (slice coords) to viewer.

    Accepts [[x, y, z], ...] or [{x, y, z} / {pos: [...]}, ...]
    in slice coords and returns [[x-100, alt, y-100], ...].
    """
    if not isinstance(raw, (list, tuple)):
        return []
    out = []
    for item in raw:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                out.append(_to_viewer_point(item[0], item[1], item[2]))
            elif isinstance(item, dict):
                if "x" in item and "y" in item and "z" in item:
                    pt = _to_viewer_point(item["x"], item["y"], item["z"])
                elif "pos" in item:
                    q = item["pos"]
                    pt = _to_viewer_point(q[0], q[1], q[2])
                elif "xyz" in item:
                    q = item["xyz"]
                    pt = _to_viewer_point(q[0], q[1], q[2])
                elif "point" in item:
                    q = item["point"]
                    pt = _to_viewer_point(q[0], q[1], q[2])
                else:
                    continue
                # Preserve recorded LOS/swath fields for the viewer.
                out.append({
                    "pos": pt,
                    "losClear": bool(item.get("los_clear", True)),
                    "blockedBy": (None if item.get("los_blocked_by") is None
                                  else int(item["los_blocked_by"])),
                    "losMinClearanceM": item.get("los_min_clearance_m"),
                    "swathRadiusM": item.get("swath_radius_m"),
                    "coveragePct": item.get("coverage_pct"),
                })
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _fallback_frames(slice_path, max_frames=60):
    viewer = _viewer_path_from_slice(slice_path)
    if len(viewer) <= max_frames:
        return viewer
    idx = np.linspace(0, len(viewer) - 1, max_frames).astype(int)
    seen = []
    for i in idx:
        if not seen or int(i) != seen[-1]:
            seen.append(int(i))
    return [viewer[i] for i in seen]


def _dem_at(dem, x, y):
    ix = int(np.clip(round(float(x)), 0, dem.shape[1] - 1))
    iy = int(np.clip(round(float(y)), 0, dem.shape[0] - 1))
    return float(dem[iy, ix])


def _fallback_flight_metrics(result, slice_path, dem, layers):
    """Real altitude / AGL / wind / shadow stats along the flown path."""
    arr = np.asarray(slice_path, dtype=float)
    if arr.shape[0] == 0:
        return {"avg_altitude_m": None, "min_altitude_m": None,
                "max_altitude_m": None, "altitude_range_m": None,
                "avg_speed_mps": None, "mean_wind_factor": None,
                "max_wind_factor": None, "min_agl_m": None,
                "mean_agl_m": None, "max_agl_m": None}
    alts = arr[:, 2]
    agls = np.array([float(alt) - _dem_at(dem, x, y)
                     for x, y, alt in arr], dtype=float)
    length = float(result.length_m)
    time_s = float(result.time_s) if float(result.time_s) > 0 else 0.0
    wind_vals = []
    try:
        wind = layers.get("wind") if isinstance(layers, dict) else None
        if wind is not None:
            res_xy = float(getattr(routing, "DRONE_RES_XY_M", 4.0))
            res_z = float(getattr(routing, "DRONE_RES_Z_M", 4.0))
            nx, ny, nz = wind.shape
            for x, y, z in arr:
                ix = int(np.clip(int(x // res_xy), 0, nx - 1))
                iy = int(np.clip(int(y // res_xy), 0, ny - 1))
                iz = int(np.clip(int(z // res_z), 0, nz - 1))
                wind_vals.append(float(wind[ix, iy, iz]))
    except Exception:
        wind_vals = []
    return {"avg_altitude_m": round(float(alts.mean()), 2),
            "min_altitude_m": round(float(alts.min()), 2),
            "max_altitude_m": round(float(alts.max()), 2),
            "altitude_range_m": round(float(alts.max() - alts.min()), 2),
            "avg_speed_mps": (round(length / time_s, 3)
                              if time_s > 0 else None),
            "mean_wind_factor": (round(float(np.mean(wind_vals)), 3)
                                 if wind_vals else None),
            "max_wind_factor": (round(float(np.max(wind_vals)), 3)
                                if wind_vals else None),
            "min_agl_m": round(float(agls.min()), 2),
            "mean_agl_m": round(float(agls.mean()), 2),
            "max_agl_m": round(float(agls.max()), 2)}


def flight_metrics_extra(result, slice_path, sdf_bundle, shadow_bundle,
                         dem, layers, slice_dict=None):
    """Extra per-gen flight metrics, merged over the real fallback.

    aether.metrics.flight_metrics(path, buildings, sdf, shadow, dem)
    is preferred; its fields are merged over the fallback so routing
    keys (wind / AGL) survive alongside the metrics breakdown.
    """
    fallback = _fallback_flight_metrics(result, slice_path, dem, layers)
    if _aether_metrics is not None:
        fn = getattr(_aether_metrics, "flight_metrics", None)
        if callable(fn):
            buildings = []
            if isinstance(slice_dict, dict):
                buildings = list(slice_dict.get("buildings", []) or [])
            attempts = (
                lambda: fn(np.asarray(slice_path, dtype=np.float64),
                           buildings, sdf_bundle, shadow_bundle, dem),
                lambda: fn(slice_path),
                lambda: fn(slice_path, result),
            )
            for att in attempts:
                try:
                    out = att()
                except TypeError:
                    continue
                except Exception as exc:
                    print(f"aether.metrics.flight_metrics failed: {exc}")
                    break
                if isinstance(out, dict) and out:
                    merged = dict(fallback)
                    for k, v in out.items():
                        merged[k] = v
                    return merged
    return fallback


def drone_frames_for(slice_path, slice_dict, layers, dem=None,
                   target_xy=None):
    """LOS swath frames in viewer coords ([x-100, alt, y-100]).

    aether.metrics.los_swath_frames(path, buildings, target, dem)
    yields one dict per path point; only its position is kept,
    converted slice (x, y, z) -> viewer [x-100, z, y-100].
    """
    if _aether_metrics is not None:
        fn = getattr(_aether_metrics, "los_swath_frames", None)
        if callable(fn):
            buildings = []
            if isinstance(slice_dict, dict):
                buildings = list(slice_dict.get("buildings", []) or [])
            path_arr = np.asarray(slice_path, dtype=np.float64)
            if target_xy is None and path_arr.shape[0]:
                target_xy = (float(path_arr[-1, 0]), float(path_arr[-1, 1]))
            dem_arr = dem
            if dem_arr is None and isinstance(slice_dict, dict):
                dem_arr = slice_dict.get("dem")
            attempts = (
                lambda: fn(path_arr, buildings, target_xy, dem_arr),
                lambda: fn(path_arr, buildings, target_xy, dem_arr, 60.0),
                lambda: fn(slice_path),
            )
            for att in attempts:
                try:
                    out = att()
                except TypeError:
                    continue
                except Exception as exc:
                    print(f"aether.metrics.los_swath_frames failed: {exc}")
                    break
                if isinstance(out, (list, tuple)) and out:
                    norm = _normalise_frames(out)
                    if norm:
                        return norm
    return _fallback_frames(slice_path)


def build_environment(slice_dict, layers, flood):
    """Environment block: wind / flood / DEM / GNSS-shadow screening.

    aether.metrics.environment_metrics(enriched_slice, flood_depth)
    is preferred and merged over the key fallback so wind means
    survive alongside relief / footprint / flood-blocked fields.
    """
    fallback = _fallback_environment(slice_dict, layers, flood)
    if _aether_metrics is not None:
        for name in ("environment_metrics", "environment",
                     "build_environment"):
            fn = getattr(_aether_metrics, name, None)
            if not callable(fn):
                continue
            if isinstance(flood, dict):
                flood_depth = np.asarray(flood.get("depth"), dtype=np.float32) if flood.get("depth") is not None else None
            else:
                try:
                    flood_depth = (np.asarray(flood, dtype=np.float32)
                                   if flood is not None else None)
                except Exception:
                    flood_depth = None
            enriched = dict(slice_dict) if isinstance(slice_dict, dict) else {}
            if isinstance(layers, dict) and layers.get("gnss") is not None:
                enriched["gnss_shadow"] = layers["gnss"]
            attempts = (
                lambda f=fn: f(enriched, flood_depth),
                lambda f=fn: f(slice_dict, flood_depth),
                lambda f=fn: f(slice_dict, flood),
                lambda f=fn: f(slice_dict),
            )
            for att in attempts:
                try:
                    out = att()
                except TypeError:
                    continue
                except Exception as exc:
                    print(f"aether.metrics.{name} failed: {exc}")
                    break
                if isinstance(out, dict) and out:
                    merged = dict(fallback)
                    for k, v in out.items():
                        merged[k] = v
                    return merged
    return fallback


def _fallback_environment(slice_dict, layers, flood):
    meta = slice_dict.get("meta", {}) if isinstance(slice_dict, dict) else {}
    try:
        wind = layers.get("wind") if isinstance(layers, dict) else None
        wind_mean = float(np.asarray(wind, dtype=float).mean()) if wind is not None else None
        wind_max = float(np.asarray(wind, dtype=float).max()) if wind is not None else None
        wind_p95 = float(np.percentile(np.asarray(wind, dtype=float), 95)) if wind is not None else None
    except Exception:
        wind_mean = wind_max = wind_p95 = None
    try:
        gnss = layers.get("gnss") if isinstance(layers, dict) else None
        gnss_mean = float(np.asarray(gnss, dtype=float).mean()) if gnss is not None else None
    except Exception:
        gnss_mean = None
    fmeta = (flood.get("meta", {}) if isinstance(flood, dict) else {}) or {}
    return {"wind_mean": wind_mean, "wind_max": wind_max,
            "wind_p95": wind_p95, "gnss_shadow_mean": gnss_mean,
            "flood_wet_cells": fmeta.get("wet_cells"),
            "flood_max_depth_m": fmeta.get("max_depth_m"),
            "flood_water_level_m": fmeta.get("water_level_m"),
            "dem_min_m": meta.get("dem_min_m"),
            "dem_max_m": meta.get("dem_max_m"),
            "dem_mean_m": meta.get("dem_mean_m"),
            "building_count": meta.get("n_buildings")}


def ground_metrics_extra(ground_result, slice_dict, sdf_bundle,
                         shadow_bundle, dem, flood):
    """Extended ground metrics: aether.metrics.ground_metrics + slope."""
    metrics_extra = {}
    if _aether_metrics is not None:
        for name in ("ground_metrics", "ground_flight_metrics",
                     "evaluate_ground"):
            fn = getattr(_aether_metrics, name, None)
            if not callable(fn):
                continue
            attempts = (
                lambda f=fn: f(ground_result),
                lambda f=fn: f(ground_result, None),
                lambda f=fn: f(ground_result, slice_dict),
            )
            for att in attempts:
                try:
                    out = att()
                except TypeError:
                    continue
                except Exception as exc:
                    print(f"aether.metrics.{name} failed: {exc}")
                    break
                if isinstance(out, dict) and out:
                    metrics_extra = {k: v for k, v in out.items()
                                     if k not in ("length_m", "time_s",
                                                    "max_flood_m",
                                                    "success")}
                    break
            if metrics_extra or name == "ground_metrics":
                break
    node_by_id = {n["id"]: n for n in slice_dict.get("nodes", [])}
    extra = {"path_nodes": len(getattr(ground_result, "path", []) or [])}
    try:
        if getattr(ground_result, "success", False) and ground_result.path:
            pts = [[float(node_by_id[i]["x"]), float(node_by_id[i]["y"])]
                   for i in ground_result.path]
            ev = evaluator.score_path(np.asarray(pts, dtype=float),
                                      sdf_bundle, shadow_bundle, dem,
                                      flood=flood, ground=True)
            extra["evaluator_score"] = (None if not math.isfinite(ev)
                                        else float(ev))
            slopes = []
            for i in range(len(ground_result.path) - 1):
                u = node_by_id[ground_result.path[i]]
                v = node_by_id[ground_result.path[i + 1]]
                seg = math.hypot(v["x"] - u["x"], v["y"] - u["y"])
                if seg > 0:
                    slopes.append(abs(float(v["z"]) - float(u["z"])) / seg)
            extra["max_slope"] = (round(float(max(slopes)), 4)
                                  if slopes else None)
            extra["mean_slope"] = (round(float(np.mean(slopes)), 4)
                                   if slopes else None)
        else:
            extra["evaluator_score"] = None
            extra["max_slope"] = None
            extra["mean_slope"] = None
    except Exception as exc:
        print(f"ground fallback metrics failed: {exc}")
        extra.setdefault("evaluator_score", None)
    for k, v in metrics_extra.items():
        extra[k] = v
    return extra


def main(argv=None) -> int:
    t0 = time.time()
    args = parse_args(argv)
    capture_dir = Path(args.capture_dir)
    out_path = Path(args.out)
    print(f"capture_dir={capture_dir} out={out_path} "
          f"metrics={'imported' if _aether_metrics is not None else 'fallback'}")
    s = build_real_slice(capture_dir)
    dem = s["dem"]
    meta = s["meta"]
    print(f"slice: {meta['n_buildings']} buildings, {meta['n_nodes']} nodes "
          f"{meta['n_edges']} edges, {meta['n_pads']} pads, "
          f"{meta['n_photos']} photos, dem {meta['dem_shape']} "
          f"max {meta['dem_max_m']:.1f} m")

    sdf_bundle = evaluator.build_sdf(s["buildings"])
    shadow_bundle = evaluator.gnss_shadow(s["buildings"])
    flood = evaluator.flood_tensor(dem, float(dem.min()) + 0.3)
    flood_depth = np.asarray(flood["depth"], dtype=np.float32)
    print(f"flood screening: {flood['meta']['wet_cells']} wet cells, "
          f"max depth {flood['meta']['max_depth_m']:.2f} m "
          f"(water {flood['meta']['water_level_m']:.2f} m)")

    layers = {"sdf": routing.build_sdf(s),
              "gnss": routing.build_gnss_shadow(s),
              "wind": routing.build_wind(s)}

    g_start, g_goal, comp = pick_swne_pair(s["nodes"], s["edges"])
    by_id = {n["id"]: n for n in s["nodes"]}
    g_start_node, g_goal_node = by_id[g_start], by_id[g_goal]
    print(f"ground passable component: {len(comp)} nodes "
          f"(of {len(s['nodes'])})")
    print(f"ground SW->NE: {g_start} ({g_start_node['x']:.0f},"
          f"{g_start_node['y']:.0f}) -> {g_goal} "
          f"({g_goal_node['x']:.0f},{g_goal_node['y']:.0f})")

    start_pad = nearest_pad(s["landing_pads"], 20.0, 20.0)
    goal_pad = nearest_pad(s["landing_pads"], 180.0, 180.0)
    drone_start = (float(start_pad["x"]), float(start_pad["y"]),
                   float(start_pad["z"]))
    drone_goal = (float(goal_pad["x"]), float(goal_pad["y"]),
                  float(goal_pad["z"]))
    print(f"drone pads: SW pad {start_pad['id']} {drone_start} -> "
          f"NE pad {goal_pad['id']} {drone_goal}")

    print("evolving weights on the real slice "
          f"(gens {GUIDE_GENS} pop {GUIDE_POP})...", flush=True)
    ev = evolve_on_slice(s, layers, drone_start, drone_goal,
                         g_start, g_goal, flood_depth)
    hist = ev["history"]
    w_gen25 = dict(hist[len(hist) // 2]["best_weights"])
    w_final = dict(ev["best_weights"])
    print(f"Gen0 {ev['gen0']:.1f} -> GenMid {ev['gen_intermediate']:.1f} "
          f"-> GenFinal {ev['gen_final']:.1f} best {w_final}")

    drone_paths, metrics = {}, {}
    slice_paths, drone_frames = {}, {}
    any_fail = False
    for name, w in (("Gen0", BLIND_WEIGHTS), ("Gen25", w_gen25),
                    ("final", w_final)):
        r = routing.drone_astar(s, drone_start, goal_xyz=drone_goal,
                                weights=w, sdf=layers["sdf"],
                                gnss_shadow=layers["gnss"],
                                wind=layers["wind"])
        if not r.success:
            print(f"{name}: drone A* FAILED: {r.notes}")
            any_fail = True
            continue
        slice_paths[name] = list(r.path)
        pts = [[cx(p[0]), float(p[2]), cx(p[1])] for p in r.path]
        drone_paths[name] = [[round(a, 1), round(b, 1), round(c, 1)]
                             for a, b, c in pts]
        base = route_metric(r, r.path, sdf_bundle, shadow_bundle, dem)
        extra = flight_metrics_extra(r, r.path, sdf_bundle, shadow_bundle,
                                     dem, layers, slice_dict=s)
        # aether.metrics `score` is a breakdown dict; keep the routing
        # scalar (float) under `score` and the breakdown under its own key.
        if isinstance(extra.get("score"), dict):
            extra["flight_score"] = extra.pop("score")
        base.update(extra)
        metrics[name] = base
        drone_frames[name] = drone_frames_for(r.path, s, layers, dem=dem,
                                              target_xy=(drone_goal[0],
                                                         drone_goal[1]))
        m = metrics[name]
        print(f"{name}: len={r.length_m:.0f} m time={m['time']:.1f} s "
              f"min_sdf={m['min_sdf_m']:.2f} score={m['score']:.1f}")
    if any_fail:
        return 1

    g = routing.ground_astar(s, g_start, g_goal, weights=w_final,
                             flood_tensor=flood_depth)
    if g.success:
        node_by_id = {n["id"]: n for n in s["nodes"]}
        ground_path = [[cx(node_by_id[i]["x"]), cx(node_by_id[i]["y"])]
                       for i in g.path]
        print(f"ground: len={g.length_m:.0f} m max_flood="
              f"{g.max_flood_m:.2f} m path_nodes={len(ground_path)} "
              f"time={g.time_s:.1f} s")
    else:
        ground_path = []
        print(f"ground: A* FAILED: {g.notes}")
    ground_block = {"success": bool(g.success),
                    "length_m": float(g.length_m),
                    "time_s": float(g.time_s) if g.success else None,
                    "max_flood_m": float(g.max_flood_m),
                    "start": [round(float(g_start_node["x"]), 1),
                              round(float(g_start_node["y"]), 1)],
                    "goal": [round(float(g_goal_node["x"]), 1),
                             round(float(g_goal_node["y"]), 1)],
                    "notes": g.notes}
    ground_block.update(ground_metrics_extra(g, s, sdf_bundle, shadow_bundle,
                                             dem, flood))
    environment = build_environment(s, layers, flood)

    gnss_mid = layers["gnss"][:, :, 3]  # (nx, ny) @ ~12 m altitude
    data = json.loads(out_path.read_text(encoding="utf-8"))
    stats = dict(data.get("stats") or {})
    stats.update({
        "buildings": meta["n_buildings"],
        "ground_nodes": meta["n_nodes"],
        "ground_edges": meta["n_edges"],
        "landing_pads": meta["n_pads"],
        "streetview_photos": meta["n_photos"],
        "mesh_vertices": meta["n_vertices"],
        "gen0_drone_len_m": round(metrics["Gen0"]["length_m"], 1),
        "gen25_drone_len_m": round(metrics["Gen25"]["length_m"], 1),
        "final_drone_len_m": round(metrics["final"]["length_m"], 1),
        "ground_len_m": round(float(g.length_m), 1) if g.success else None,
    })
    data.update({
        "label": meta["label"],
        "buildings": [{"x": cx(b["cx"]), "z": cx(b["cy"]), "w": b["w"],
                       "d": b["d"], "h": b["height_m"]}
                      for b in s["buildings"]],
        "dem": {"nx": 41, "nz": 41, "x0": -100, "z0": -100, "dx": 5, "dz": 5,
                "values": downsample(dem)},
        "gnss": {"nx": 41, "nz": 41, "x0": -100, "z0": -100, "dx": 5,
                 "dz": 5, "values": downsample(gnss_mid.T)},
        "dronePaths": drone_paths,
        "droneFrames": drone_frames,
        "groundPath": [[round(a, 1), round(b, 1)]
                       for a, b in ground_path],
        "metrics": metrics,
        "environment": environment,
        "evolution": {"gen0": ev["gen0"],
                      "genIntermediate": ev["gen_intermediate"],
                      "genFinal": ev["gen_final"],
                      "bestWeights": ev["best_weights"],
                      "note": "local evolution on the real Aleph slice "
                              f"(gens {GUIDE_GENS} pop {GUIDE_POP}); "
                              "CD cloud LONG would produce Gen500"},
        "ground": ground_block,
        "slice": {"capture": meta["capture"], "bbox": meta["bbox"],
                  "sources": meta["sources"]},
        "stats": stats,
    })
    out_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes) "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
