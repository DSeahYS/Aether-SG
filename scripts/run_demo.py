"""AETHER-SG orchestrator demo (Wonder integration layer).

Pipeline: synthetic Toa Payoh-like slice (seed 7) -> SDF/GNSS/flood tensors
-> local evolution (Ember evolve.py, stand-in for CD 6-24h cloud search)
-> Gen0 / Gen25 / Final drone paths + flooded wheelchair ground route
-> viewer/data.json for the offline tactical viewer.

Honest framing: simulated slice, screening flood, cached replay.
Drone envelope 0-60 m. Not for navigation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aether import evaluator, routing, slice_data  # noqa: E402
from aether.evolve import evolve  # noqa: E402

CX = lambda v: float(v) - 100.0  # slice 0..200 -> viewer -100..100


def downsample(grid: np.ndarray, n: int = 41) -> list:
    idx_y = np.linspace(0, grid.shape[0] - 1, n).astype(int)
    idx_x = np.linspace(0, grid.shape[1] - 1, n).astype(int)
    return grid[np.ix_(idx_y, idx_x)].astype(float).round(3).tolist()


def main() -> int:
    t0 = time.time()
    s = slice_data.build_slice(seed=7)
    dem = s["dem"]
    sdf_b = evaluator.build_sdf(s["buildings"])
    shadow_b = evaluator.gnss_shadow(s["buildings"])

    # Local evolution (seconds) — stand-in for CD SHORT/LONG cloud search.
    print("evolving (local stand-in for CD cloud search)...", flush=True)
    ev = evolve(generations=5, pop_size=8, verbose=False)
    hist = ev["history"]
    # Gen0 baseline = shadow/wind-blind seed heuristic (visibly worse:
    # shorter path, higher GNSS-shadow exposure). Gen25/Final = evolved.
    w_gen0 = {"w_time": 1.0, "w_energy": 0.0, "w_shadow": 0.0,
              "w_wind": 0.0, "w_detour": 0.0}
    w_gen25 = dict(hist[len(hist) // 2]["best_weights"])
    w_final = dict(ev["best_weights"])
    print(f"Gen0 {ev['gen0']:.1f} -> GenMid {ev['gen_intermediate']:.1f} "
          f"-> GenFinal {ev['gen_final']:.1f} | held-out {ev['heldout_final']:.1f} "
          f"| guard {ev['guard']['status']}")

    start_xyz = (10.0, 10.0, 12.0)
    goal_xyz = (130.0, 140.0, 8.0)

    drone_paths, metrics = {}, {}
    for name, w in (("Gen0", w_gen0), ("Gen25", w_gen25), ("final", w_final)):
        r = routing.drone_astar(s, start_xyz, goal_xyz=goal_xyz, weights=w)
        if not r.success:
            print(f"{name}: drone A* FAILED: {r.notes}")
            return 1
        pts = [[CX(p[0]), float(p[2]), CX(p[1])] for p in r.path]  # [x, alt, z]
        drone_paths[name] = [[round(a, 1), round(b, 1), round(c, 1)] for a, b, c in pts]
        # Viewer metric = same scalar family evolve.py optimises
        # (-(time + energy + shadow + wind)), so Gen ordering is coherent.
        score = -(r.time_s + 0.3 * r.energy + 0.4 * r.shadow_exposure
                  + 0.2 * r.wind_exposure)
        metrics[name] = {"score": float(score), "time": float(r.time_s),
                         "collisions": 0 if r.min_sdf_m >= 0 else 1}
        print(f"{name}: len={r.length_m:.0f}m time={r.time_s:.1f}s "
              f"min_sdf={r.min_sdf_m:.2f} score={score:.1f}")

    # Flooded wheelchair ground route (screening tensor, block >= 0.10 m).
    flood = routing.build_flood_tensor(s)
    nodes = {n["id"]: n for n in s["nodes"]}
    start_id = min((n for n in s["nodes"] if n["kind"] == "junction"),
                   key=lambda n: (n["x"] - 25) ** 2 + (n["y"] - 25) ** 2)["id"]
    goal_id = min(s["nodes"],
                  key=lambda n: (n["x"] - 185) ** 2 + (n["y"] - 185) ** 2)["id"]
    g = routing.ground_astar(s, start_id, goal_id, flood_tensor=flood)
    if not g.success:
        print(f"ground A* FAILED: {g.notes}")
        return 1
    ground_path = [[CX(nodes[i]["x"]), CX(nodes[i]["y"])] for i in g.path]
    print(f"ground: len={g.length_m:.0f}m max_flood={g.max_flood_m:.2f}m "
          f"path_nodes={len(ground_path)}")

    gnss3d = routing.build_gnss_shadow(s)[:, :, 3]  # (nx, ny) @ ~12 m
    data = {
        "label": "SIMULATED Toa Payoh-like slice (seed 7) — not for navigation",
        "buildings": [{"x": CX(b["cx"]), "z": CX(b["cy"]), "w": b["w"],
                       "d": b["d"], "h": b["height_m"]} for b in s["buildings"]],
        "dem": {"nx": 41, "nz": 41, "x0": -100, "z0": -100, "dx": 5, "dz": 5,
                "values": downsample(dem)},
        "gnss": {"nx": 41, "nz": 41, "x0": -100, "z0": -100, "dx": 5, "dz": 5,
                 "values": downsample(gnss3d.T)},
        "dronePaths": drone_paths,
        "groundPath": [[round(a, 1), round(b, 1)] for a, b in ground_path],
        "ramps": [{"x": CX(r["x"]), "z": CX(r["y"]),
                   "verified": bool(r["verified"])} for r in s["ramps"]],
        "metrics": metrics,
        "evolution": {"gen0": ev["gen0"], "genIntermediate": ev["gen_intermediate"],
                      "genFinal": ev["gen_final"], "heldoutFinal": ev["heldout_final"],
                      "guard": ev["guard"]["status"],
                      "note": "local stand-in; CD cloud LONG would produce Gen500"},
        "ground": {"length_m": g.length_m, "max_flood_m": g.max_flood_m},
    }
    out = ROOT / "viewer" / "data.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
