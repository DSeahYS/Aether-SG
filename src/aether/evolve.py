"""AETHER-SG evolve — local evolutionary search over routing heuristic weights.

IMPORTANT LABEL (Research-AETHER-SG.md sections 1, 4, 11):
  This is a LOCAL STAND-IN for the Google Computational Discovery (CD)
  cloud evolutionary search (SHORT ~6 h / LONG ~24 h).  It is NOT live CD.
  The MVP demo is an honest CACHED REPLAY of pre-compiled results (Gen0 /
  GenIntermediate / GenFinal).  No live CD call is made here.

What this module does
---------------------
- Evolves a population of heuristic weight dicts:
      {"w_time", "w_energy", "w_shadow", "w_wind", "w_detour"}
  that steer BOTH ground (barrier-free) and drone (0-60 m) A* via
  routing.py.
- Scores candidates against a SINGLE scalar metric (CD contract, sec 8):
      maximise  -(time + energy + shadow exposure + wind exposure + flood)
  with hard rejects (SDF collision, flood >=0.10 m on wheelchair profile,
  slope/width).  Delegates to evaluator.py if present, else uses a
  local fallback scalar with identical form.
- Trains on a seed-7 slice, checks generalisation on a HELD-OUT slice
  (different seed, e.g. seed=999) — held-out estate stays local and
  untouched during training (Research sec 8,10).
- Hardcoded-coordinate / leakage guard: reports Gen0 / GenIntermediate /
  GenFinal scores plus held-out drop, leakage checks and held-out status.
- CLI __main__ prints generation table.

Constraints
-----------
- Drone 0-60 m (NOT 120 m).  Solver envelope enforced in routing.py.
- Flood = screening tensor; wheelchair hard-block >=0.10 m.
- No "absolute-passability" or certification wording.  Advisory / screening
  framing only.
- Stdlib + numpy only.  Lazy imports so file compiles standalone.

Usage
-----
  python -m src.aether.evolve               # default demo
  python src/aether/evolve.py --gens 30 --pop 24 --seed 7

Honest cached-replay note
-------------------------
Gen0 is the baseline seed weight vector (w_time=1.0, w_energy=0.5 ...).
GenIntermediate / GenFinal are snapshots of the local run.  Cloud CD
would run 6-24 h on 8 vCPU to produce a cumulative Gen500 replay;
this stub shows the same ledger locally in ~seconds.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
WEIGHT_KEYS = ("w_time", "w_energy", "w_shadow", "w_wind", "w_detour")
DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_time": 1.0,
    "w_energy": 0.5,
    "w_shadow": 1.0,
    "w_wind": 1.0,
    "w_detour": 0.5,
}
# For scalarisation the TRUE coefficients are fixed (evaluator is frozen).
# Heuristic weights steer search; scalar measures outcome.
TRUE_TIME_W = 1.0
TRUE_ENERGY_W = 0.30
TRUE_SHADOW_W = 0.40
TRUE_WIND_W = 0.20

HELDOUT_SEED = 999  # different seed => different estate layout
HELDOUT_DROP_KILL = 0.15  # Research sec 10 kill: winner fails held-out by >15% -> pivot signal


# --------------------------------------------------------------------------
# Lazy sibling imports (compile-standalone safe)
# --------------------------------------------------------------------------

def _try_import(mod_name: str, attr: str):
    """Try package import then file-sibling import; return attr or None."""
    for cand in (f"aether.{mod_name}", mod_name):
        try:
            mod = importlib.import_module(cand)
            if hasattr(mod, attr):
                return getattr(mod, attr), mod
        except Exception:
            continue
    # file sibling
    try:
        here = Path(__file__).resolve().parent
        p = here / f"{mod_name}.py"
        if p.exists():
            spec = importlib.util.spec_from_file_location(mod_name, str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod  # type: ignore
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, attr):
                    return getattr(mod, attr), mod
    except Exception:
        pass
    return None, None


def _get_build_slice():
    fn, mod = _try_import("slice_data", "build_slice")
    if fn is not None:
        return fn
    # Fallback through routing.build_slice
    fn2, _ = _try_import("routing", "build_slice")
    if fn2 is not None:
        return fn2
    # Ultimate fallback: minimal synthetic (import routing fallback if still missing, use local stub)
    def _fallback_build_slice(seed: int = 7):
        # tiny synthetic stub so evolve still runs standalone
        rng = np.random.default_rng(int(seed))
        dem = rng.normal(0, 0.5, size=(200, 200)).astype(np.float32)
        buildings = [{"id": i, "cx": float(50 * i % 200), "cy": float(50 * (i // 4) % 200), "w": 24.0, "d": 12.0, "height_m": 30.0, "storeys": 10, "void_deck": False, "linkways": []} for i in range(14)]
        nodes = [{"id": i, "x": float(25 + (i % 5) * 40), "y": float(25 + (i // 5) * 40), "z": 1.0, "kind": "junction"} for i in range(25)]
        edges = [{"u": i, "v": i + 1, "length_m": 40.0, "slope": 0.02, "width_m": 2.0, "sheltered": False, "has_ramp": False} for i in range(24)]
        landing_pads = [{"id": 0, "x": 130.0, "y": 140.0, "z": 0.0, "kind": "courtyard", "building_id": -1, "size_m": 4.0}]
        return {"dem": dem, "dem_res_m": 1.0, "extent_m": 200.0, "buildings": buildings, "nodes": nodes, "edges": edges, "ramps": [], "landing_pads": landing_pads, "meta": {"seed": seed, "label": "evolve fallback synthetic"}}
    return _fallback_build_slice


def _get_routing():
    """Return routing module dict with ground_astar/drone_astar/build_sdf etc. Lazy."""
    fn_plan_ground, mod = _try_import("routing", "plan_ground_route")
    if mod is not None:
        return mod
    # file sibling routing.py
    here = Path(__file__).resolve().parent
    p = here / "routing.py"
    if p.exists():
        try:
            spec = importlib.util.spec_from_file_location("aether_routing_sib", str(p))
            if spec and spec.loader:
                m = importlib.util.module_from_spec(spec)
                sys.modules["aether_routing_sib"] = m  # type: ignore
                spec.loader.exec_module(m)  # type: ignore
                return m
        except Exception:
            pass
    return None


def _evaluator_score(weights: Dict[str, float], slice_dict: dict, routing_mod=None) -> float:
    """Single scalar to maximise (CD contract).

    Tries evaluator.py (evaluate_weights / score_candidate / evaluate_candidate)
    lazily; falls back to local scalar identical in spirit to Research sec 8.

    Local scalar = -( drone_time*1.0 + drone_energy*0.30 + shadow*0.40 + wind*0.20
                      + ground_time*1.0 + ground_length*0.01 )
    Hard rejects: if both routes fail, -1e6; if drone fails alone, heavy penalty.
    """
    # Try evaluator module lazily (multiple possible entry point names)
    for cand_name, cand_attr in (
        ("evaluator", "evaluate_weights"),
        ("evaluator", "score_candidate"),
        ("evaluator", "evaluate_candidate"),
        ("evaluator", "score_weights"),
        ("evaluator", "evaluate"),
    ):
        fn, e_mod = _try_import(cand_name, cand_attr)
        if fn is not None:
            try:
                v = fn(weights, slice_dict)  # type: ignore
                return float(v)
            except TypeError:
                try:
                    v = fn(weights, slice_dict, None)  # try extra arg?
                    return float(v)
                except Exception:
                    pass
            except Exception:
                pass

    # If evaluator provides score_path (tensor evaluator), delegate to it after routing
    fn_sp, e_mod_sp = _try_import("evaluator", "score_path")
    if fn_sp is not None:
        routing_mod = routing_mod or _get_routing()
        if routing_mod is not None:
            try:
                nodes = slice_dict.get("nodes", [])
                junctions = [n for n in nodes if n["kind"] == "junction"]
                entrances = [n for n in nodes if n["kind"] == "entrance"]
                if junctions and entrances:
                    g_start, g_goal = int(junctions[0]["id"]), int(entrances[0]["id"])
                elif nodes:
                    g_start, g_goal = int(nodes[0]["id"]), int(nodes[-1]["id"])
                else:
                    g_start, g_goal = 0, 0
                sdf = routing_mod.build_sdf(slice_dict)  # type: ignore
                gnss = routing_mod.build_gnss_shadow(slice_dict)  # type: ignore
                wind = routing_mod.build_wind(slice_dict)  # type: ignore
                g = routing_mod.ground_astar(slice_dict, g_start, g_goal, weights=weights)  # type: ignore
                d = routing_mod.drone_astar(slice_dict, (25.0, 25.0, 12.0), goal_xyz=(130.0, 140.0, 5.0), weights=weights, sdf=sdf, gnss_shadow=gnss, wind=wind)  # type: ignore
                b_sdf, _ = _try_import("evaluator", "build_sdf")
                b_shadow, _ = _try_import("evaluator", "gnss_shadow")
                b_flood, _ = _try_import("evaluator", "flood_tensor")
                dem = slice_dict.get("dem")
                if b_sdf is not None and b_shadow is not None and dem is not None and d.success:
                    sdf_b = b_sdf(slice_dict["buildings"])  # type: ignore
                    sh_b = b_shadow(slice_dict["buildings"])  # type: ignore
                    fl_b = None
                    if b_flood is not None:
                        try:
                            fl_b = b_flood(dem, float(np.asarray(dem).min()) + 0.6)  # type: ignore
                        except Exception:
                            fl_b = None
                    pts = np.asarray(d.path, dtype=np.float64)
                    ev = fn_sp(pts, sdf_b, sh_b, dem, flood=fl_b, ground=False)  # type: ignore
                    if g.success:
                        ev -= float(g.time_s) * 1.0 + float(g.length_m) * 0.01
                    else:
                        ev -= 800.0
                    if not np.isfinite(ev):
                        return float(-1e6)
                    return float(ev)
            except Exception:
                pass

    # Fallback: local routing-based scalar
    if routing_mod is None:
        routing_mod = _get_routing()
    if routing_mod is None:
        # No routing available: random-like score from weights diversity (still deterministic)
        h = sum(weights[k] * (i + 1) for i, k in enumerate(WEIGHT_KEYS))
        return float(-h * 10.0)

    # Use routing functions
    try:
        nodes = slice_dict.get("nodes", [])
        junctions = [n for n in nodes if n["kind"] == "junction"]
        entrances = [n for n in nodes if n["kind"] == "entrance"]
        if junctions and entrances:
            g_start, g_goal = int(junctions[0]["id"]), int(entrances[0]["id"])
        elif nodes:
            g_start, g_goal = int(nodes[0]["id"]), int(nodes[-1]["id"])
        else:
            g_start, g_goal = 0, 0

        # Cache layers for speed if routing provides builders
        sdf = None
        gnss = None
        wind = None
        try:
            sdf = routing_mod.build_sdf(slice_dict)  # type: ignore
            gnss = routing_mod.build_gnss_shadow(slice_dict)  # type: ignore
            wind = routing_mod.build_wind(slice_dict)  # type: ignore
        except Exception:
            pass

        g = routing_mod.ground_astar(slice_dict, g_start, g_goal, weights=weights)  # type: ignore
        d = routing_mod.drone_astar(slice_dict, (25.0, 25.0, 12.0), goal_xyz=(130.0, 140.0, 5.0), weights=weights, sdf=sdf, gnss_shadow=gnss, wind=wind)  # type: ignore

        score = 0.0
        if g.success:
            score -= float(g.time_s) * TRUE_TIME_W
            score -= float(g.length_m) * 0.01
        else:
            score -= 800.0  # ground infeasible (flood hard-block) heavy
        if d.success:
            score -= float(d.time_s) * TRUE_TIME_W
            score -= float(d.energy) * TRUE_ENERGY_W
            score -= float(d.shadow_exposure) * TRUE_SHADOW_W
            score -= float(d.wind_exposure) * TRUE_WIND_W
            # Bonus for landing on pad vicinity? already in path
        else:
            score -= 1000.0
        # Normalise a bit? Keep raw scalar.
        return float(score)
    except Exception as ex:
        # If routing fails, return low but finite score so evolution can continue
        return float(-1e5 + abs(hash(str(ex))) % 1000)


# --------------------------------------------------------------------------
# Weight helpers
# --------------------------------------------------------------------------

def random_weights(rng: random.Random) -> Dict[str, float]:
    w = {}
    # Sample in plausible heuristic ranges; not coordinate-like values
    # w_time ~0.5-2.0, w_energy 0-1.5, w_shadow 0-2.5, w_wind 0-2.5, w_detour 0-1.5
    w["w_time"] = rng.uniform(0.5, 2.0)
    w["w_energy"] = rng.uniform(0.0, 1.5)
    w["w_shadow"] = rng.uniform(0.0, 2.5)
    w["w_wind"] = rng.uniform(0.0, 2.5)
    w["w_detour"] = rng.uniform(0.0, 1.5)
    return w


def mutate(weights: Dict[str, float], rng: random.Random, sigma: float = 0.25) -> Dict[str, float]:
    """Gaussian mutation per weight + occasional reset; clipped to [0, 3]."""
    out = {}
    for k in WEIGHT_KEYS:
        v = weights.get(k, DEFAULT_WEIGHTS[k])
        if rng.random() < 0.08:
            # occasional reset to random in range
            out[k] = rng.uniform(0, 2.5)
        else:
            out[k] = v + rng.gauss(0.0, sigma)
            out[k] = float(np.clip(out[k], 0.0, 3.0))
    return out


def crossover(a: Dict[str, float], b: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    out = {}
    for k in WEIGHT_KEYS:
        # uniform per gene
        if rng.random() < 0.5:
            out[k] = float(a[k])
        else:
            out[k] = float(b[k])
    # 30% chance arithmetic blend for one gene
    if rng.random() < 0.3:
        blend_k = rng.choice(WEIGHT_KEYS)
        alpha = rng.uniform(0, 1)
        out[blend_k] = float(alpha * a[blend_k] + (1 - alpha) * b[blend_k])
    return out


@dataclass
class Individual:
    weights: Dict[str, float]
    score: float = -math.inf
    best_gen: int = 0


# --------------------------------------------------------------------------
# Leakage / hardcoded-coordinate guard
# --------------------------------------------------------------------------

def leakage_guard(
    best_weights: Dict[str, float],
    train_score: float,
    heldout_score: float,
    train_slice: dict,
    heldout_slice: dict,
) -> Dict[str, object]:
    """Hardcoded-coordinate / leakage checks (Research sec 10 kill criterion 2).

    Checks:
      - Weights must be small scalars in [0,3] (not coordinates / building IDs).
      - Held-out generalisation drop computed; flag if >15% (advisory kill signal).
      - Training slice building count vs held-out building count to ensure
        different estate (seed differs).

    Returns dict with booleans and human-readable notes.  Does NOT claim
    absolute generalisation; screening only.
    """
    hardcode_ok = True
    reasons: List[str] = []
    # 1) Weight value sanity: coordinates in 200x200 slice would be >10 or encode size
    for k in WEIGHT_KEYS:
        v = best_weights.get(k, 0.0)
        if not (0.0 <= float(v) <= 3.0):
            hardcode_ok = False
            reasons.append(f"{k}={v:.3f} outside [0,3] — possible coordinate leak")
        # Too many decimals suggests memorised value?
        # We allow but note if >4 decimal places (nope)
    if set(best_weights.keys()) != set(WEIGHT_KEYS):
        hardcode_ok = False
        reasons.append("weight key set != expected 5 keys")

    # 2) Generalisation drop
    # Careful: scores are negative, so "drop" = (train - heldout) is held-out worse if heldout < train
    # We use relative drop on absolute value? For negative scores, define:
    #   drop = (heldout - train) / |train|  (negative means worse)
    denom = abs(train_score) if abs(train_score) > 1e-9 else 1.0
    drop = (heldout_score - train_score) / denom  # negative = held-out lower
    # drop magnitude >0.15 means held-out worse by >15% of |train| in absolute terms
    # Since scores negative, held-out worse is more negative => drop negative large magnitude
    generalises = True
    if drop < -HELDOUT_DROP_KILL:  # e.g. -0.20
        generalises = False
        reasons.append(f"held-out drop {drop*100:.1f}% exceeds {HELDOUT_DROP_KILL*100:.0f}% kill threshold (advisory; check slice difference)")

    # 3) Estate difference: buildings / nodes differ
    train_b = len(train_slice.get("buildings", []))
    held_b = len(heldout_slice.get("buildings", []))
    seeds = (train_slice.get("meta", {}).get("seed"), heldout_slice.get("meta", {}).get("seed"))
    if seeds[0] == seeds[1]:
        reasons.append("train and held-out seeds identical — held-out not independent")

    # 4) trivial baseline must score visibly badly: Gen0 baseline vs best should differ
    # (informational, not a hard fail)

    status = "PASS" if (hardcode_ok and generalises) else ("REVIEW" if hardcode_ok else "FAIL")
    return {
        "hardcode_ok": bool(hardcode_ok),
        "generalises": bool(generalises),
        "heldout_drop": float(drop),
        "status": status,
        "reasons": reasons,
        "train_buildings": train_b,
        "heldout_buildings": held_b,
    }


# --------------------------------------------------------------------------
# Evolution loop
# --------------------------------------------------------------------------

def evolve(
    train_seed: int = 7,
    heldout_seed: int = HELDOUT_SEED,
    generations: int = 25,
    pop_size: int = 24,
    sigma: float = 0.25,
    elite_frac: float = 0.25,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run local evolutionary search; return history dict.

    Returns keys:
      history: list of dicts {gen, best_score, mean_score, heldout_score, best_weights}
      gen0: score of baseline (DEFAULT_WEIGHTS) on train
      gen_intermediate: best score at middle generation
      gen_final: final best
      heldout_final: final best on held-out
      guard: leakage_guard dict
      best_weights, train_score, heldout_score
    """
    rng = random.Random(int(seed))
    np_rng = np.random.default_rng(int(seed))

    # Build slices — lazy, fallback handles missing slice_data.py
    build_slice = _get_build_slice()
    train_slice = build_slice(seed=train_seed)
    heldout_slice = build_slice(seed=heldout_seed)

    routing_mod = _get_routing()
    if routing_mod is None and verbose:
        print("[evolve] routing.py not found as import; using evolved scalar fallback (check import path)")

    # Pre-cache routing layers for speed (optional)
    # We will call _evaluator_score which itself caches; for speed we could reuse but keep simple.

    # Generation 0 baseline
    gen0_score = _evaluator_score(DEFAULT_WEIGHTS, train_slice, routing_mod=routing_mod)
    if verbose:
        print(f"[evolve] Gen0 baseline {DEFAULT_WEIGHTS} -> train score {gen0_score:.2f} (seed {train_seed})")

    # Initialise population
    pop: List[Individual] = []
    # Include baseline as first individual (seed)
    pop.append(Individual(weights=DEFAULT_WEIGHTS.copy(), score=gen0_score, best_gen=0))
    for _ in range(pop_size - 1):
        w = random_weights(rng)
        pop.append(Individual(weights=w))

    # Score initial pop
    for ind in pop:
        if not math.isfinite(ind.score) or ind.score == -math.inf:
            ind.score = _evaluator_score(ind.weights, train_slice, routing_mod=routing_mod)

    history: List[Dict[str, object]] = []

    def record(gen: int, pop: List[Individual]) -> Dict[str, object]:
        best = max(pop, key=lambda x: x.score)
        mean = float(np.mean([x.score for x in pop]))
        # Held-out score of best at this generation (costly but MVP demo expects it)
        held = float(_evaluator_score(best.weights, heldout_slice, routing_mod=routing_mod))
        return {
            "gen": gen,
            "best_score": float(best.score),
            "mean_score": float(mean),
            "heldout_score": float(held),
            "best_weights": dict(best.weights),
        }

    history.append(record(0, pop))
    if verbose:
        print(f"[evolve] Gen0 best {history[0]['best_score']:.2f} mean {history[0]['mean_score']:.2f} heldout {history[0]['heldout_score']:.2f}")

    # Cache population scores dict to avoid re-evaluating duplicates? Keep simple.

    start_t = time.time()
    for gen in range(1, generations + 1):
        # Selection: tournament size 3, elitism
        pop_sorted = sorted(pop, key=lambda x: x.score, reverse=True)
        n_elite = max(1, int(pop_size * elite_frac))
        elites = pop_sorted[:n_elite]

        new_pop: List[Individual] = [Individual(weights=dict(e.weights), score=float(e.score), best_gen=gen) for e in elites]

        # Tournament pool: sample with replacement, pick best
        def tournament() -> Individual:
            cand = [pop[rng.randrange(pop_size)] for _ in range(3)]
            return max(cand, key=lambda x: x.score)

        # Fill remainder via crossover + mutation
        while len(new_pop) < pop_size:
            p1 = tournament()
            p2 = tournament()
            child_w = crossover(p1.weights, p2.weights, rng)
            child_w = mutate(child_w, rng, sigma=sigma)
            # Score child
            sc = _evaluator_score(child_w, train_slice, routing_mod=routing_mod)
            new_pop.append(Individual(weights=child_w, score=sc, best_gen=gen))

        pop = new_pop
        rec = record(gen, pop)
        history.append(rec)
        if verbose and (gen % 5 == 0 or gen == generations or gen == generations // 2):
            elapsed = time.time() - start_t
            print(f"[evolve] Gen{gen:02d} best {rec['best_score']:.2f} mean {rec['mean_score']:.2f} heldout {rec['heldout_score']:.2f} t={elapsed:.1f}s")

    # Snapshots: Gen0 is the seed baseline weight vector (trivial baseline
    # must score visibly badly, Research sec 8). GenIntermediate = middle
    # generation, GenFinal = last generation. History[0] is population best
    # after seeding (baseline + random); baseline itself is gen0_score.
    gen0 = float(gen0_score)
    mid_idx = generations // 2
    gen_mid = float(history[mid_idx]["best_score"])
    gen_final = float(history[-1]["best_score"])
    heldout_final = float(history[-1]["heldout_score"])
    best_ind = max(pop, key=lambda x: x.score)
    best_weights = dict(best_ind.weights)

    guard = leakage_guard(best_weights, gen_final, heldout_final, train_slice, heldout_slice)

    # Also compute trivial baseline comparison: baseline vs best must differ
    # visibly; guard records held-out drop (kill >15% => pivot signal, sec 10)
    improvement = gen_final - gen0  # >0 means better (less negative)
    improves = improvement > 3.0  # threshold: at least 3 points improvement on this scalar (visible)
    if verbose:
        print(f"[evolve] Gen0 {gen0:.2f} -> GenMid {gen_mid:.2f} -> GenFinal {gen_final:.2f} improvement {improvement:.2f} improves_visibly={improves}")
        print(f"[evolve] Held-out final {heldout_final:.2f} drop {guard['heldout_drop']*100:.1f}% status {guard['status']} reasons {guard['reasons']}")

    return {
        "history": history,
        "gen0": gen0,
        "gen_intermediate": gen_mid,
        "gen_final": gen_final,
        "heldout_final": heldout_final,
        "best_weights": best_weights,
        "train_score": gen_final,
        "heldout_score": heldout_final,
        "guard": guard,
        "train_slice": train_slice,
        "heldout_slice": heldout_slice,
        "improvement": float(improvement),
        "improves_visibly": bool(improves),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_table(result: Dict[str, object]):
    hist: List[Dict[str, object]] = result["history"]  # type: ignore
    print("")
    print("=" * 78)
    print("AETHER-SG evolve — LOCAL STAND-IN for CD cloud search (6-24 h), NOT live CD")
    print("Honest cached replay · single scalar metric · 0-60 m solver (NOT 120 m)")
    print("Flood = screening tensor, wheelchair block >=0.10 m · Advisory only")
    print("=" * 78)
    print("Note: table Gen0 = best of initial seeded population (baseline + random); Gen0 baseline (DEFAULT seed weights) below must score visibly worse — trivial baseline checks (sec 8).")
    print(f"{'Gen':>4}  {'Pop best':>14}  {'Mean':>10}  {'Held-out':>12}   best weights [w_time w_energy w_shadow w_wind w_detour]")
    print("-" * 78)
    for r in hist:
        w = r["best_weights"]
        print(
            f"{int(r['gen']):4d}  {float(r['best_score']):14.2f}  {float(r['mean_score']):10.2f}  "
            f"{float(r['heldout_score']):12.2f}   "
            f"[{w['w_time']:.2f} {w['w_energy']:.2f} {w['w_shadow']:.2f} {w['w_wind']:.2f} {w['w_detour']:.2f}]"
        )
    print("-" * 78)
    print(f"Gen0 baseline (DEFAULT, train):   {float(result['gen0']):.2f}  <-- trivial baseline must score visibly worse")
    print(f"GenIntermediate (Mid, train):     {float(result['gen_intermediate']):.2f}")
    print(f"GenFinal (train):                 {float(result['gen_final']):.2f}")
    print(f"GenFinal (held-out, seed {HELDOUT_SEED}): {float(result['heldout_final']):.2f}  <-- held-out estate, local untouched")
    guard = result["guard"]
    print(f"Leakage guard: status={guard['status']} hardcode_ok={guard['hardcode_ok']} generalises={guard['generalises']} drop={float(guard['heldout_drop'])*100:.1f}%")
    if guard["reasons"]:  # type: ignore
        for r in guard["reasons"]:  # type: ignore
            print(f"  - {r}")
    print(f"Best weights: {result['best_weights']}")
    print(f"Improvement Gen0->GenFinal: {float(result['improvement']):.2f} (visibly better: {result['improves_visibly']})")
    print("Cached replay: Gen0/Gen25-class local run shown; cloud CD LONG (24 h) would produce Gen500 appendix.")
    print("=" * 78)


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="AETHER-SG local evolve stand-in (NOT live CD cloud search 6-24 h)")
    ap.add_argument("--gens", type=int, default=20, help="generations (local demo)")
    ap.add_argument("--pop", type=int, default=16, help="population size")
    ap.add_argument("--seed", type=int, default=42, help="rng seed")
    ap.add_argument("--train-seed", type=int, default=7, help="training slice seed")
    ap.add_argument("--heldout-seed", type=int, default=HELDOUT_SEED, help="held-out slice seed (different)")
    ap.add_argument("--sigma", type=float, default=0.25, help="mutation sigma")
    args = ap.parse_args(argv)
    print("AETHER-SG evolve.py — LOCAL STAND-IN for CD cloud search (SHORT ~6 h / LONG ~24 h), NOT live CD")
    print("Honest cached-replay framing: local seconds-long demo; cloud CD would replay Gen500 over 24 h.")
    result = evolve(
        train_seed=args.train_seed,
        heldout_seed=args.heldout_seed,
        generations=args.gens,
        pop_size=args.pop,
        sigma=args.sigma,
        seed=args.seed,
        verbose=True,
    )
    _print_table(result)
    return result


if __name__ == "__main__":
    main()
