"""Focused regression: Gen0 baseline must differ from, and score worse
than, the shadow/wind-aware route (demo generation narrative).

Guards the run_demo fix where all generations rendered the identical path.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aether import routing, slice_data  # noqa: E402

BLIND = {"w_time": 1.0, "w_energy": 0.0, "w_shadow": 0.0,
         "w_wind": 0.0, "w_detour": 0.0}
START, GOAL = (10.0, 10.0, 12.0), (130.0, 140.0, 8.0)


def _proxy(r):
    return -(r.time_s + 0.3 * r.energy + 0.4 * r.shadow_exposure
             + 0.2 * r.wind_exposure)


def test_gen0_blind_differs_and_scores_worse():
    s = slice_data.build_slice(seed=7)
    blind = routing.drone_astar(s, START, goal_xyz=GOAL, weights=BLIND)
    aware = routing.drone_astar(
        s, START, goal_xyz=GOAL, weights=dict(routing.DEFAULT_WEIGHTS))
    assert blind.success and aware.success
    assert blind.path != aware.path
    assert blind.shadow_exposure > aware.shadow_exposure
    assert _proxy(blind) < _proxy(aware)
    assert max(p[2] for p in aware.path) <= 60.0
