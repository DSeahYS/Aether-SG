# AETHER-SG

**Autonomous Emergency Trajectory, HADR & Evacuation Twin for Dense Urban Canyons (Singapore)**

---

## 1. Documentation
- [Admin/Idea.md](Admin/Idea.md) — Core concept specification, problem validation, and architecture.
- [Admin/Research-AETHER-SG.md](Admin/Research-AETHER-SG.md) — Comprehensive technical feasibility, regulatory boundaries (CAAS 60 m ceiling), and empirical research report.

---

## 2. Initial Core Modules
- `src/aether/slice_data.py` — Synthetic 200×200 m urban slice generator (DEM terrain, HDB building geometry, pedestrian ground graph, curb ramps, landing pads).
- `src/aether/evaluator.py` — 3D voxel Signed Distance Field (SDF) collision checker (0–60 m envelope), GNSS shadow proxy, and flood screening tensor.
- `src/aether/routing.py` — Coupled dual-pathfinder with wheelchair ground A* (respecting water depth <0.10 m, slope <1:12) and 3D UAV A* corridor solver.
- `src/aether/evolve.py` — Local evolutionary heuristic optimizer for multi-objective search weights.
- `cd/make_notebook.py` — Computational Discovery (CD) 3-part experiment notebook generator (Setup / Solution / Scoring contract).
- `viewer/` — Offline Three.js tactical 3D visualizer.
- `tests/test_core.py` — Unit test suite (9 tests).

---

## 3. Quickstart

### Run Tests
```powershell
python -m pytest tests/test_core.py -v
```

### Run 3D Tactical Viewer
Open `viewer/index.html` in your web browser or serve locally:
```powershell
python -m http.server 8100
```
Then visit `http://127.0.0.1:8100/viewer/`.
