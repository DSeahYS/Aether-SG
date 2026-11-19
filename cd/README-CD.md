# AETHER-SG: Computational Discovery (CD) notebook

Cached-replay build only. The heavy search runs in the cloud beforehand;
the chosen solver then runs offline and is replayed on stage. Do not
present the stage replay as a live search. Data staged here is a
deterministic synthetic Toa Payoh-like estate (SIMULATED). Official
datasets would replace the files without changing the solver interface.

Notebook: `aether_evolve.ipynb` built by `make_notebook.py`
(setup -> mutable -> scoring). Builder needs only Python 3.11+ stdlib
and numpy.

## 1. What the notebook expects

Staged files live under `/data/input/` in the CD sandbox:

| Display name | Role |
|---|---|
| `toapayoh_train_slice.npz` | train tensors (`dem`, `flood_depth`, `sdf`, `shadow`) |
| `toapayoh_train_slice.json` | task, buildings, graph, water level, seed |

Holdout files (`toapayoh_holdout_slice.*`, seed 99) stay local. They are
never staged, so the search cannot fit to them (same-kernel leakage
guard).

Tensor contract (2 m grid unless noted):

- `dem` (100, 100) float32: ground height from the 200 x 200 m DEM.
- `flood_depth` (100, 100) float32: screening depth in metres,
  water level minus ground, kept only where drain-connected
  (bathtub plus connectivity). Screening aid only; operational levels
  sit with the water agency.
- `sdf` (30, 100, 100) float32: signed distance, negative inside a
  building box, positive in free air. Levels are 0, 2, ..., 60 m.
- `shadow` (30, 100, 100) uint8 0/1: building-occlusion proxy for
  sky visibility. Advisory only.
- task: `ground_start_xy`, `ground_goal_xy`,
  `drone_start_xyz`, `drone_goal_xyz` (metres).

Planner envelope is 0-60 m above the datum. Ground screening blocks
cells whose depth is at or above 0.10 m for the wheelchair profile,
plus steep cells and building footprints. The candidate is judged by
one scalar score computed in the hidden scoring cells. Multi-objective
scoring by eye is out of scope for the search.

## 2. Build the staged slice locally

From the repo root:

```bat
python cd\make_notebook.py
```

That writes:

- `cd\aether_evolve.ipynb` (the notebook)
- `cd\data\toapayoh_train_slice.npz` + `.json` (seed 7)
- `cd\data\toapayoh_holdout_slice.npz` + `.json` (seed 99, keep local)

`make_notebook.py` prefers the sibling `src/aether` core so the staged
tensors match the viewer and routing schema. With no sibling import it
falls back to a standalone synthetic set with the same keys. Either way
the output shapes above are stable.

Local smoke (no cloud):

```bat
python -m py_compile cd\make_notebook.py
python cd\make_notebook.py
python -m pytest tests\test_core.py -v
```

The builder's last lines run a tiny local preview search and report the
Gen0 score and the incumbent. That preview is a compile check only. The
shippable efficiency delta comes from the cloud search lineage, not from
this preview.

## 3. Stage with cd_upload

1. Build the slice (step 2 above) and note the water level in the JSON
   sidecar (train: lowest ground point plus 0.5 m).
2. Upload the train pair through the CD dataset tool with the display
   names exactly as above:

   ```text
   cd_upload file=cd\data\toapayoh_train_slice.npz displayName=toapayoh_train_slice.npz
   cd_upload file=cd\data\toapayoh_train_slice.json displayName=toapayoh_train_slice.json
   ```

   Use the dataset you already open in the CD web app. Do not stage the
   holdout pair.
3. Open `aether_evolve.ipynb`, confirm the first setup cell sees
   `DATA_DIR = "/data/input"` and loads both files, then run the
   test run (next section) before launching any long search.

If the setup cell raises `FileNotFoundError` for the staged path, the
upload did not land. Re-upload with the exact display names and rerun.

## 4. Test run (25-minute cap)

Minimum check before any long run (Research sec. 4):

- Sandbox is CPU-only, 8 vCPU / 32 GiB, Python 3.12, no internet.
- Staged input is mounted read-only at `/data/input/`.
- The test-run harness must finish within 25 minutes or it is stopped.
- First scores for a healthy setup appear within 15-30 minutes.

Checklist:

- Setup cell loads the staged tensors and prints their shapes.
- Mutable A* baseline returns non-empty ground and drone paths.
- Scoring cell prints exactly one scalar line. A missing or repeated
  print means the run is unscored; fix the notebook and rerun the test.
- No candidate exceeds the evaluator budget (target under 5 s per
  candidate on the sandbox). If scoring is slow, slim the grid or the
  densification before launching.

## 5. SHORT vs LONG

- SHORT: about 6 hours of cloud search.
- LONG: about 24 hours of cloud search.

For the demo lineage, treat an interrupted SHORT run as the Gen25
replay and a finished LONG run as the Gen500 replay, with the CD web-app
lineage shown next to each. Gen0 is the shipped A* baseline. Do not
quote an efficiency delta until a CD run exists; the placeholder in
earlier notes stays blank until measured.

Budget honestly: published AlphaEvolve guidance plans on the order of
minutes per candidate, with up to roughly one hundred compute-hours for
hard tasks. AETHER's evaluator is far cheaper because it is precomputed
lookups plus analytic penalties, but still plan the search offline and
cache the winner. Do not budget stage evolution.

## 6. Publication note (TTA)

CD terms for the Early Access programme (Research sec. 4):

- There is no data opt-out for staged content.
- Any public write-up or demo that uses CD results needs email
  sign-off at least 30 days ahead of publication.
- Keep input rights records; drop pre-release material when the
  programme ends.

File publication requests inside the same 30-day window as flight
checks and agency reviews. See sections 12-13 of the research report for
the full open list.

## 7. Honest latency framing (what to say on stage)

Cloud plus offline, in that order:

1. Pre-mission compilation (cloud): ingest the slice, precompute the
   SDF / flood / shadow tensors, run the SHORT or LONG search against
   the single scalar.
2. Air-gapped execution (offline): sever connectivity, run the winning
   solver deterministically from cached tensors.

Stage wording:

- "This controller was evolved over a 24 h CD run and now executes
  offline." Never "evolving live."
- "The flight path is advisory mission planning inside a 0-60 m
  envelope, checked against staged OneMap-style zone data in the
  simulator." Do not present it as a cleared flight plan. Checks shown
  are simulated (broadcast ID, notice-to-fly, clearance feeds) and
  landing needs permits and agency sign-off.
- "Water routing uses a screening depth tensor. The water agency owns
  operational hydrology." Do not present street-level depth as a
  forecast.
- "Ramp and curb flags are triage from perspective images with human
  review required before anyone depends on them." Only
  human-verified ramps ship in the slice (10 of 20).
- "GNSS and wind layers are analytic proxies, not precision models." Do
  not quote centimetre accuracy or live wind data.

Kill metrics to watch before claiming a win: evaluator over 5 s per
candidate, winner that drops sharply on the holdout estate, or fixed
positions baked into the winner. Any two of those mean pause and pivot
to the human-in-loop auditor line (Research sec. 10).
