# AETHER-SG — Aleph Feature Map

**Status: research plumbing verified against a real Singapore capture. Not a production data pipeline.**

Honest framing (Research-AETHER-SG.md §5): Aleph downloads Google Street View, satellite and 3D-mesh layers through undocumented endpoints in breach of Google Maps Platform Terms §3.2.3 + Maps Additional Terms. Everything in this file that touches Google is **research-only, never redistributed** and must not ship in a production basemap. OSM (ODbL) and AWS Terrain are lawfully reusable. The AETHER viewer and CD notebook stay on the synthetic slice until production re-sourcing (OneMap3D / Virtual Singapore / own capture) is approved.

Python: `C:\VSCode Folder\Ather-SG\.venv\Scripts\python.exe` (alephgeo installed from `vendor\Aleph`).

Real capture studied: `captures\aleph-20261119T022946Z-1upsr2v9` — Toa Payoh polygon `1.3311,103.8461 → 1.3311,103.8479 → 1.3329,103.8479 → 1.3329,103.8461`, `include=[streetview,satellite,osm,terrain,mesh]`, `state=complete`. 118 `streetview/photos/*.jpg`, `map.osm` 1.45 MB (1,041 nodes / 94 ways / 110 relations), `terrain.tif` 512×512 Float32 (min −2.69 m, max 87.57 m, mean 15.86 m), `mesh.glb` 7.57 MB, `satellite.tif` 336×336 RGBA EPSG:3857 + `satellite.png`, `manifest.json` + `README.txt` + `streetview/photos.geojson` (118 features) + `paths.geojson` + `plan.svg` + `satellite/patches.geojson` + `mesh/nodes/*.bin` + `terrain/tiles/`. Satellite-only companion: `captures\aleph-20261119T022852Z-tp82suab` (`satellite` only, zoom 19, 672×672 `satellite.tif` / `satellite.png`, `result.json` complete).

---

## 1. Feature matrix

Every `alephgeo` capability from `vendor\Aleph\docs\cli.md` mapped to the AETHER wrapper, CLI, output, licence and production status. AETHER wrappers live in `src\aether\ingest.py` (`AlephError`, argv builders, `run_aleph`) and `src\aether\converter.py` (`convert_capture` / `capture.json`) — both import-defensive: `tests\test_aleph.py` skips these tests when they are absent.

| # | Aleph capability | `alephgeo` CLI | AETHER file / function | AETHER CLI / argv | Output | Licence per layer | Production? |
|---|---|---|---|---|---|---|---|
| 1 | Place search / reverse-geocode / nearby POIs | `resolve "name"`, `resolve --at LAT LON`, `resolve --at … --nearby --radius M --streets --limit N` | `src\aether\ingest.py :: build_resolve_argv(*, query, at, streets, nearby, radius, limit)`, `run_aleph(argv)` | `alephgeo resolve … --json` | stdout JSON (id for `--match`), no folder | Geocoder: Photon / OSM · ODbL; provider to avoid redistribution limits | Yes, with attribution; keep `© OpenStreetMap contributors` |
| 2 | Street View single view (point / place / pano-id) | `streetview --at LAT LON [--heading --radius --pitch --fov --look-at]`, `--place NAME [--best-match/--match]`, `--pano-id ID` | `ingest.py :: build_streetview_single_argv(*, at, place, pano_id, heading, radius, pitch, fov, look_at, full_sphere, sphere_zoom, streetview_format, workers, output)` | `alephgeo streetview --at … --json -o captures` | `aleph-*/result.json` + `001.jpg…` or sphere tiles | Google Maps Platform Terms — **research-only, never redistributed** | **No** — drop or re-source for production |
| 3 | Street View along a street (stops / step, view) | `streetview --street NAME [--stops N / --step M] [--view forward\|backward\|left\|right\|both] [--route N --reverse]` | `ingest.py :: build_streetview_street_argv(*, street, stops, step, view, route, reverse, …)` | `alephgeo streetview --street … --json` | numbered photos + `result.json` with pano_id/location/date per photo; gaps listed | Google — **research-only** | **No** |
| 4 | Street View full-sphere panorama | `streetview … --full-sphere [--sphere-zoom 0–5]` | same as #2/#3 via `full_sphere`, `sphere_zoom` params | `alephgeo streetview … --full-sphere --json` | one 360° panorama per stop (heading/pitch/fov ignored) | Google — **research-only**; faces/plates are PDPA personal data | **No** — blur before any KAG storage |
| 5 | Satellite patch / rectangle / tile | `satellite --at/--place --size M --zoom 1–21`, `--bbox S W N E`, `--tile Z/X/Y` | `ingest.py :: build_satellite_argv(*, at, place, bbox, tile, size, zoom, satellite_format, workers, output)` | `alephgeo satellite … --json -o captures` | `satellite.png` + `satellite.tif` (COG, EPSG:3857, north-up) + `result.json` (exact bounds, zoom, source) | Google — **research-only**; transparent = missing/outside polygon | **No** as Aleph cache; production must use OneMap / other licensed imagery |
| 6 | Capture area — 5 sources | `capture create --bbox S W N E / --polygon LAT LON… --sources streetview satellite osm terrain mesh --depth --step --fov --satellite-zoom --terrain-zoom --mesh-level --*-workers --delay --yes/--plan --json -o DIR` | `ingest.py :: build_capture_argv(*, bbox, polygon, sources, depth, step, fov, satellite_zoom, terrain_zoom, mesh_level, workers…, output, yes)` | `alephgeo capture create --bbox … --sources streetview satellite osm terrain mesh --yes --json -o captures` | `captures\aleph-*/manifest.json, README.txt, satellite.png/.tif, terrain.tif, mesh.glb, map.osm, streetview/{photos,photos.geojson,paths.geojson,plan.svg}, satellite/{patches,plan.svg}, terrain/tiles/, mesh/nodes/` | Mixed — see §3 per layer | Mixed — **only OSM+terrain rows are production-safe as captured** |
| 7 | Capture resume | `capture resume FOLDER [--yes] [--json]` | `ingest.py :: build_capture_resume_argv(folder, yes)` | `alephgeo capture resume FOLDER --json` | continues a stopped/planned run in-place (keeps downloaded files) | Inherits — offline merge, no new Google fetch | Yes **iff** the run's remaining fetch needs no Google layer you cannot ship |
| 8 | Capture export (offline rebuild) | `capture export FOLDER [--json]` | `ingest.py :: build_capture_export_argv(folder)` | `alephgeo capture export FOLDER --json` | rebuilds `satellite.png/.tif, terrain.tif, mesh.glb` + `manifest.json` from `terrain/tiles/` + `mesh/nodes/` **offline** | Airgap boundary: after capture, export/rebuild is network-free | Yes — the legitimate airgap step (see Research §3) |
| 9 | Dashboard (browser capture + explorer) | `dashboard -o DIR --port 8100 --no-browser` | `ingest.py :: build_dashboard_argv(output, port)` | `alephgeo dashboard -o captures --port 8101` (AETHER uses **8101**, §6) | HTTP UI at `http://127.0.0.1:8101`: draw rect/polygon, preview+estimate, capture, toggle layers, 3D mesh/terrain, resume | Basemap `OpenFreeMap` (needs internet); captures inherit per-layer licence | Yes for OSM/terrain-only captures; not for publishing Google layers |
| 10 | HTTP cache + `--refresh` | `--cache-dir DIR` (default `~/.cache/aleph`, `$XDG_CACHE_HOME/aleph`), `--refresh` on all commands except `capture export` | `ingest.py :: run_aleph(argv, cache_dir, refresh)` passes `--cache-dir` / `--refresh` through | `alephgeo … --cache-dir … --refresh --json` | reuses Geofabrik regional OSM + provider responses across runs | Cache does not change licence — cached Google bytes are still ToS-restricted | Yes — set a project-local `cache-dir` for reproducibility |
| 11 | `--json` scripting contract | `--json` on every command | `ingest.py :: run_aleph` → dict; raises `AlephError(code, message, **details)` when `{"status":"error",…}` | `alephgeo … --json > result.json`; exit `0` success / `1` request failed / `2` invalid args / `130` interrupted | Single JSON object on stdout; progress on stderr; error codes `invalid_arguments, place_not_found, ambiguous_place(+candidates), no_coverage, provider_unavailable, request_failed, interrupted`; `folder` points at saved files even on failure | No licence impact; enables headless/agent wrapping | **Yes** — this is the MCP-wrap seam (Research §3) |

Notes: `--delay SECONDS` (pacing between requests) exists on `resolve, streetview, satellite, capture create`. `--geocoder URL` / `ALEPH_GEOCODER_URL` exists on `resolve, streetview, satellite`. `--workers N` (1–64, default 8) exists on `streetview, satellite`; per-source `--*-workers` on `capture create`. `--tile-format` / `--satellite-zoom` / `--terrain-zoom` / `--mesh-level` / `--depth` / `--step` / `--fov` / `--streetview-format` are capture-only (see row 6). `capture create` prompts for confirmation unless `--yes`; `--plan` writes a plan without downloading (start later with `capture resume`).

---

## 2. Capture folder anatomy (what AETHER reads)

```
captures\aleph-*/            ← one folder per `capture create` / `streetview` / `satellite`
├── manifest.json            ← polygon, options{include,step,fov,depth,formal…,satellite_zoom,terrain_zoom,mesh_level,workers}, started_at/finished_at, state, stages[]
├── README.txt               ← file guide + credits (Google / OSM / AWS)
├── satellite.png / satellite.tif   ← COG RGBA EPSG:3857 (north-up, transparent gaps)
├── terrain.tif              ← metres EPSG:3857 (Int16 ≤z12 else Float32); bbox coverage, exceeds polygon
├── mesh.glb                 ← textured Google Earth photogrammetry (m, y-up, y=0 low point)
├── map.osm                  ← OSM XML (WGS84, full tags; contributor names omitted by Geofabrik)
├── streetview/photos/*.jpg  ← e.g. 000001_lorong-1-toa-payoh_p0001_s0001_left_h008.jpg (h = heading°)
├── streetview/photos.geojson  ← per-photo pano_id, lat/lon, heading, path_id, imagery_date, filename, source_url
├── streetview/paths.geojson + plan.svg
├── satellite/patches.geojson + patches/ + plan.svg
├── terrain/tiles/           ← keep for resume/export
└── mesh/nodes/*.bin         ← keep for resume/export
```

AETHER's `src\aether\converter.py :: convert_capture(capture_dir, out_dir)` normalises this to `out_dir/capture.json` + copied `mesh.glb` / `terrain.tif` (+ `satellite.tif/png` when present) so the offline viewer/SDF pipeline never reads Aleph's `manifest.json` directly. `tests\test_aleph.py::test_converter_on_real_capture` asserts: mesh copied byte-identical, `capture.json.photos > 50` (118 on the real capture), `osm` counts > 0 (nodes/ways/relations), `terrain` stats finite (`min/max/mean`).

---

## 3. Licence per layer — what you may ship

| Layer | Source & licence | AETHER claim allowed | Production use |
|---|---|---|---|
| **OSM (`map.osm`, `place`, geocoding)** | OpenStreetMap contributors, **ODbL** `https://opendatacommons.org/licenses/odbl/1-0/` https://www.openstreetmap.org/copyright | "Map data © OpenStreetMap contributors, ODbL" — commercial use allowed; shared modified DBs stay ODbL | **Yes** with attribution |
| **Terrain (`terrain.tif`)** | AWS Terrain Tiles, sources at `https://github.com/tilezen/joerd/blob/master/docs/attribution.md` — open | "Terrain: AWS Terrain Tiles; credit its sources" | **Yes** with the listed credits; note z14 ≈ 9.6 m/px tiles over ~30 m SRTM source — **not sub-metre** (Research §3) |
| **Street View (`streetview/`)** | © Google and its providers. Downloaded via undocumented APIs contrary to Google Maps Platform Terms §3.2.3 + Maps Additional Terms (27 Jan 2026) | "Research only; not for redistribution or production basemap" | **No** — strip before any published demo; uploading Google-derived data to Computational Discovery compounds the breach |
| **Satellite (`satellite.*`)** | © Google and its providers (same ToS) | Same research-only label | **No** as an Aleph cache |
| **Mesh (`mesh.glb`)** | © Google and its providers (same ToS) | Same research-only label | **No**; photogrammetry is **non-watertight** — SDF sign determination is the hard part (Research §6) |

Capture credits above are copied verbatim from `captures\aleph-20261119T022946Z-1upsr2v9\README.txt` and `vendor\Aleph\README.md`. AETHER never claims redistribution or production rights for Google layers.

---

## 4. How to run a new capture

All commands use the project venv so `alephgeo` resolves to the vendored build:

```powershell
$py = "C:\VSCode Folder\Ather-SG\.venv\Scripts\python.exe"
& $py -m pip install -e vendor\Aleph
```

**CLI capture (all 5 sources, Toa Payoh-scale rectangle):**

```powershell
& "C:\VSCode Folder\Ather-SG\.venv\Scripts\python.exe" -m alephgeo capture create `
  --bbox 1.3311 103.8461 1.3329 103.8479 `
  --sources streetview satellite osm terrain mesh `
  --depth roads --step 30 --fov 75 --satellite-zoom 18 --terrain-zoom 14 --mesh-level 21 `
  --streetview-workers 8 --satellite-workers 8 --terrain-workers 8 --mesh-workers 8 `
  --yes --json -o "C:\VSCode Folder\Ather-SG\captures" > capture.json
```

* `mesh` must be requested explicitly; the default four are `streetview satellite osm terrain`.
* Drop to `--sources satellite` for a satellite-only run (matches `aleph-20261119T022852Z-tp82suab`). Drop `mesh` for a faster run.
* `--plan` writes the plan without downloading; continue later with `capture resume`.

**Resume / export (offline-safe):**

```powershell
& $py -m alephgeo capture resume "C:\VSCode Folder\Ather-SG\captures\aleph-*" --yes --json
& $py -m alephgeo capture export "C:\VSCode Folder\Ather-SG\captures\aleph-*" --json
```

**Quick probes:**

```powershell
& $py -m alephgeo resolve "Lorong 2 Toa Payoh, Singapore" --json
& $py -m alephgeo streetview --place "Lorong 2 Toa Payoh, Singapore" --best-match --json -o captures
& $py -m alephgeo satellite --at 1.33165 103.84619 --size 200 --zoom 19 --json -o captures
```

**Dashboard (browser):** AETHER's tactical viewer owns port 8100 (`python -m http.server 8100 --directory viewer`, `src\aether` pipeline). Run Aleph on **8101** to avoid collision:

```powershell
& $py -m alephgeo dashboard -o "C:\VSCode Folder\Ather-SG\captures" --port 8101
# open http://127.0.0.1:8101 → draw rect/polygon → pick sources/zooms → preview → Start
```

The dashboard basemap is OpenFreeMap (internet required); capturing itself needs connectivity, but `capture export` and the AETHER offline execution need none.

**AETHER wrapper (agent/script path):** call `src\aether\ingest.py:run_aleph(argv)` (always appends `--json`, parses stdout, raises `AlephError` on `status=error`) or the `build_*_argv` helpers, then `src\aether\converter.py:convert_capture(capture_dir, out_dir)` to produce `capture.json`. Verify with `& $py -m pytest tests\test_aleph.py -v`.

---

## 5. `--json` scripting contract

Every command prints a **single JSON object on stdout**; progress/messages go to **stderr**. Success objects carry `status` plus command-specific fields (`folder`, `path`, `bounds`, `place`, `candidates`, …). Failure objects are `{"status":"error","error":{"code":…,"message":…}}` with codes `invalid_arguments`, `place_not_found`, `ambiguous_place` (plus `candidates` — choose with `--match TYPE/ID` or `--best-match`), `no_coverage`, `provider_unavailable`, `request_failed`, `interrupted`. Exit codes: `0` success, `1` request failed, `2` invalid arguments, `130` interrupted. If files were already saved, `folder` still points at them. Set `NO_COLOR` to disable coloured stderr. AETHER's `run_aleph` raises `AlephError` (carrying `code`/`message`) on `status=error` — tested by `tests\test_aleph.py::test_aleph_error_on_status_error`.

---

## 6. Singapore coverage findings (from the real capture)

Measured on `aleph-20261119T022946Z-1upsr2v9` (200×200 m Toa Payoh slice, `--depth roads --step 30`):

* **Street View density:** 118 photos from road sampling — ~2 per stop (left+right), headings consistent with OSM node order. Imagery from 2024-03 (sampled feature), so freshness is 1–3 yr stale as Research §3 warns.
* **Single-point streetview is brittle in Singapore.** A bare `streetview --at LAT LON` picks the nearest panorama within `--radius` (default **50 m**). HDB estates have many set-back nodes where the nearest pano is >50 m or on a service road. **Always pass `--radius 100`–`200` (or higher) for a single-point probe, or prefer `--street NAME` / `capture create` road sampling** which snaps to the road graph (`nearest-road` + `road_distance` in `photos.geojson`). The ingest builders default single-point `radius` to 75 m for this reason; dashboard `capture` sampling is the robust path.
* **Full-sphere cost:** non-sphere `fov 75` photos are small; `--full-sphere --sphere-zoom 3` multiplies size roughly 10–20× per stop (Research envelope 0.5–2 GB @ 1 km²). The Toa Payoh capture used perspective crops (`full_sphere:false`) — also the VLM-preferred framing (perspective crops −0.41 vs panoramas −0.25, Research §7).
* **OSM completeness:** 1,041 nodes / 94 ways / 110 relations in 200×200 m is sufficient for building/road anchoring but does **not** encode linkways/void decks — the exact blind spot in the thesis. Street View `depth` only follows `main/roads/all` paths; under-void-deck coverage must be verified per-estate.
* **Terrain fidelity:** `terrain.tif` 512×512 at z14 is fine for screening but **too coarse for 0.5 m flood steps** (~30 m SRTM source, 9.7–16 m RMSE, Research §7). Flood screening must be validated against a PUB-grade DEM before operational use.
* **Mesh:** `mesh.glb` 7.5 MB at L21 (≈8 cm/texel) covers the slice; per-estate depth below L21 is UNVERIFIED island-wide (Research §3 #1 empirical gap).
* **Satellite:** `satellite.tif` 336×336 RGBA @ z18 (≈0.60 m/px) is context-only; transparent pixels mark out-of-polygon/missing tiles, not errors.

---

## 7. Checks that viewer/notebook stay untouched

`tests\test_aleph.py::test_notebook_viewer_untouched` asserts `cd\aether_evolve.ipynb` still has 6 cells, `# Begin/End mutable cells` markers, `DATA_DIR` + `toapayoh_train_slice.npz`, exactly one `Candidate metric to be maximized:`, and `viewer\data.json` label remains `SIMULATED … not for navigation`. The converter writes only to the caller-supplied `out_dir`; it never edits `viewer/` or `cd/`.

---

## 8. Open gaps (not claimed)

* No measured Singapore Aleph benchmark (time/bytes per source mix, rate-limit thresholds, per-estate mesh depth/freshness) — Research §13 #1.
* Production re-sourcing for the eye-level layer (OneMap3D gated, Virtual Singapore closed) — see Research §5 "Toggle verdict: interface parity YES, data parity NO".
* PDPA: faces/plates in spheres must be blurred before any downstream storage.
