# AETHER-SG: Comprehensive Technical Feasibility & Research Report

> **Concept Reference:** [Idea.md](Idea.md)  
> **Methodology:** Systematic multi-angle research and synthesis, grounded in primary data sources across Singapore statutory boards (CAAS, SLA, PUB, SCDF, PDPC), peer-reviewed publications (*Nature*, arXiv, MDPI), and the `alephgeo` open-source extraction suite.  
> **Annotations:** Derived calculations are flagged `[DERIVED]`; empirical inferences are flagged `[INFERENCE]`.

---

## Table of Contents

1. [Executive Verdict](#1-executive-verdict)
2. [Problem Validation & Singapore Demographics](#2-problem-validation--singapore-demographics)
3. [Aleph Geospatial Capability Audit](#3-aleph-geospatial-capability-audit)
4. [Google Computational Discovery Audit (ERA + AlphaEvolve)](#4-google-computational-discovery-audit-era--alphaevolve)
5. [Regulation, Licensing & The Clean-Room Enterprise Toggle](#5-regulation-licensing--the-clean-room-enterprise-toggle)
6. [Physics & Algorithms State of the Art (Realism Grades)](#6-physics--algorithms-state-of-the-art-realism-grades)
7. [Ground Evacuation & VLM Accessibility Audit](#7-ground-evacuation--vlm-accessibility-audit)
8. [Proposed System Architecture (Research-Grade)](#8-proposed-system-architecture-research-grade)
9. [Competitive Landscape & Operational Feasibility](#9-competitive-landscape--operational-feasibility)
10. [Red-Team Risks & Kill Criteria](#10-red-team-risks--kill-criteria)
11. [Target MVP & Engineering Scope](#11-target-mvp--engineering-scope)
12. [Open Technical Decisions](#12-open-technical-decisions)
13. [Identified Empirical Gaps](#13-identified-empirical-gaps)
14. [Primary References & Authorities](#14-primary-references--authorities)

---

## 1. Executive Verdict

> [!NOTE]
> **Summary Assessment:** SOUND as an offline planning and evolved-heuristic thesis; NOVEL as a coupled air-and-ground emergency twin; BUILDABLE as a pre-cached offline replay.

Three essential technical corrections must govern implementation:

| Original Assumption | Ground Reality | Engineering Fix |
|---|---|---|
| **0–120 m Drone Flight** ("aligned with CAAS low-altitude corridors") | Singapore default ceiling is **200 ft AMSL (~61 m)**; 400 ft is restricted to designated areas for Operator Permit holders. **No formal CAAS corridor network exists** (only UTM trials 2019–2021). | Hardcode solver ceiling to **0–60 m**; validate airspace boundaries against OneMap Drone Query. |
| **Dynamic Flood Tensor (0.5–3 m)** | PUB drainage infrastructure reduced flood-prone areas by ~99% (3,200 ha → <25 ha). Maximum observed flash floods reach **~0.3 m** (e.g., Jalan Seaview, Jan 2025). | Reframe as **micro-routing and last-100m access**. Set thresholds: 0.3 m prepare, 1.0 m evacuate, wheelchair impassable at ≥0.10–0.15 m `[INFERENCE]`. |
| **Live CD Evolution During Operations** | Google CD evolutionary searches require **6 hours (SHORT)** to **24 hours (LONG)** in cloud environments. Data capture, CD search, and regulatory telemetry require connectivity. | **Enforce two-phase pipeline:** Pre-mission cloud compilation + Airgapped offline execution. Demo via pre-computed Gen0 / Gen25 / Gen500 replays. |

The verified architecture loop:
```
Aleph Polygon Capture ──▶ Precomputed SDF/Flood/GNSS Tensors ──▶ Single-Scalar CD Evolution ──▶ <50 KB Deterministic Controller ──▶ Offline Execution
```

Each isolated component is commoditized (Virtual Singapore for data, Everdrone for flight, PUB/SUEZ for flood, SLA OneMap for ground routing). The core technical moat is the **tight coupling of 3D micro-canyon physics with evolutionary search heuristics**.

---

## 2. Problem Validation & Singapore Demographics

- **HDB Density & Urban Form:** Approximately 80% of Singapore residents reside in HDB developments across 1.28 million units (2025). Residential density reaches 8,028–8,565 persons/km² in estates like Jurong East (91% HDB) and Toa Payoh (82% HDB), with tower heights frequently exceeding 40 storeys.
- **Micro-Flood Dynamics:** 36 flood-prone areas and 22 hotspots are monitored nationally (Nov 2025). Significant rainfall events (e.g., 255.2 mm over two days at Changi, exceeding monthly means) produce localized flash inundations. True utility lies in micro-routing around water pooling, not regional flood prediction.
- **Out-of-Hospital Cardiac Arrest (OHCA):** Exceeds 2,500–4,000 cases annually in Singapore, with ~80% occurring in residential estates. Survival drops by 7–10% per minute without defibrillation. SCDF emergency response targets are 8 minutes for fire bikes (90%) and 11 minutes for ambulances (80%). The myResponder community network includes >246,000 CFRs with >5,000 public HDB AEDs deployed.
- **Drone Delivery Delta:** In dense Singapore estates, drone flight gains are measured in minutes compared to road vehicles navigating ground traffic and complex block layouts. The critical challenge is navigating the HDB vertical access and covered walkway maze.
- **Micro-Obstacles Invisible to Satellite 2D:** Covered linkways (Walk2Ride program covers >200 km; CowayMap estimates 1,133 km of sheltered walkways) and HDB void decks cannot be resolved by top-down satellite imagery. Urban canyons induce GNSS multipath errors and Venturi wind accelerations up to **3.42×** ambient speeds.
- **Operating Agencies:** Deployment requires integration with identified statutory operators: SCDF, PUB, HDB, Town Councils, LTA, SLA, and CAAS.

---

## 3. Aleph Geospatial Capability Audit (`alephgeo` v0.2.0)

- **Supported Extraction Capabilities:** Polygon and rectangle capture of satellite imagery, OSM vector data, terrain rasters, and Street View 360° panoramas with textured `.glb` city meshes. Python ≥3.11 with minimal dependencies (`Pillow`, `numpy`, `truststore`), requiring **zero proprietary API keys**.
- **Resolution Limits:** Mesh L21 ≈ 8 cm/texel; satellite zoom 18 ≈ 0.60 m/pixel; terrain zoom 14 (~9.6 m/pixel tiles derived from ~30 m SRTM source, which is **not sub-meter**).
- **Agent Integration:** Direct CLI output via `--json`, structured error codes, deterministic exit codes (0/1/2/130), and offline rebuild capabilities via `manifest.json`.
- **Known Limitations:** Heartland HDB depth varies; Street View is restricted to public vehicular roads and cannot look underneath HDB void decks or pedestrian linkways; imagery may be 1–3 years stale; undocumented endpoints have no formal uptime Service Level Agreements (SLAs).
- **Scale Envelope `[DERIVED]`:** A 1 km² capture requires 42–49 satellite tiles (z18) and 200–330 Street View stops (~40–200 MB, scaling to 0.5–2 GB if full-sphere z3 is pulled). Mesh data ranges from 100 MB to 1 GB+.
- **Geometry Characteristics:** Generated `.glb` models represent photogrammetric surfaces without semantic building entity tags or height metadata. Semantic tagging must be fused from OpenStreetMap vector layers (`map.osm`).

---

## 4. Google Computational Discovery Audit (ERA + AlphaEvolve)

- **Platform Architecture:** Integrates AlphaEvolve (arXiv:2506.13131, evolutionary code mutation and parallel scoring) with Empirical Research Assistance (ERA, *Nature* 2026, s41586-026-10658-6, LLM-guided scientific code optimization).
- **Notebook Interface Contract:**
  - **Setup Cells (Fixed):** Environment initialization, tensor loading, and helper routines.
  - **Solution Cells (Mutable):** Bound between explicit `# Begin mutable cells` and `# End mutable cells` comments.
  - **Scoring Cells (Fixed):** Emits exactly one instance of `Candidate metric to be maximized: <score>` on stdout.
- **Execution Sandbox:** 8 vCPU, 32 GiB RAM, **CPU-only**, Python 3.12, strict network isolation. Input assets staged under `/data/input/<displayName>`.
- **Fit for AETHER:** AlphaEvolve excels at evolving heuristic functions (e.g., A* cost weights, neighbor pruning rules) against fixed physical evaluation tensors. Single-scalar formulation with hard rejection filters (for collisions and boundary violations) is mandatory.
- **Search Latencies:** Realistic evolutionary runs require 6 to 24 hours. Offline pre-mission compilation is the only supported operational pattern.

---

## 5. Regulation, Licensing & The Clean-Room Enterprise Toggle

### 5.1 Airspace & UAV Regulations (CAAS)

> [!WARNING]
> **Regulatory Constraints:**
> - Unmanned aircraft exceeding 250 g require mandatory registration.
> - Commercial/emergency operations require Operator Permits and Class 1 Activity Permits.
> - Beyond Visual Line of Sight (BVLOS) applications require a minimum of **3 months** advance review (AC ANR-101-2-1 Rev 8).
> - Broadcast Remote Identification (B-RID) is mandatory under Singapore law.
> - Default altitude ceiling is strictly **200 ft AMSL (~61 m)**. OneMap Drone Query is the authoritative boundary provider.

### 5.2 Data Licensing & Enterprise Architecture

> [!CAUTION]
> **Data Compliance Note:**
> Scraping Google Street View, satellite tiles, and 3D meshes for commercial distribution violates Google Maps Platform Terms §3.2.3. Aleph serves exclusively as a rapid research prototyping harness.
>
> **The Clean-Room Toggle:** In enterprise deployment, data ingestion adapters swap cleanly to licensed Singapore Open Data (OneMap SG, SLA 3D CityGML, and PUB sensor telemetry) without altering downstream routing and physics algorithms.

---

## 6. Physics & Algorithms State of the Art (Realism Grades)

| Subsystem | Realism Grade | Technical Evidence & Characteristics |
|---|:---:|---|
| **Urban Canyon Wind** | **B** (Surrogate) / **F** (Live CFD) | Flow speedup reaches 3.42× in aligned corridors. Full Computational Fluid Dynamics (CFD) requires hours to days; Large Eddy Simulations (LES) require weeks. Fast neural surrogates (10 h/32 CPU → ~1 s inference) provide viable corridor risk estimates. |
| **GNSS Shadow Modeling** | **A / B** | 3D shadow matching algorithms reduce multipath error from 25.9 m to 6.1 m RMS. Requires 3D building geometry and offline satellite ephemeris calculations. |
| **SDF Collision Checking** | **A** (Concept) / **C** (Dense Grid) | O(1) lookup with spatial gradients (e.g., Voxblox/nvblox). `[DERIVED]` A 1000×1000×120 m grid at 1 m resolution yields 120M voxels (~480 MB float32), making sparse octree structures mandatory for sub-meter resolution. |
| **Flood Modeling** | **B** (Screening) / **F** (Hydrodynamics) | Bathtub elevation screening provides basic ponding estimates but overestimates behind linkways and barriers. Operational hydrology requires drainage network integration. |
| **Multi-UAV Routing** | **Sound** (Heuristics) | Multi-Agent Pathfinding (MAPF) is NP-hard. Heuristic evolution (rather than exact global solving) is the defensible mathematical framing. |
| **<10 ms Evaluator** | **Sound** (Tensor Lookup) | Pre-computed 3D NumPy array lookups and vectorized matrix operations achieve sub-10 ms cycle times per trajectory evaluation. |

---

## 7. Ground Evacuation & VLM Accessibility Audit

- **Baseline Landscape:** SLA OneMap Barrier-Free Access (BFA) already indexes over 1,000 km of accessible pathways with voice guidance. AETHER's differentiation is **real-time flood pooling avoidance, obstruction freshness, and street-level triage**.
- **Vision-Language Triage Reliability:** Local vision models excel at detecting ramp presence, pedestrian crosswalks, stairs, and major barricades. However, VLMs cannot reliably quantify curb heights in centimeters or measure fine slope angles from perspective imagery alone.
- **Terminology Standard:** Never claim "guaranteed wheelchair-passable." Use **"verified ramp-accessible, unblocked under screening criteria."**
- **Evacuation Destination Standards:** Under Singapore Fire Safety Code, residential HDB blocks are not required to provide dedicated internal disabled refuge areas; residents defend in place or evacuate to ground-level exterior assembly areas.

---

## 8. Proposed System Architecture (Research-Grade)

```
                    ┌────────────────────────────┐
                    │      Pre-Mission Phase     │
                    │   (Cloud Connected / CD)   │
                    └─────────────┬──────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      [Geospatial Extraction]            [Computational Discovery]
     • 3D Building Mesh (.glb)          • AlphaEvolve Code Search
     • OSM Vector Topography            • Physics Surrogate Model
     • Elevation Rasters & Pano         • Heuristic Evolution
                 │                                 │
                 └────────────────┬────────────────┘
                                  │ Winning Compact Controller (<50 KB)
                                  ▼
                    ┌────────────────────────────┐
                    │     Operational Phase      │
                    │   (100% Offline Airgap)    │
                    └─────────────┬──────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      [3D Air Corridor Solver]          [Ground Evacuation Router]
      • 0–60 m CAAS Ceiling             • Wheelchair Incline Limits
      • GNSS Multipath Avoidance        • Dynamic Flood Masking
      • Wake Turbulence Margin          • Verified Curb Ramp Links
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                 [Tactical 3D Visualizer & Control]
```

- **Dual-MCP Coordination:** Ingestion server (`alephgeo`) provides spatial ground truth; CD server (`science.googleapis.com/mcp`) optimizes algorithmic heuristics.
- **Airgap Security Boundary:** Cloud connection is utilized strictly during data extraction and evolutionary search. Emergency deployment runs completely offline using local edge weights and pre-compiled Python/C++ binaries.

---

## 9. Competitive Landscape & Operational Feasibility

| Operation Phase | Feasibility & Technical Strategy |
|---|---|
| **1. Geospatial Ingestion** | High technical feasibility with cached hero districts (Toa Payoh, Jurong East). Live fetching over wide polygons is gated by network bandwidth and source availability. |
| **2. Offline Sovereign Airgap** | Completely feasible. All models, evaluation tensors, and static viewer assets run locally without external telemetry. |
| **3. Incident Injection & Screening** | Local flood screening and obstacle tagging apply instantaneous masks across the pre-compiled graph. |
| **4. Coupled Trajectory Replay** | High visual and operational impact. Evolved heuristic trajectories render seamlessly in Three.js tactical map interfaces alongside ground evacuation corridors. |

---

## 10. Red-Team Risks & Kill Criteria

### Identified Risk Vectors
1. **Evaluator Gaming:** Evolved algorithms may exploit mesh discretization artifacts or non-watertight geometry rather than learning genuine physics.
2. **Regulatory Barriers:** Autonomous BVLOS operations over populated Singapore estates are legally restricted without dedicated statutory authorization.
3. **Elevation Accuracy:** Open elevation datasets lack vertical resolution for curb-level (10 cm) water routing.
4. **VLM False Negatives:** Undetected walkway obstacles could misdirect mobility-impaired individuals into dead ends.

### Kill / Pivot Thresholds

> [!WARNING]
> If any **two (2)** of the following conditions are met, halt core UAV development and pivot to the **TOM-Mesh Human-in-the-Loop Accessibility Auditor**:
> 1. [ ] Geospatial fetch for a 500 m sector exceeds 90 seconds or fails >20% of extraction attempts.
> 2. [ ] Fitness evaluator exceeds 5 seconds per candidate on 8 vCPU, or the evolved winner degrades >15% on a held-out estate.
> 3. [ ] Vision model ramp recall drops below 90% against human ground truth on Singapore walkway imagery.
> 4. [ ] Inundation routing conflicts with validated PUB drainage data on >20% of network edges.
> 5. [ ] CAAS formal pre-consultation confirms absolute prohibition of BVLOS emergency drone flights in urban estates.
> 6. [ ] Edge vision inference exceeds 10 seconds per panorama on target field hardware.

---

## 11. Target MVP & Engineering Scope

- **MVP Deliverables:**
  - One pre-cached 200×200 m Toa Payoh urban sector (OSM road network + verified curb ramps).
  - Offline 3D Signed Distance Field (SDF) tensor and lightweight Three.js tactical visualizer.
  - Pre-computed evolutionary generations (Gen0, Gen25, Gen500) showcasing trajectory optimization over time.
  - Coupled dual-routing: Low-altitude drone corridor (0–60 m) and flood-aware accessible ground route.
- **Stretch Goals:**
  - Wind velocity surrogates trained on urban canyon CFD simulations.
  - Direct integration with SLA OneMap3D CityGML data layers.
  - Dedicated lightweight edge detection models for Singapore-specific curb ramp designs.
- **Explicit Exclusions:**
  - Real-world flight operations or physical drone control.
  - Hydrodynamic reservoir-level flood simulations (owned by PUB).
  - "Guaranteed obstacle-free" navigation claims.

---

## 12. Open Technical Decisions

1. **Primary Evaluation Sector:** Select and benchmark a standardized urban sector (e.g., Toa Payoh Central vs. Jurong East).
2. **Statutory Engagement:** Determine engagement strategy with SCDF, CAAS, and SLA for data access and operational validation.
3. **Fitness Metric Balancing:** Finalize scalar penalty weights balancing flight duration, aerodynamic drag, GNSS shadow exposure, and ground route detour ratios.
4. **Edge Hardware Target:** Benchmark target runtime performance across embedded edge accelerators (e.g., NVIDIA Jetson Orin NX).

---

## 13. Identified Empirical Gaps

- Empirical measurement of end-to-end `alephgeo` capture times and mesh depth across various Singapore HDB building typologies.
- Systematic benchmark of AlphaEvolve generation counts versus fitness improvement on dense urban pathfinding problems.
- Field validation of vision model classification accuracy across diverse Singapore covered walkway designs and sheltered linkways.

---

## 14. Primary References & Authorities

- **Civil Aviation Authority of Singapore (CAAS):** Advisory Circular AC ANR-101-2-1 (Unmanned Aircraft Operations), Air Navigation Act.
- **Singapore Land Authority (SLA):** OneMap3D API Documentation, Barrier-Free Access Network Specifications.
- **Public Utilities Board (PUB):** Code of Practice on Surface Water Drainage, Singapore Flood Alleviation History & Hotspot Registry.
- **Singapore Civil Defence Force (SCDF):** Fire Safety Code, Community First Responder Framework.
- **DeepMind & Google Research:** AlphaEvolve: Evolutionary Coding Agents (arXiv:2506.13131); Empirical Research Assistance (*Nature* 2026, s41586-026-10658-6).
- **Urban Aerodynamics & Robotics:** Canyon wind amplification studies (MDPI), 3D Shadow Matching for Urban GNSS Positioning.
