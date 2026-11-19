# AETHER-SG: Concept Specification

**Autonomous Emergency Trajectory, HADR & Evacuation Twin for Dense Urban Canyons**

---

## 1. Executive Overview

In high-density Housing & Development Board (HDB) estates and Central Business District (CBD) commercial canyons in Singapore, standard 2D navigation and coarse GIS datasets fail during flash floods, hazardous material incidents, and sudden out-of-hospital cardiac arrests (OHCA). 

Autonomous emergency drones face unmapped micro-obstacles, including tree canopies, overhead structures, sheltered linkways, and Venturi wind channeling. Simultaneously, wheelchair users and mobility-impaired residents face flooded curb cuts, sudden level changes, stairs, and obstructed walkways.

AETHER-SG is an edge-first digital twin platform that ingests 3D building geometry, street panoramas, and elevation rasters for any Singapore urban polygon, compiles an offline physics and collision tensor, and computes coupled air-and-ground rescue operations:
- **Air Operations**: Physics-validated, low-altitude (0–60 m) micro-drone corridors avoiding aerodynamic turbulence and GNSS shadow zones.
- **Ground Operations**: Dynamic barrier-free evacuation routes verified against flood depth thresholds and curb ramp accessibility.

---

## 2. Core Architecture

```
[Geospatial Ingestion: Aleph / OneMap]
                │
                ▼
  [3D SDF / Terrain / Obstacle Tensors]
                │
         ┌──────┴──────┐
         ▼             ▼
[Air Corridor Solver] [Ground Evacuation Solver]
   (0–60m Drone A*)     (Wheelchair A* + Flood Mask)
         └──────┬──────┘
                ▼
[Offline 3D Tactical Viewer / Controller]
```

### 2.1 Zero-Key Ingestion & Sovereign Edge Execution
Using lightweight geospatial extraction tools (`alephgeo`), the system captures a target sector's `.glb` 3D building mesh, elevation raster, OpenStreetMap (OSM) vector geometry, and 360° Street View panoramas without paid external API dependencies. Once downloaded into local cache, the pipeline executes completely offline in a sovereign airgap, requiring zero active internet connection during emergency response.

### 2.2 Cross-Cued Spatial Knowledge & Semantic Fusion
Standard top-down satellite imagery cannot resolve features underneath HDB void decks or covered walkways, while raw vector maps miss overhanging rain trees, temporary construction barriers, and curb bollards.
- **Vision-Language Triage**: Local vision models scan Street View perspective crops to flag semantic ground hazards (stairs, missing curb ramps, construction hoardings, low overhead clearance) and map them onto OSM road/walkway nodes.
- **Volumetric Voxelization**: The `.glb` 3D building geometry and terrain elevations are rasterized into a 3D Signed Distance Field (SDF) occupancy grid for sub-meter collision verification.

### 2.3 Physics-Informed Urban Canyon Flight Solver
Operating strictly within Civil Aviation Authority of Singapore (CAAS) low-altitude regulatory boundaries (default ceiling 60 m AMSL / 200 ft):
- **Aerodynamic Canyon Modeling**: Estimates HDB canyon wind channeling and Venturi speedups to route micro-UAVs through leeward wake zones.
- **GNSS Shadow Avoidance**: Computes line-of-sight blockage from adjacent high-rise blocks to penalize flight paths through satellite multipath dead zones.
- **Dynamic Waypoint Optimization**: Synthesizes collision-free 3D trajectory profiles for emergency payload delivery (e.g., automated external defibrillators).

### 2.4 Dynamic Inundation Screening & Inclusive Evacuation
- **Local Flood Screening Tensor**: Simulates localized water accumulation across low-lying terrain depressions and drainage bottlenecks.
- **Barrier-Free Routing**: Cross-references non-flooded walkable paths with verified curb ramp locations and sheltered walkway networks to guarantee accessible evacuation routes for wheelchair users.

---

## 3. Frontier Architectural Extensions

### 3.1 Civic Accessibility Twin (TOM-Mesh SG)
Existing map providers route mobility-impaired citizens down routes with unmapped vertical curbs (>10 cm), steep gradients (>1:12), or obstructed pathways. This subsystem continuously audits terrain elevation gradients, analyzes street-level pedestrian infrastructure for tactile paving continuity and curb ramps, and generates accessible navigation paths alongside tactile neighborhood representations.

### 3.2 GPS-Denied Visual-Terrain Localization (EQUATOR-LOC)
In dense urban canyons where GNSS signals are degraded, multipathed, or unavailable, an onboard edge agent matches real-time downward and forward camera features against the pre-cached satellite ortho-imagery, 3D building silhouettes, and terrain elevation profiles to sustain drift-free visual-inertial localization.

---

## 4. Dual-MCP Integration: Google Computational Discovery (CD)

By pairing geospatial world ingestion with Google's Computational Discovery (CD) platform (integrating Empirical Research Assistance [ERA] and AlphaEvolve), AETHER-SG evolves bespoke pathfinding heuristics tailored to specific Singapore neighborhoods.

### 4.1 Two-Phase Execution Model
1. **Pre-Mission Compilation (Cloud Search)**:
   - Evaluator compilation: Ingests the 3D mesh, terrain raster, and obstacle graphs to produce a vectorized NumPy/JAX fitness evaluator (`score_candidate.py`).
   - AlphaEvolve Heuristic Search: Evolves heuristic weights and corridor search functions across generations, optimizing multi-objective trade-offs between travel time, energy, aerodynamic drag, GNSS shadow exposure, and collision margins.
2. **Operational Deployment (Offline Edge)**:
   - The winning heuristic function is exported as a compact (<50 KB) deterministic Python/C++ controller with zero LLM runtime latency and zero hallucination risk, operating at high frequency on edge microcontrollers.

### 4.2 Extended Domain Concept: Autonomous Vector & Plume Defense
Fusing geospatial 3D topology with empirical dispersion modeling enables automated multi-drone trajectory planning for public health and civil defense:
- **Environmental Plume Modeling**: Empirical simulation of aerosol or pathogen dispersion through HDB void decks, shaded perimeter drains, and foliage.
- **Volumetric Swarm Dispersion**: AlphaEvolve discovers optimal multi-UAV release waypoints, flight altitudes, and hover durations for vector control or hazmat detection within minimal battery windows.

---

## 5. Performance & Latency Strategy

To achieve real-time response during emergency operations:
- **Vectorized SDF Lookup**: Convert 3D geometry into pre-computed 3D NumPy Signed Distance Field tensors so path evaluations run in under 10 milliseconds.
- **Pre-Compiled Heuristic Replay**: Store pre-computed heuristic configurations for key high-density districts (e.g., Toa Payoh, Jurong East), enabling instantaneous path generation upon incident alert.
