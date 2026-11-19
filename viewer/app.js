/* AETHER-SG real-capture viewer.
   Renders ONLY ./capture/capture.json plus the mesh / satellite / photo
   assets it indexes. No invented scene data: if the capture file is absent
   the page shows an empty state with the generation command.
   Drone solver envelope 0-60 m; flood screening with wheelchair block at
   depth >= 0.10 m; Google-derived layers are research-only.
   Scene frame: metres, origin at capture bbox centre, x east, z south,
   y up (matches the Aleph mesh.glb frame). Backend slice coords are
   0..200; the viewer frame is coord - 100. */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const WHEELCHAIR_BLOCK_M = 0.10;
const TAN30 = Math.tan(THREE.MathUtils.degToRad(30));
/* Warm charcoal ops palette (amber / olive / rust only). */
const PAL = {
  bg: '#16130E', gridA: '#3A342A', gridB: '#201C15',
  hemiTop: '#F3EFE8', hemiGround: '#3A342A', sun: '#FFF7E8',
  meshEdge: '#A79E91', flood: '#6B8E23', block: '#D97706',
  shadowLow: '#A79E91', shadowMid: '#D97706', shadowHigh: '#C2410C',
  drone0: '#C2410C', drone25: '#D97706', droneFinal: '#6B8E23',
  ground: '#F3EFE8', photo: '#D97706', marker: '#F3EFE8',
  losClear: '#6B8E23', losBlocked: '#C2410C', swath: '#D97706',
  buildingEdge: '#D97706', pad: '#6B8E23', padRing: '#D97706',
};

const $id = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '—');

/* ---------- tolerant accessors (unknown schema keys are skipped) ---------- */
const pick = (obj, keys) => { for (const k of keys) if (obj && obj[k] != null) return obj[k]; return undefined; };
function ciGet(obj, key) {
  if (!obj || typeof obj !== 'object') return undefined;
  const want = String(key).toLowerCase();
  for (const k of Object.keys(obj)) if (k.toLowerCase() === want) return obj[k];
  return undefined;
}

/* Routes are scene-local metres: [x, y, z] with x east, z south, y world
   height above local datum. Ground points are [x, z] draped on the DEM. */
function normPts(arr) {
  if (!Array.isArray(arr)) return [];
  return arr.map(p => Array.isArray(p)
    ? [Number(p[0]), Number(p[1] ?? 0), Number(p[2] ?? 0)]
    : [Number(p.x ?? 0), Number(p.y ?? 0), Number(p.z ?? 0)])
    .filter(p => p.every(Number.isFinite));
}
function normGround(arr) {
  if (!Array.isArray(arr)) return [];
  return arr.map(p => Array.isArray(p)
    ? [Number(p[0]), Number(p[1] ?? 0)]
    : [Number(p.x ?? 0), Number(p.z ?? p.y ?? 0)])
    .filter(p => p.every(Number.isFinite));
}
function normDronePaths(raw) {
  const src = pick(raw, ['dronePaths', 'paths', 'pathsByGen', 'trajectories', 'drone', 'paths_by_gen'])
    ?? ciGet(raw, 'dronePaths') ?? raw;
  const out = {};
  if (Array.isArray(src)) {
    if (src.length >= 3) { out.Gen0 = normPts(src[0]); out.Gen25 = normPts(src[1]); out.final = normPts(src[2]); }
    else if (src.length >= 1) out.final = normPts(src[src.length - 1]);
    return out;
  }
  if (src && typeof src === 'object') {
    const g0 = ciGet(src, 'Gen0') ?? ciGet(src, 'gen0') ?? ciGet(src, 'baseline');
    const g25 = ciGet(src, 'Gen25') ?? ciGet(src, 'gen25') ?? ciGet(src, 'intermediate');
    const fin = ciGet(src, 'final') ?? ciGet(src, 'Gen500') ?? ciGet(src, 'gen500') ?? ciGet(src, 'winner');
    if (g0) out.Gen0 = normPts(g0);
    if (g25) out.Gen25 = normPts(g25);
    if (fin) out.final = normPts(fin);
    const single = normPts(pick(src, ['dronePath', 'path', 'points']));
    if (!out.final && single.length) out.final = single;
  }
  return out;
}
/* Extended metrics: keep every numeric field the bundle carries. */
function normMetrics(raw) {
  const src = pick(raw, ['metrics', 'metricsByGen', 'generationMetrics', 'scores']) ?? ciGet(raw, 'metrics') ?? {};
  const one = v => (v && typeof v === 'object') ? {
    score: Number(v.score ?? v.value ?? v.scalar ?? NaN),
    time: Number(v.time ?? v.timeS ?? v.duration ?? NaN),
    collisions: Number(v.collisions ?? v.collisionCount ?? 0),
    length_m: Number(v.length_m ?? v.length ?? v.pathLength ?? NaN),
    energy: Number(v.energy ?? NaN),
    shadow_exposure: Number(v.shadow_exposure ?? v.shadow ?? NaN),
    wind_exposure: Number(v.wind_exposure ?? v.wind ?? NaN),
    min_sdf_m: Number(v.min_sdf_m ?? v.minSdf ?? NaN),
    evaluator_score: Number(v.evaluator_score ?? v.evaluatorScore ?? NaN),
    avg_altitude_m: Number(v.avg_altitude_m ?? NaN),
    min_altitude_m: Number(v.min_altitude_m ?? NaN),
    max_altitude_m: Number(v.max_altitude_m ?? NaN),
    altitude_range_m: Number(v.altitude_range_m ?? NaN),
    avg_speed_mps: Number(v.avg_speed_mps ?? NaN),
    mean_wind_factor: Number(v.mean_wind_factor ?? NaN),
    max_wind_factor: Number(v.max_wind_factor ?? NaN),
    min_agl_m: Number(v.min_agl_m ?? NaN),
    mean_agl_m: Number(v.mean_agl_m ?? NaN),
    max_agl_m: Number(v.max_agl_m ?? NaN),
  } : null;
  const m = {};
  for (const k of ['Gen0', 'Gen25', 'final']) { const v = one(ciGet(src, k)); if (v) m[k] = v; }
  return m;
}
function asGrid(g) {
  if (!g) return null;
  if (Array.isArray(g)) return { nx: g[0]?.length ?? 0, nz: g.length, x0: -100, z0: -100, dx: 5, dz: 5, values: g };
  const vals = pick(g, ['values', 'elevations', 'heights', 'grid', 'data', 'cells']);
  if (!vals || !vals[0]) return null;
  return { nx: vals[0].length, nz: vals.length, x0: pick(g, ['x0', 'minX']) ?? -100, z0: pick(g, ['z0', 'minZ', 'y0']) ?? -100, dx: pick(g, ['dx', 'cell', 'res']) ?? 5, dz: pick(g, ['dz', 'cell', 'res']) ?? 5, values: vals };
}
function normBbox(c) {
  const b = c.bbox ?? ciGet(c, 'bbox');
  if (b && Number.isFinite(b.minLat) && Number.isFinite(b.maxLat)) return b;
  if (b && Number.isFinite(b.min_lat)) return { minLat: b.min_lat, maxLat: b.max_lat, minLon: b.min_lon, maxLon: b.max_lon };
  const poly = c.polygon ?? c.poly ?? [];
  if (Array.isArray(poly) && poly.length >= 2) {
    const lats = poly.map(p => Number(p[0])), lons = poly.map(p => Number(p[1]));
    if (lats.every(Number.isFinite) && lons.every(Number.isFinite)) {
      return { minLat: Math.min(...lats), maxLat: Math.max(...lats), minLon: Math.min(...lons), maxLon: Math.max(...lons) };
    }
  }
  return null;
}

/* ---------- three.js scene (warm charcoal, no glare) ---------- */
const container = $id('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(PAL.bg);
scene.fog = new THREE.Fog(PAL.bg, 260, 720);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 3000);
camera.position.set(130, 130, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 6, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;
scene.add(new THREE.HemisphereLight(PAL.hemiTop, PAL.hemiGround, 0.9));
const sun = new THREE.DirectionalLight(PAL.sun, 1.6);
sun.position.set(120, 220, 80);
scene.add(sun);
const grid = new THREE.GridHelper(400, 40, PAL.gridA, PAL.gridB);
grid.position.y = -0.5;
scene.add(grid);

let capture = null;      /* parsed capture.json */
let dem = null;          /* {nx,nz,x0,z0,dx,dz,values} local-metre DEM */
let gnssGrid = null;     /* same shape, GNSS shadow values 0..1 */
let demMin = null;
let floodLevel = 0.10;
let currentGen = 'final';
let droneMarker = null;
let photoMarkers = [];
let buildingGroup = new THREE.Group();
let buildingMeshes = [];
let buildings = [];      /* raw {x,z,w,d,h} scene-frame boxes */
let losGroup = new THREE.Group();
let losLine = null, coneMesh = null, swathRing = null, swathDisc = null;
let padGroup = new THREE.Group();
let padPos = new THREE.Vector3(74, 24, 78);
let flightT = 0;         /* 0..1 arc-length fraction along the active path */
let playing = true;
let speed = 1;
let pathCache = null;    /* {pts, cum, total} for currentGen */
let losState = { clear: true, blockedBy: -1, losPct: 100, coverage: 0, alt: 0, radius: 0 };

let meshGroup = new THREE.Group();
let satPlane = null;
let floodGroup = new THREE.Group();
let floodPlane = null;
let blockMesh = null;
let shadowMesh = null;
let droneGroup = new THREE.Group();
let droneLines = {};
let groundLine = null;
let photoGroup = new THREE.Group();
scene.add(meshGroup); scene.add(floodGroup); scene.add(droneGroup);
scene.add(photoGroup); scene.add(buildingGroup); scene.add(losGroup); scene.add(padGroup);

/* ---------- asset path candidates (capture.json paths are viewer-relative) ---------- */
function assetCandidates(p) {
  if (!p) return [];
  const out = [String(p)];
  const m = String(p).match(/^\.\.\/captures\/[^/]+\/(.+)$/);
  if (m) out.push('./capture/' + m[1]);
  else if (/^capture\//.test(p)) out.push('./' + p);
  out.push('./capture/' + String(p).split('/').pop());
  return [...new Set(out)];
}
function loadTexture(urls) {
  return new Promise((resolve, reject) => {
    const tryNext = (i) => {
      if (i >= urls.length) return reject(new Error('satellite texture failed: ' + urls[0]));
      new THREE.TextureLoader().load(urls[i], (t) => {
        t.colorSpace = THREE.SRGBColorSpace; resolve(t);
      }, undefined, () => tryNext(i + 1));
    };
    tryNext(0);
  });
}
async function loadGLB(urls) {
  const loader = new GLTFLoader();
  for (const u of urls) { try { return await loader.loadAsync(u); } catch { /* next candidate */ } }
  throw new Error('mesh.glb failed to load: ' + urls[0]);
}

/* ---------- DEM ---------- */
function computeDemMin() {
  demMin = Infinity;
  for (const row of dem.values) for (const v of row) demMin = Math.min(demMin, v);
  return demMin;
}
function demAt(x, z) {
  if (!dem) return 0;
  const i = Math.max(0, Math.min(dem.nx - 1, Math.round((x - dem.x0) / dem.dx)));
  const j = Math.max(0, Math.min(dem.nz - 1, Math.round((z - dem.z0) / dem.dz)));
  return dem.values[j][i];
}
function gnssAt(x, z) {
  if (!gnssGrid) return 0;
  const i = Math.max(0, Math.min(gnssGrid.nx - 1, Math.round((x - gnssGrid.x0) / gnssGrid.dx)));
  const j = Math.max(0, Math.min(gnssGrid.nz - 1, Math.round((z - gnssGrid.z0) / gnssGrid.dz)));
  return gnssGrid.values[j][i];
}

/* Build the DEM from the loaded mesh heightfield (min height per cell ~ ground).
   This is the real mesh's own surface data — the same source the 3D mesh renders from. */
function deriveDemFromMesh(obj) {
  obj.updateMatrixWorld(true);
  const pos = new THREE.Box3();
  obj.traverse(o => { if (o.geometry) { o.geometry.computeBoundingBox(); if (o.geometry.boundingBox) pos.union(o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld)); } });
  if (!Number.isFinite(pos.min.x) || !pos.min) return null;
  const NX = 41, NZ = 41;
  const bx0 = pos.min.x, bz0 = pos.min.z, bx1 = pos.max.x, bz1 = pos.max.z;
  const dx = (bx1 - bx0) / (NX - 1) || 5, dz = (bz1 - bz0) / (NZ - 1) || 5;
  const vals = Array.from({ length: NZ }, () => new Array(NX).fill(Infinity));
  const v = new THREE.Vector3();
  obj.traverse(o => {
    if (!o.isMesh) return;
    const p = o.geometry.attributes.position;
    for (let k = 0; k < p.count; k++) {
      v.fromBufferAttribute(p, k).applyMatrix4(o.matrixWorld);
      const i = Math.max(0, Math.min(NX - 1, Math.round((v.x - bx0) / dx)));
      const j = Math.max(0, Math.min(NZ - 1, Math.round((v.z - bz0) / dz)));
      if (v.y < vals[j][i]) vals[j][i] = v.y;
    }
  });
  let gmin = Infinity;
  for (const row of vals) for (const c of row) gmin = Math.min(gmin, c);
  for (let j = 0; j < NZ; j++) for (let i = 0; i < NX; i++) if (!Number.isFinite(vals[j][i])) vals[j][i] = gmin;
  return { nx: NX, nz: NZ, x0: bx0, z0: bz0, dx, dz, values: vals, bbox: { x0: bx0, z0: bz0, x1: bx1, z1: bz1 } };
}

/* Shadow proxy: local relief (highest neighbour minus cell) over the DEM,
   an advisory screening is needed because true shadow vectors live in the
   offline evaluator; the strip stays muted amber/rust. */
function buildShadowPlane(demGrid) {
  const W = demGrid.dx * (demGrid.nx - 1), D = demGrid.dz * (demGrid.nz - 1);
  const geo = new THREE.PlaneGeometry(W, D, demGrid.nx - 1, demGrid.nz - 1);
  geo.rotateX(-Math.PI / 2);
  const gp = geo.attributes.position;
  const gc = [];
  const cLow = new THREE.Color(PAL.shadowLow), cMid = new THREE.Color(PAL.shadowMid), cHigh = new THREE.Color(PAL.shadowHigh);
  let reliefMax = 1e-9;
  const relief = [];
  for (let j = 0; j < demGrid.nz; j++) { const row = []; for (let i = 0; i < demGrid.nx; i++) {
    let mx = demGrid.values[j][i];
    for (const dj of [-1, 0, 1]) for (const di of [-1, 0, 1]) {
      const ii = Math.max(0, Math.min(demGrid.nx - 1, i + di)), jj = Math.max(0, Math.min(demGrid.nz - 1, j + dj));
      mx = Math.max(mx, demGrid.values[jj][ii]);
    }
    const r = mx - demGrid.values[j][i]; row.push(r); reliefMax = Math.max(reliefMax, r);
  } relief.push(row); }
  for (let k = 0; k < gp.count; k++) {
    const x = gp.getX(k), z = gp.getZ(k);
    const i = Math.max(0, Math.min(demGrid.nx - 1, Math.round((x + W / 2 - demGrid.x0) / demGrid.dx)));
    const j = Math.max(0, Math.min(demGrid.nz - 1, Math.round((z + D / 2 - demGrid.z0) / demGrid.dz)));
    gp.setY(k, demGrid.values[j][i] + 0.35);
    const t = Math.max(0, Math.min(1, relief[j][i] / reliefMax * 1.6));
    const c = t < 0.5 ? cLow.clone().lerp(cMid, t * 2) : cMid.clone().lerp(cHigh, (t - 0.5) * 2);
    gc.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(gc), 3));
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false }));
  mesh.position.set(demGrid.x0 + W / 2, 0, demGrid.z0 + D / 2);
  return mesh;
}

/* ---------- flood: water plane at demMin + level + amber blocked cells ---------- */
function buildFloodLayers() {
  for (const o of [...floodGroup.children]) floodGroup.remove(o);
  floodPlane = null; blockMesh = null;
  if (!dem) return;
  const W = dem.dx * (dem.nx - 1), D = dem.dz * (dem.nz - 1);
  floodPlane = new THREE.Mesh(new THREE.PlaneGeometry(W + 4, D + 4),
    new THREE.MeshBasicMaterial({ color: PAL.flood, transparent: true, opacity: 0.5, depthWrite: false }));
  floodPlane.rotation.x = -Math.PI / 2;
  floodPlane.position.set(dem.x0 + W / 2, demMin + floodLevel, dem.z0 + D / 2);
  floodGroup.add(floodPlane);
  const cellGeo = new THREE.PlaneGeometry(dem.dx * 0.96, dem.dz * 0.96);
  cellGeo.rotateX(-Math.PI / 2);
  blockMesh = new THREE.InstancedMesh(cellGeo,
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9, depthWrite: false }),
    dem.nx * dem.nz);
  blockMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  floodGroup.add(blockMesh);
  applyFlood();
}

const _m4 = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3();
function applyFlood() {
  if (!dem || !blockMesh) return;
  let flooded = 0, blocked = 0, idx = 0;
  const col = new THREE.Color();
  for (let j = 0; j < dem.nz; j++) {
    for (let i = 0; i < dem.nx; i++) {
      const x = dem.x0 + i * dem.dx, z = dem.z0 + j * dem.dz;
      const depth = floodLevel - (dem.values[j][i] - demMin);
      if (depth >= WHEELCHAIR_BLOCK_M) {
        _p.set(x, dem.values[j][i] + 0.5, z); _q.identity(); _s.set(1, 1, 1);
        blockMesh.setMatrixAt(idx, _m4.compose(_p, _q, _s));
        blockMesh.setColorAt(idx, col.set(PAL.block)); blocked++;
      } else if (depth > 0) {
        _p.set(x, dem.values[j][i] + 0.5, z); _q.identity(); _s.set(1, 1, 1);
        blockMesh.setMatrixAt(idx, _m4.compose(_p, _q, _s));
        blockMesh.setColorAt(idx, col.set(PAL.flood)); flooded++;
      } else {
        _p.set(x, -100, z); _q.identity(); _s.set(0.0001, 0.0001, 0.0001);
        blockMesh.setMatrixAt(idx, _m4.compose(_p, _q, _s));
        blockMesh.setColorAt(idx, col.set('#000000'));
      }
      idx++;
    }
  }
  blockMesh.instanceMatrix.needsUpdate = true;
  if (blockMesh.instanceColor) blockMesh.instanceColor.needsUpdate = true;
  if (floodPlane) floodPlane.position.y = demMin + floodLevel;
  const ground = capture.groundPath || [];
  let pathBlocked = 0;
  for (const [x, z] of ground) if (floodLevel - (demAt(x, z) - demMin) >= WHEELCHAIR_BLOCK_M) pathBlocked++;
  $id('floodStats').textContent =
    `Flooded cells ${flooded} · blocked (≥0.10 m) ${blocked} · ground route in blocked zone ${pathBlocked}/${ground.length} pts`;
  $id('floodValue').textContent = floodLevel.toFixed(2) + ' m';
  $id('hudFlood').textContent = floodLevel.toFixed(2) + ' m';
}

/* ---------- LOS + swath (live flight overlay) ----------
   LOS drone → goal pad is screened client-side against the building boxes
   from capture.json (slab segment-vs-AABB). If the bundle ever carries
   per-frame droneFrames the recorded flags win instead. Swath radius is
   altitude above ground × tan(30°). Coverage and LOS-clear % accumulate
   over the path vertices traversed so far. */
function pathCacheFor(pts) {
  const cum = [0];
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i][0] - pts[i - 1][0], dy = pts[i][1] - pts[i - 1][1], dz = pts[i][2] - pts[i - 1][2];
    cum.push(cum[i - 1] + Math.hypot(dx, dy, dz));
  }
  return { pts, cum, total: cum[cum.length - 1] || 1 };
}
function posAt(t) {
  const { pts, cum, total } = pathCache;
  const d = Math.max(0, Math.min(total, t * total));
  let i = 1;
  while (i < cum.length - 1 && cum[i] < d) i++;
  const segLen = Math.max(1e-9, cum[i] - cum[i - 1]);
  const s = (d - cum[i - 1]) / segLen;
  const a = pts[i - 1], b = pts[i];
  return new THREE.Vector3(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s);
}
function segHitsBox(a, b, minX, maxX, minY, maxY, minZ, maxZ) {
  let t0 = 0, t1 = 1;
  const axes = [
    [a.x, b.x - a.x, minX, maxX], [a.y, b.y - a.y, minY, maxY], [a.z, b.z - a.z, minZ, maxZ],
  ];
  for (const [av, dv, lo, hi] of axes) {
    if (Math.abs(dv) < 1e-9) { if (av < lo || av > hi) return false; }
    else {
      let ta = (lo - av) / dv, tb = (hi - av) / dv;
      if (ta > tb) { const tmp = ta; ta = tb; tb = tmp; }
      t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
      if (t0 > t1) return false;
    }
  }
  return true;
}
function framesClear(t) {
  /* Recorded per-frame flags win when the bundle carries them. */
  let frames = capture && (capture.droneFrames || capture.frames);
  if (frames && !Array.isArray(frames)) frames = frames[currentGen] || frames.final;
  if (!Array.isArray(frames) || frames.length < 2) return null;
  const f = frames[Math.max(0, Math.min(frames.length - 1, Math.round(t * (frames.length - 1))))];
  if (!f || typeof f !== 'object') return null;
  const clear = f.losClear ?? f.clear ?? f.los ?? f.visible;
  const blockedBy = f.blockedBy ?? f.blocked_by ?? f.blocker ?? -1;
  if (clear == null && blockedBy == null) return null;
  return { clear: clear !== false && blockedBy < 0, blockedBy: Number.isFinite(blockedBy) ? blockedBy : -1 };
}
function checkLOS(pos, t) {
  const recorded = framesClear(t);
  if (recorded) return recorded;
  for (let i = 0; i < buildings.length; i++) {
    const b = buildings[i];
    if (segHitsBox(pos, padPos, b.x - b.w / 2, b.x + b.w / 2, 0, b.h, b.z - b.d / 2, b.z + b.d / 2)) {
      return { clear: false, blockedBy: i };
    }
  }
  return { clear: true, blockedBy: -1 };
}
function swathRadius(pos) {
  const alt = Math.max(0.5, pos.y - demAt(pos.x, pos.z));
  return alt * TAN30;
}
function buildBuildingBoxes() {
  for (const o of [...buildingGroup.children]) buildingGroup.remove(o);
  buildingMeshes = [];
  const edgeMat = new THREE.LineBasicMaterial({ color: PAL.buildingEdge, transparent: true, opacity: 0.75 });
  const fillMat = new THREE.MeshBasicMaterial({ color: PAL.buildingEdge, transparent: true, opacity: 0.10, depthWrite: false });
  buildings.forEach((b, i) => {
    const geo = new THREE.BoxGeometry(Math.max(0.5, b.w), Math.max(0.5, b.h), Math.max(0.5, b.d));
    const mesh = new THREE.Mesh(geo, fillMat.clone());
    mesh.position.set(b.x, b.h / 2, b.z);
    mesh.userData.building = { index: i, x: b.x, z: b.z, w: b.w, d: b.d, h: b.h };
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat.clone());
    edges.position.copy(mesh.position);
    buildingGroup.add(mesh); buildingGroup.add(edges);
    buildingMeshes.push(mesh);
  });
}
function buildLosHelpers() {
  for (const o of [...losGroup.children]) losGroup.remove(o);
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  losLine = new THREE.Line(g, new THREE.LineBasicMaterial({ color: PAL.losClear, transparent: true, opacity: 0.9 }));
  losGroup.add(losLine);
  coneMesh = new THREE.Mesh(new THREE.ConeGeometry(1, 1, 28, 1, true),
    new THREE.MeshBasicMaterial({ color: PAL.swath, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
  losGroup.add(coneMesh);
  const ringGeo = new THREE.RingGeometry(0.9, 1, 48); ringGeo.rotateX(-Math.PI / 2);
  swathRing = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: PAL.swath, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide }));
  losGroup.add(swathRing);
  const discGeo = new THREE.CircleGeometry(0.95, 40); discGeo.rotateX(-Math.PI / 2);
  swathDisc = new THREE.Mesh(discGeo, new THREE.MeshBasicMaterial({ color: PAL.swath, transparent: true, opacity: 0.20, depthWrite: false }));
  losGroup.add(swathDisc);
}
function buildGoalPad(tx, ty, tz) {
  for (const o of [...padGroup.children]) padGroup.remove(o);
  const gy = ty;
  padPos.set(tx, gy, tz);
  const ringGeo = new THREE.RingGeometry(3, 4, 40); ringGeo.rotateX(-Math.PI / 2);
  const ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: PAL.padRing, transparent: true, opacity: 0.95, depthWrite: false, side: THREE.DoubleSide }));
  ring.position.set(tx, gy + 0.1, tz);
  const discGeo = new THREE.CircleGeometry(3, 32); discGeo.rotateX(-Math.PI / 2);
  const disc = new THREE.Mesh(discGeo, new THREE.MeshBasicMaterial({ color: PAL.pad, transparent: true, opacity: 0.35, depthWrite: false }));
  disc.position.set(tx, gy, tz);
  padGroup.add(ring); padGroup.add(disc);
}

/* ---------- routes ---------- */
function buildRoutes() {
  for (const o of [...droneGroup.children]) droneGroup.remove(o);
  droneLines = {};
  if (droneMarker) { scene.remove(droneMarker); droneMarker = null; }
  const paths = capture.dronePaths || {};
  const cols = { Gen0: PAL.drone0, Gen25: PAL.drone25, final: PAL.droneFinal };
  for (const gen of ['Gen0', 'Gen25', 'final']) {
    const pts = paths[gen] || [];
    if (!pts.length) continue;
    const v = pts.map(([x, y, z]) => new THREE.Vector3(x, y, z));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(v),
      new THREE.LineBasicMaterial({ color: cols[gen], transparent: true, opacity: gen === currentGen ? 1 : 0.25 }));
    droneGroup.add(line);
    droneLines[gen] = line;
  }
  const active = droneLines[currentGen] || droneLines.final || Object.values(droneLines)[0];
  if (active) {
    droneMarker = new THREE.Mesh(new THREE.OctahedronGeometry(2.2),
      new THREE.MeshBasicMaterial({ color: PAL.marker }));
    droneGroup.add(droneMarker);
  }
  if (groundLine) droneGroup.remove(groundLine);
  groundLine = null;
  const g = capture.groundPath || [];
  if (g.length) {
    const pts = g.map(([x, z]) => new THREE.Vector3(x, (dem ? demAt(x, z) : 0) + 0.6, z));
    groundLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: PAL.ground, linewidth: 2 }));
    droneGroup.add(groundLine);
  }
  applyGen(); applyToggles();
}

function applyGen() {
  for (const gen of ['Gen0', 'Gen25', 'final']) {
    const line = droneLines[gen];
    if (line) line.material.opacity = gen === currentGen ? 1 : 0.25;
    const btn = document.querySelector(`[data-gen="${gen}"]`);
    if (btn) btn.classList.toggle('active', gen === currentGen);
  }
  $id('hudGen').textContent = currentGen === 'Gen0' ? 'Gen 0' : currentGen === 'Gen25' ? 'Gen 25' : 'Final';
  const pts = (capture.dronePaths && capture.dronePaths[currentGen]) || [];
  if (pts.length) {
    pathCache = pathCacheFor(pts);
    buildGoalPad(pts[pts.length - 1][0], pts[pts.length - 1][1], pts[pts.length - 1][2]);
  }
  renderMetrics();
  renderScienceSections();
  updateFlightState(0);
}

function applyToggles() {
  if (satPlane) satPlane.visible = $id('toggleSatellite').checked;
  if (floodPlane) floodPlane.visible = $id('toggleFlood').checked;
  if (blockMesh) blockMesh.visible = $id('toggleFlood').checked;
  if (shadowMesh) shadowMesh.visible = $id('toggleGnss').checked;
  meshGroup.visible = $id('toggleMesh').checked;
  for (const g of Object.values(droneLines)) g.visible = $id('toggleDrone').checked;
  if (droneMarker) droneMarker.visible = $id('toggleDrone').checked;
  if (groundLine) groundLine.visible = $id('toggleGround').checked;
  photoGroup.visible = $id('togglePhotos').checked;
  const bEl = $id('toggleBuildings'); if (bEl) buildingGroup.visible = bEl.checked;
  const lEl = $id('toggleLos');
  losGroup.visible = !!(lEl && lEl.checked && $id('toggleDrone').checked);
}

/* ---------- live flight state: drone + LOS + swath + cumulative % ---------- */
function updateFlightState(time) {
  if (!capture || !pathCache) return;
  const pts = pathCache.pts;
  const pos = posAt(flightT);
  if (droneMarker) droneMarker.position.copy(pos);
  const groundY = dem ? demAt(pos.x, pos.z) : 0;
  const alt = Math.max(0.5, pos.y - groundY);
  const radius = alt * TAN30;
  const los = checkLOS(pos, flightT);
  losState = { clear: los.clear, blockedBy: los.blockedBy, alt, radius, losPct: 100, coverage: 0 };
  if (losLine && losGroup.visible) {
    const p = losLine.geometry.attributes.position;
    p.setXYZ(0, pos.x, pos.y, pos.z); p.setXYZ(1, padPos.x, padPos.y, padPos.z);
    p.needsUpdate = true;
    losLine.material.color.set(los.clear ? PAL.losClear : PAL.losBlocked);
    losLine.material.opacity = los.clear ? 0.9 : 0.65 + 0.35 * Math.sin(time * 6);
  }
  if (coneMesh && losGroup.visible) {
    coneMesh.position.set(pos.x, (pos.y + groundY) / 2, pos.z);
    coneMesh.scale.set(radius, Math.max(0.5, pos.y - groundY), radius);
  }
  for (const o of [swathRing, swathDisc]) {
    if (o && losGroup.visible) {
      o.position.set(pos.x, groundY + 0.2, pos.z);
      o.scale.set(radius, 1, radius);
    }
  }
  /* Cumulative coverage + LOS-clear over traversed vertices. */
  if (dem) {
    const upto = Math.max(1, Math.ceil(flightT * (pts.length - 1)));
    const covered = new Set();
    let clearCount = 0;
    for (let k = 0; k <= Math.min(upto, pts.length - 1); k++) {
      const v = new THREE.Vector3(pts[k][0], pts[k][1], pts[k][2]);
      const r = swathRadius(v);
      for (let j = 0; j < dem.nz; j++) {
        for (let i = 0; i < dem.nx; i++) {
          const cx = dem.x0 + i * dem.dx, cz = dem.z0 + j * dem.dz;
          if (Math.hypot(cx - v.x, cz - v.z) <= r) covered.add(j * dem.nx + i);
        }
      }
      if (checkLOS(v, k / Math.max(1, pts.length - 1)).clear) clearCount++;
    }
    const n = Math.min(upto, pts.length - 1) + 1;
    losState.losPct = n ? (clearCount / n) * 100 : 100;
    losState.coverage = (covered.size / (dem.nx * dem.nz)) * 100;
  }
  renderLinkRows();
  $id('hudLos').textContent = 'LOS ' + (losState.clear ? 'clear' : 'BLOCKED');
  $id('progressLabel').textContent = Math.round(flightT * 100) + '%';
  const slider = $id('progressSlider');
  if (slider && document.activeElement !== slider) slider.value = String(Math.round(flightT * 1000));
}
function renderLinkRows() {
  const el = $id('linkRows');
  if (!el) return;
  const st = losState.clear ? 'Clear' : (losState.blockedBy >= 0 ? `Blocked by building #${losState.blockedBy}` : 'Blocked');
  const rows = [
    ['LOS status', st],
    ['LOS clear (cumulative)', fmt(losState.losPct, 1) + ' %'],
    ['Coverage (cumulative)', fmt(losState.coverage, 1) + ' %'],
    ['Drone altitude AGL', fmt(losState.alt, 1) + ' m'],
    ['Swath radius (tan30)', fmt(losState.radius, 1) + ' m'],
    ['Goal pad (scene x, y, z)', `${fmt(padPos.x, 0)}, ${fmt(padPos.y, 0)}, ${fmt(padPos.z, 0)}`],
  ];
  el.innerHTML = '';
  for (const [k, v] of rows) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    if (k === 'LOS status') dd.className = losState.clear ? 'link-clear' : 'link-blocked';
    el.append(dt, dd);
  }
}

/* ---------- metrics (numbers only where capture.json carries them) ---------- */
function renderMetrics() {
  const m = (capture && capture.metrics) || {};
  const body = $id('metricsBody');
  body.innerHTML = '';
  for (const gen of ['Gen0', 'Gen25', 'final']) {
    const v = m[gen] || { score: NaN, time: NaN, collisions: NaN };
    const tr = document.createElement('tr');
    if (gen === currentGen) tr.className = 'active';
    const sc = Number.isFinite(v.score) ? v.score.toFixed(1) : '—';
    const tm = Number.isFinite(v.time) ? v.time.toFixed(1) + ' s' : '—';
    const cl = Number.isFinite(v.collisions) ? String(v.collisions) : '—';
    tr.innerHTML = `<td>${gen === 'final' ? 'Final' : gen}</td><td>${sc}</td><td>${tm}</td><td>${cl}</td>`;
    body.appendChild(tr);
  }
  const cur = m[currentGen] || {};
  $id('metricsDetail').textContent =
    `Active ${currentGen === 'final' ? 'Final' : currentGen} · CD scalar (maximise) = ${Number.isFinite(cur.score) ? cur.score.toFixed(2) : '—'} · time ${Number.isFinite(cur.time) ? cur.time.toFixed(1) + ' s' : '—'} · collisions ${Number.isFinite(cur.collisions) ? cur.collisions : '—'} · drone altitude 0–60 m envelope.`;
}
function fillRows(el, rows) {
  el.innerHTML = '';
  for (const [k, v] of rows) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    el.append(dt, dd);
  }
}
/* ---------- science sections: Environment / Flight / Ground / Link ---------- */
function renderScienceSections() {
  if (!capture) return;
  /* Environment: capture-level facts that the bundle actually carries. */
  const st = (capture.statsRaw && typeof capture.statsRaw === 'object') ? capture.statsRaw : {};
  const envExtra = (capture.environment && typeof capture.environment === 'object') ? capture.environment : {};
  const bbox = capture.bbox;
  const env = [
    ['Capture', capture.name || '—'],
    ['BBox (lat/lon)', bbox ? `${bbox.minLat.toFixed(4)}–${bbox.maxLat.toFixed(4)} N, ${bbox.minLon.toFixed(4)}–${bbox.maxLon.toFixed(4)} E` : '—'],
    ['Buildings', st.buildings ?? (buildings.length || '—')],
    ['Landing pads', st.landing_pads ?? '—'],
    ['Ground nodes / edges', (st.ground_nodes != null && st.ground_edges != null) ? `${st.ground_nodes} / ${st.ground_edges}` : '—'],
    ['Street View photos', st.streetview_photos ?? (Array.isArray(capture.photos) ? capture.photos.length : '—')],
    ['Imagery dates', st['imagery dates'] ?? '—'],
    ['Mesh vertices', st.mesh_vertices ?? '—'],
    ['bbox (m)', st['bbox (m)'] ?? '—'],
  ];
  for (const k of Object.keys(envExtra)) env.push([k, String(envExtra[k])]);
  fillRows($id('envRows'), env);
  /* Flight: extended metrics for the active generation. */
  const m = capture.metrics[currentGen] || capture.metrics.final || {};
  $id('flightGenLabel').textContent = `(${currentGen === 'final' ? 'Final' : currentGen})`;
  fillRows($id('flightRows'), [
    ['Score (CD scalar)', fmt(m.score, 1)],
    ['Evaluator score', fmt(m.evaluator_score, 1)],
    ['Path length', fmt(m.length_m, 1) + ' m'],
    ['Time', fmt(m.time, 1) + ' s'],
    ['Energy', fmt(m.energy, 1)],
    ['Shadow exposure', fmt(m.shadow_exposure, 1)],
    ['Wind exposure', fmt(m.wind_exposure, 1)],
    ['Min clearance (SDF)', fmt(m.min_sdf_m, 2) + ' m'],
    ['Collisions', Number.isFinite(m.collisions) ? String(m.collisions) : '—'],
    ['Avg altitude', fmt(m.avg_altitude_m, 1) + ' m'],
    ['Altitude min–max', Number.isFinite(m.min_altitude_m) && Number.isFinite(m.max_altitude_m) ? `${fmt(m.min_altitude_m, 0)}–${fmt(m.max_altitude_m, 0)} m` : '—'],
    ['Altitude range', fmt(m.altitude_range_m, 1) + ' m'],
    ['Avg speed', fmt(m.avg_speed_mps, 1) + ' m/s'],
    ['Mean wind factor', fmt(m.mean_wind_factor, 2)],
    ['Max wind factor', fmt(m.max_wind_factor, 2)],
    ['AGL min–max', Number.isFinite(m.min_agl_m) && Number.isFinite(m.max_agl_m) ? `${fmt(m.min_agl_m, 1)}–${fmt(m.max_agl_m, 1)} m` : '—'],
    ['Mean AGL', fmt(m.mean_agl_m, 1) + ' m'],
  ]);
  /* Score-components breakdown: raw component × evolved weight. */
  const w = (capture.evolution && capture.evolution.bestWeights) || {};
  const comps = [
    ['Time', m.time, w.w_time, 's'], ['Energy', m.energy, w.w_energy, ''],
    ['Shadow', m.shadow_exposure, w.w_shadow, ''], ['Wind', m.wind_exposure, w.w_wind, ''],
    ['Detour (length)', m.length_m, w.w_detour, 'm'],
  ];
  fillRows($id('scoreRows'), comps.map(([name, raw, wt, unit]) =>
    Number.isFinite(wt) && Number.isFinite(raw)
      ? [name, `${fmt(raw, 1)}${unit} × ${fmt(wt, 2)} = ${fmt(raw * wt, 1)}`]
      : [name, fmt(raw, 1) + unit]));
  /* Ground: cached wheelchair route (slice coords 0..200 shown as scene). */
  const g = (capture.groundRaw && typeof capture.groundRaw === 'object') ? capture.groundRaw : {};
  const toScene = (p) => (Array.isArray(p) ? `${fmt(p[0] - 100, 1)}, ${fmt(p[1] - 100, 1)}` : '—');
  const grows = [
    ['Route success', g.success == null ? '—' : (g.success ? 'Yes' : 'No')],
    ['Length', fmt(g.length_m, 1) + ' m'],
    ['Time', fmt(g.time_s, 1) + ' s'],
    ['Max flood depth', fmt(g.max_flood_m, 2) + ' m'],
    ['Start (scene x, z)', toScene(g.start)],
    ['Goal (scene x, z)', toScene(g.goal)],
  ];
  if (g.notes) grows.push(['Notes', String(g.notes)]);
  fillRows($id('groundRows'), grows);
}

/* ---------- stats panel ---------- */
function renderStats() {
  const el = $id('captureStats');
  el.innerHTML = '';
  const dl = document.createElement('dl');
  dl.className = 'stats';
  const rows = [];
  if (capture.name) rows.push(['Capture', capture.name]);
  const bbox = capture.bbox;
  if (bbox) rows.push(['BBox', `${bbox.minLat.toFixed(4)}–${bbox.maxLat.toFixed(4)} N, ${bbox.minLon.toFixed(4)}–${bbox.maxLon.toFixed(4)} E`]);
  if (capture.stats && typeof capture.stats === 'object') {
    for (const k of Object.keys(capture.stats)) rows.push([k, String(capture.stats[k])]);
  }
  for (const [k, val] of rows) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = val;
    dl.append(dt, dd);
  }
  if (!rows.length) {
    const p = document.createElement('p');
    p.className = 'micro'; p.textContent = 'No stats in capture.json.';
    el.appendChild(p);
  } else el.appendChild(dl);
}

/* ---------- satellite plane + mesh + photos ---------- */
function latLonToScene(lat, lon, bbox) {
  const x = ((lon - bbox.minLon) / (bbox.maxLon - bbox.minLon)) * 200 - 100;
  const z = 100 - ((lat - bbox.minLat) / (bbox.maxLat - bbox.minLat)) * 200;
  return [x, z];
}
function sceneToGeo(x, z) {
  const b = capture.bbox;
  if (!b) return null;
  const lon = b.minLon + ((x + 100) / 200) * (b.maxLon - b.minLon);
  const lat = b.minLat + ((100 - z) / 200) * (b.maxLat - b.minLat);
  return { lat, lon };
}

async function buildCapture(c) {
  capture = c;
  capture.bbox = normBbox(c);
  capture.dronePaths = normDronePaths(c);
  capture.groundPath = normGround(pick(c, ['groundPath', 'ground_path', 'wheelchairPath', 'ground']) ?? []);
  capture.metrics = normMetrics(c);
  capture.metricsRaw = (c.metrics && typeof c.metrics === 'object') ? c.metrics : {};
  capture.statsRaw = (c.stats && typeof c.stats === 'object') ? c.stats : null;
  capture.evolution = (c.evolution && typeof c.evolution === 'object') ? c.evolution : null;
  capture.environment = (c.environment && typeof c.environment === 'object') ? c.environment : null;
  capture.groundRaw = (c.ground && typeof c.ground === 'object') ? c.ground : null;
  capture.droneFrames = pick(c, ['droneFrames', 'frames']) ?? null;
  buildings = Array.isArray(c.buildings) ? c.buildings : [];
  dem = asGrid(pick(c, ['dem', 'elevation', 'terrain', 'demGrid']) ?? null);
  gnssGrid = asGrid(pick(c, ['gnss', 'shadow', 'gnssShadow', 'gnss_grid']) ?? null);

  $id('captureTitle').textContent = c.label || c.name || 'Toa Payoh';
  $id('sourceNote').textContent = `Source: ./capture/capture.json · ${c.name || ''}`.trim();
  $id('captureNote').textContent =
    `Real Aleph capture loaded. ${Array.isArray(c.photos) ? c.photos.length : 0} Street View photos. ` +
    'Google-derived layers (satellite, mesh, Street View) are research-only.';
  renderStats();

  const bbox = c.bbox;
  let planeW = 200, planeH = 200;
  if (bbox) {
    const midLat = (bbox.minLat + bbox.maxLat) / 2 * Math.PI / 180;
    const wM = (bbox.maxLon - bbox.minLon) * 111320 * Math.cos(midLat);
    const hM = (bbox.maxLat - bbox.minLat) * 110540;
    if (wM >= hM) planeH = Math.max(1, 200 * hM / wM); else planeW = Math.max(1, 200 * wM / hM);
  }

  /* satellite ground plane (research-only Google layer) */
  satPlane = new THREE.Mesh(new THREE.PlaneGeometry(planeW, planeH),
    new THREE.MeshBasicMaterial({ color: PAL.gridA }));
  satPlane.rotation.x = -Math.PI / 2;
  satPlane.position.y = -0.5;
  scene.add(satPlane);
  if (c.satellite) {
    loadTexture(assetCandidates(c.satellite))
      .then(tex => { satPlane.material = new THREE.MeshBasicMaterial({ map: tex }); })
      .catch(() => { $id('captureNote').textContent += ' satellite.png failed to load — mesh and photos remain.'; });
  }

  /* 3D mesh (research-only Google layer) */
  if (c.mesh) {
    try {
      const gltf = await loadGLB(assetCandidates(c.mesh));
      const obj = gltf.scene;
      /* scale mesh so its bbox spans the capture bbox (metres -> scene units) */
      const box = new THREE.Box3().setFromObject(obj);
      const size = new THREE.Vector3(); box.getSize(size);
      const bboxW = (bbox ? (bbox.maxLon - bbox.minLon) * 111320 * Math.cos((bbox.minLat + bbox.maxLat) / 2 * Math.PI / 180) : size.x);
      const s = bboxW / Math.max(1e-6, size.x);
      obj.scale.setScalar(s);
      meshGroup.add(obj);
      if (!dem) dem = deriveDemFromMesh(obj);
    } catch (err) {
      $id('captureNote').textContent += ' mesh.glb failed to load — satellite and photos remain.';
    }
  }
  if (dem) {
    computeDemMin();
    shadowMesh && scene.remove(shadowMesh);
    shadowMesh = buildShadowPlane(dem);
    scene.add(shadowMesh);
    buildFloodLayers();
  } else {
    $id('floodStats').textContent = 'No DEM grid in capture.json and mesh unavailable — flood screening unavailable.';
  }

  buildRoutes();
  buildBuildingBoxes();
  buildLosHelpers();
  buildPhotos();
  prefillLocationBbox();
  renderScienceSections();
  updateFlightState(0);

  $id('filmstrip').hidden = false;
  applyToggles();
  applyFlood();
}

function buildPhotos() {
  for (const o of [...photoGroup.children]) photoGroup.remove(o);
  photoMarkers = [];
  const photos = Array.isArray(capture.photos) ? capture.photos : [];
  const bbox = capture.bbox;
  const scroll = $id('filmScroll');
  scroll.innerHTML = '';
  const markerGeo = new THREE.SphereGeometry(1.3, 10, 8);
  const markerMat = new THREE.MeshBasicMaterial({ color: PAL.photo });
  for (const photo of photos) {
    photo._cands = assetCandidates(photo.file);
    if (bbox && Number.isFinite(photo.lat) && Number.isFinite(photo.lon)) {
      const [x, z] = latLonToScene(photo.lat, photo.lon, bbox);
      const mk = new THREE.Mesh(markerGeo, markerMat);
      mk.position.set(x, 2.5, z);
      mk.userData.photo = photo;
      photoGroup.add(mk);
      photoMarkers.push(mk);
    }
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.alt = photo.path ? `Street View photo — ${photo.path}` : 'Street View photo';
    img.title = `${photo.lat.toFixed(6)}, ${photo.lon.toFixed(6)} · ${photo.date || 'date unknown'}`;
    let ci = 0;
    img.onerror = () => { ci += 1; if (ci < photo._cands.length) img.src = photo._cands[ci]; };
    img.src = photo._cands[0] || '';
    img.addEventListener('click', () => openPhoto(photo));
    scroll.appendChild(img);
  }
  if (!photos.length) scroll.innerHTML = '<span class="micro">No photos in capture.json.</span>';
}

function openPhoto(photo) {
  const cands = photo._cands || assetCandidates(photo.file);
  const img = $id('photoOverlayImg');
  let ci = 0;
  img.onerror = () => { ci += 1; if (ci < cands.length) img.src = cands[ci]; };
  img.src = cands[0] || '';
  $id('photoOverlayMeta').textContent =
    `lat ${Number.isFinite(photo.lat) ? photo.lat.toFixed(6) : '—'} · lon ${Number.isFinite(photo.lon) ? photo.lon.toFixed(6) : '—'} · date ${photo.date || 'unknown'}` +
    (photo.heading != null ? ` · heading ${Number(photo.heading).toFixed(0)}°` : '') +
    (photo.path ? ` · ${photo.path}` : '');
  $id('photoOverlay').hidden = false;
}
function closePhoto() { $id('photoOverlay').hidden = true; }

/* ---------- click to understand: detail card ---------- */
function showDetail(title, rows, imgSrc) {
  $id('detailTitle').textContent = title;
  const img = $id('detailImg');
  if (imgSrc) {
    img.hidden = false;
    let ci = 0;
    img.onerror = () => { ci += 1; if (ci < imgSrc.length) img.src = imgSrc[ci]; else img.hidden = true; };
    img.src = imgSrc[0] || '';
  } else { img.hidden = true; img.src = ''; }
  fillRows($id('detailRows'), rows);
  $id('detailCard').hidden = false;
}
function closeDetail() { $id('detailCard').hidden = true; }
function showPointDetail(point) {
  const geo = sceneToGeo(point.x, point.z);
  const elev = dem ? demAt(point.x, point.z) : 0;
  const depth = dem ? floodLevel - (elev - demMin) : 0;
  const floodTxt = dem ? (depth >= WHEELCHAIR_BLOCK_M ? `${fmt(depth, 2)} m — BLOCKED (≥ 0.10 m)` : `${fmt(Math.max(0, depth), 2)} m`) : '—';
  showDetail('Ground point', [
    ['Scene x, z', `${fmt(point.x, 1)}, ${fmt(point.z, 1)}`],
    ['Lat / lon', geo ? `${geo.lat.toFixed(6)}, ${geo.lon.toFixed(6)}` : '—'],
    ['DEM elevation', dem ? fmt(elev, 2) + ' m' : '—'],
    ['Flood depth (slider)', floodTxt],
    ['GNSS shadow', gnssGrid ? fmt(gnssAt(point.x, point.z), 3) : '—'],
  ]);
}
function showBuildingDetail(b) {
  const area = b.w * b.d, vol = b.w * b.d * b.h;
  showDetail(`Building #${b.index}`, [
    ['Height', fmt(b.h, 1) + ' m'],
    ['Footprint (w × d)', `${fmt(b.w, 0)} × ${fmt(b.d, 0)} m`],
    ['Est. area', fmt(area, 0) + ' m²'],
    ['Est. volume', fmt(vol, 0) + ' m³'],
    ['Centre (scene x, z)', `${fmt(b.x, 1)}, ${fmt(b.z, 1)}`],
    ['Base DEM', dem ? fmt(demAt(b.x, b.z), 1) + ' m' : '—'],
  ]);
}
function showPhotoDetail(photo) {
  const cands = photo._cands || assetCandidates(photo.file);
  showDetail('Photo', [
    ['Lat', Number.isFinite(photo.lat) ? photo.lat.toFixed(6) : '—'],
    ['Lon', Number.isFinite(photo.lon) ? photo.lon.toFixed(6) : '—'],
    ['Date', photo.date || 'unknown'],
    ['Heading', photo.heading != null ? fmt(Number(photo.heading), 0) + '°' : '—'],
    ['Path name', photo.path || '—'],
    ['File', photo.file ? String(photo.file).split('/').pop() : '—'],
  ], cands);
}

/* Unified click raycast: photo marker → building box → mesh → ground plane. */
(() => {
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let downX = 0, downY = 0;
  renderer.domElement.addEventListener('pointerdown', (e) => { downX = e.clientX; downY = e.clientY; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!capture) return;
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) return;
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const photoHit = ray.intersectObjects(photoMarkers, false)[0];
    if (photoHit) { showPhotoDetail(photoHit.object.userData.photo); return; }
    const bHit = ray.intersectObjects(buildingMeshes, false)[0];
    if (bHit) { showBuildingDetail(bHit.object.userData.building); return; }
    const meshHit = ray.intersectObjects(meshGroup.children, true).find(h => h.object.isMesh);
    if (meshHit) { showPointDetail(meshHit.point); return; }
    if (satPlane && satPlane.visible) {
      const gHit = ray.intersectObject(satPlane, false)[0];
      if (gHit) { showPointDetail(gHit.point); return; }
    }
    closeDetail();
  });
})();

/* ---------- location picker (Aleph-style, via /api) ---------- */
function prefillLocationBbox() {
  const b = capture && capture.bbox;
  if (!b) return;
  $id('bMinLat').value = b.minLat.toFixed(4);
  $id('bMaxLat').value = b.maxLat.toFixed(4);
  $id('bMinLon').value = b.minLon.toFixed(4);
  $id('bMaxLon').value = b.maxLon.toFixed(4);
}
function apiUnavailable() {
  $id('jobStatus').textContent = 'API unavailable (static hosting) — serve with python serve.py --port 8100, then reload.';
  $id('apiNote').textContent = 'Location API unavailable under static hosting. Serve it from the project root with python serve.py --port 8100, then reload this page for Search and Capture & rebuild.';
}
function setLocationBboxAround(lat, lon, name) {
  const d = 0.0009; /* ~200 m box, same size as the capture */
  $id('bMinLat').value = (lat - d).toFixed(4);
  $id('bMaxLat').value = (lat + d).toFixed(4);
  $id('bMinLon').value = (lon - d).toFixed(4);
  $id('bMaxLon').value = (lon + d).toFixed(4);
  $id('jobStatus').textContent = `BBox placed around ${name} — use Capture & rebuild to apply.`;
}
async function searchPlaces() {
  const q = $id('placeInput').value.trim();
  const list = $id('placeResults');
  list.innerHTML = '';
  if (!q) { $id('jobStatus').textContent = 'Enter a place name first.'; return; }
  $id('jobStatus').textContent = 'Searching…';
  let data;
  try {
    const res = await fetch('/api/resolve?q=' + encodeURIComponent(q));
    if (!res.ok) throw new Error('resolve ' + res.status);
    data = await res.json();
  } catch { apiUnavailable(); return; }
  const results = Array.isArray(data) ? data : (data.results || data.candidates || data.places || []);
  if (!results.length) { $id('jobStatus').textContent = 'No results for "' + q + '".'; return; }
  $id('jobStatus').textContent = `${results.length} result(s) — click one to place the bbox.`;
  for (const r of results.slice(0, 8)) {
    const name = r.name || r.display_name || r.displayName || r.title || 'Place';
    const lat = Number(r.lat ?? r.latitude), lon = Number(r.lon ?? r.lng ?? r.longitude);
    const li = document.createElement('li');
    li.innerHTML = `${name}<span class="mono">${Number.isFinite(lat) && Number.isFinite(lon) ? `${lat.toFixed(5)}, ${lon.toFixed(5)}` : 'no coordinates'}</span>`;
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      li.addEventListener('click', () => setLocationBboxAround(lat, lon, name));
    }
    list.appendChild(li);
  }
}
function readBboxInputs() {
  const b = {
    minLat: Number($id('bMinLat').value), maxLat: Number($id('bMaxLat').value),
    minLon: Number($id('bMinLon').value), maxLon: Number($id('bMaxLon').value),
  };
  if (![b.minLat, b.maxLat, b.minLon, b.maxLon].every(Number.isFinite)) return null;
  if (!(b.minLat < b.maxLat && b.minLon < b.maxLon)) return null;
  return b;
}
async function fetchJob(id) {
  const res = await fetch('/api/job?id=' + encodeURIComponent(id));
  if (res.ok) return res.json();
  if (res.status === 404) {
    const alt = await fetch('/api/job/' + encodeURIComponent(id));
    if (alt.ok) return alt.json();
  }
  throw new Error('job ' + res.status);
}
async function pollJob(id, tries = 0) {
  let data;
  try {
    data = await fetchJob(id);
  } catch { apiUnavailable(); return; }
  const status = String(data.status || data.state || 'running').toLowerCase();
  if (['done', 'complete', 'completed', 'success', 'finished', 'ok'].includes(status)) {
    $id('jobStatus').textContent = 'Capture done — rebuilding viewer…';
    setTimeout(() => location.reload(), 400);
    return;
  }
  if (['error', 'failed', 'cancelled', 'canceled'].includes(status)) {
    $id('jobStatus').textContent = `Capture failed (${status})${data.message ? ': ' + data.message : ''}`;
    return;
  }
  if (tries > 119) { $id('jobStatus').textContent = `Job ${id} still ${status} — check back after reload.`; return; }
  $id('jobStatus').textContent = `Job ${id}: ${status}… (poll ${tries + 1})`;
  setTimeout(() => pollJob(id, tries + 1), 1500);
}
async function startCapture() {
  const bbox = readBboxInputs();
  if (!bbox) { $id('jobStatus').textContent = 'BBox must be numeric with min < max.'; return; }
  $id('jobStatus').textContent = 'Starting capture…';
  let data;
  try {
    const res = await fetch('/api/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      /* Server contract: bbox as [S, W, N, E] array. */
      body: JSON.stringify({ bbox: [bbox.minLat, bbox.minLon, bbox.maxLat, bbox.maxLon] }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      $id('jobStatus').textContent = 'Capture rejected: ' + (err?.error?.message || ('HTTP ' + res.status));
      return;
    }
    data = await res.json();
  } catch { apiUnavailable(); return; }
  const jobId = data.jobId ?? data.job_id ?? data.id ?? (data.job && data.job.id);
  if (jobId == null) { $id('jobStatus').textContent = 'Capture started (no job id returned).'; return; }
  pollJob(jobId);
}

/* ---------- controls ---------- */
document.querySelectorAll('[data-gen]').forEach(btn => {
  btn.addEventListener('click', () => { currentGen = btn.dataset.gen; flightT = 0; applyGen(); });
});
document.querySelectorAll('.toggle input').forEach(el => {
  el.addEventListener('change', applyToggles);
});
$id('floodSlider').addEventListener('input', (e) => {
  floodLevel = Number(e.target.value);
  applyFlood();
});
$id('photoOverlayClose').addEventListener('click', closePhoto);
$id('photoOverlay').addEventListener('click', (e) => { if (e.target === $id('photoOverlay')) closePhoto(); });
$id('detailClose').addEventListener('click', closeDetail);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { closePhoto(); closeDetail(); }
});
$id('playBtn').addEventListener('click', () => {
  playing = !playing;
  $id('playBtn').textContent = playing ? 'Pause' : 'Play';
});
$id('speedSel').addEventListener('change', (e) => { speed = Number(e.target.value) || 1; });
$id('progressSlider').addEventListener('input', (e) => {
  flightT = Math.max(0, Math.min(1, Number(e.target.value) / 1000));
  playing = false;
  $id('playBtn').textContent = 'Play';
  updateFlightState(0);
});
$id('placeSearchBtn').addEventListener('click', searchPlaces);
$id('placeInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') searchPlaces(); });
$id('captureBtn').addEventListener('click', startCapture);

/* ---------- resize + render loop ---------- */
function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', resize);
const clock = new THREE.Clock();
const FLIGHT_RATE = 0.045; /* path fraction per second at 1× (~22 s lap) */
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  const time = clock.elapsedTime;
  if (capture && pathCache) {
    if (playing) flightT = (flightT + dt * FLIGHT_RATE * speed) % 1;
    updateFlightState(time);
  }
  controls.update();
  renderer.render(scene, camera);
}
resize(); animate();

/* ---------- boot: load capture, show empty state when absent ---------- */
(async () => {
  let c = null;
  try {
    const res = await fetch('./capture/capture.json', { cache: 'no-store' });
    if (res.ok) c = await res.json();
  } catch { c = null; }
  if (!c || !c.bbox) {
    $id('emptyState').hidden = false;
    $id('captureNote').textContent = 'No capture loaded — run python scripts/run_real.py from the project root, then reload.';
    $id('sourceNote').textContent = 'Source: ./capture/capture.json (absent)';
    return;
  }
  $id('emptyState').hidden = true;
  await buildCapture(c);
})();
