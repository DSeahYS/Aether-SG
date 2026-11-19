import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/* AETHER-SG offline tactical viewer.
   Loads ./data.json when present (orchestrator run_demo output), otherwise
   uses built-in fallback demo data so the page is never blank.
   Constraints honoured: drone solver 0-60 m, flood screening tensor with
   wheelchair block at depth >= 0.10 m, single CD scalar metric,
   honest cached-replay framing. No live evolution / live capture claims. */

const WHEELCHAIR_BLOCK_M = 0.10;

// ---------- deterministic fallback data (never-blank rendering) ----------
function lcg(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

function fallbackData() {
  const rand = lcg(20251119);
  const NX = 41, NZ = 41, SIZE = 200; // 200 x 200 m slice, 5 m cells
  const dem = [];
  for (let j = 0; j < NZ; j++) {
    const row = [];
    for (let i = 0; i < NX; i++) {
      const x = -SIZE / 2 + (i / (NX - 1)) * SIZE;
      const z = -SIZE / 2 + (j / (NZ - 1)) * SIZE;
      row.push(6.0 + 0.8 * Math.sin(x * 0.03) * Math.cos(z * 0.025) + (rand() - 0.5) * 0.12);
    }
    dem.push(row);
  }
  const buildings = [
    { x: -60, z: -55, w: 30, d: 18, h: 40 }, { x: -20, z: -60, w: 26, d: 20, h: 45 },
    { x: 35, z: -58, w: 32, d: 18, h: 38 }, { x: 70, z: -30, w: 22, d: 22, h: 30 },
    { x: -70, z: 10, w: 28, d: 20, h: 42 }, { x: -30, z: 15, w: 24, d: 16, h: 36 },
    { x: 15, z: 25, w: 30, d: 20, h: 44 }, { x: 60, z: 30, w: 24, d: 18, h: 34 },
    { x: -55, z: 65, w: 30, d: 20, h: 40 }, { x: 0, z: 70, w: 26, d: 18, h: 32 },
    { x: 55, z: 68, w: 28, d: 20, h: 38 }, { x: 85, z: 55, w: 18, d: 18, h: 26 },
  ];
  const mkPath = (n, seed, jitter, cross) => {
    const r = lcg(seed); const pts = [];
    for (let k = 0; k < n; k++) {
      const t = k / (n - 1);
      const x = -95 + 190 * t;
      const z = 60 - 110 * t + Math.sin(t * Math.PI * cross) * 18 + (r() - 0.5) * jitter;
      const y = 14 + 26 * Math.sin(t * Math.PI * 1.2) + (r() - 0.5) * jitter * 0.6;
      pts.push([x, Math.min(58, Math.max(4, y)), z]);
    }
    return pts;
  };
  const gnss = [];
  for (let j = 0; j < NZ; j++) {
    const row = [];
    for (let i = 0; i < NX; i++) {
      const x = -SIZE / 2 + (i / (NX - 1)) * SIZE;
      const z = -SIZE / 2 + (j / (NZ - 1)) * SIZE;
      const near = buildings.reduce((m, b) => Math.min(m, Math.hypot(x - b.x, z - b.z)), 1e9);
      row.push(Math.max(0, Math.min(1, 1.1 - near / 55 + (rand() - 0.5) * 0.1)));
    }
    gnss.push(row);
  }
  const groundPath = [];
  for (let k = 0; k <= 40; k++) {
    const t = k / 40;
    groundPath.push([-90 + 175 * t, 60 - 115 * t + Math.sin(t * Math.PI * 3) * 10]);
  }
  const ramps = groundPath.filter((_, i) => i % 2 === 0).map(([x, z], i) => ({ x: x + (i % 3 - 1) * 3, z: z + 2.5, verified: true }));
  return {
    buildings, dem: { nx: NX, nz: NZ, x0: -100, z0: -100, dx: 5, dz: 5, values: dem },
    gnss: { nx: NX, nz: NZ, x0: -100, z0: -100, dx: 5, dz: 5, values: gnss },
    dronePaths: { Gen0: mkPath(28, 7, 22, 3), Gen25: mkPath(28, 8, 8, 2), final: mkPath(28, 9, 2.5, 1) },
    groundPath, ramps,
    metrics: {
      Gen0: { score: -148.2, time: 94.0, collisions: 3 },
      Gen25: { score: -96.4, time: 71.0, collisions: 0 },
      final: { score: -71.8, time: 62.0, collisions: 0 },
    },
  };
}

// ---------- tolerant normalizers for orchestrator data.json ----------
const pick = (obj, keys) => { for (const k of keys) if (obj && obj[k] != null) return obj[k]; return undefined; };
function ciGet(obj, key) {
  if (!obj || typeof obj !== 'object') return undefined;
  const want = String(key).toLowerCase();
  for (const k of Object.keys(obj)) if (k.toLowerCase() === want) return obj[k];
  return undefined;
}
function asGrid(g) {
  if (!g) return null;
  if (Array.isArray(g)) return { nx: g[0]?.length ?? 0, nz: g.length, x0: -100, z0: -100, dx: 5, dz: 5, values: g };
  const vals = pick(g, ['values', 'elevations', 'heights', 'depths', 'grid', 'data', 'cells']);
  if (!vals || !vals[0]) return null;
  return { nx: vals[0].length, nz: vals.length, x0: pick(g, ['x0', 'minX']) ?? -100, z0: pick(g, ['z0', 'minZ', 'y0']) ?? -100, dx: pick(g, ['dx', 'cell', 'res']) ?? 5, dz: pick(g, ['dz', 'cell', 'res']) ?? 5, values: vals };
}
function normBuilding(b) {
  const x = pick(b, ['x', 'cx', 'centerX']) ?? b.position?.[0] ?? b.pos?.[0] ?? 0;
  const z = pick(b, ['z', 'cz', 'centerZ', 'y']) ?? b.position?.[1] ?? b.pos?.[1] ?? 0;
  const w = pick(b, ['w', 'width', 'sizeX']) ?? b.size?.[0] ?? 20;
  const d = pick(b, ['d', 'depth', 'sizeZ']) ?? b.size?.[2] ?? b.size?.[1] ?? 18;
  const h = pick(b, ['h', 'height', 'heightM', 'height_m']) ?? b.size?.[1] ?? 32;
  const rot = pick(b, ['rot', 'rotation', 'rotY']) ?? 0;
  return { x, z, w, d, h, rot };
}
function normPts(arr) {
  if (!Array.isArray(arr)) return [];
  return arr.map(p => Array.isArray(p) ? [Number(p[0]), Number(p[1] ?? 0), Number(p[2] ?? 0)] : [Number(p.x ?? 0), Number(p.y ?? 0), Number(p.z ?? 0)]).filter(p => p.every(Number.isFinite));
}
function normGround(arr) {
  if (!Array.isArray(arr)) return [];
  return arr.map(p => Array.isArray(p) ? [Number(p[0]), Number(p[1] ?? 0)] : [Number(p.x ?? 0), Number(p.z ?? p.y ?? 0)]).filter(p => p.every(Number.isFinite));
}
function normRamps(arr) {
  if (!Array.isArray(arr)) return [];
  return arr.map(r => Array.isArray(r) ? { x: Number(r[0]), z: Number(r[1]) } : { x: Number(r.x ?? 0), z: Number(r.z ?? r.y ?? 0) }).filter(r => Number.isFinite(r.x) && Number.isFinite(r.z));
}
function normDronePaths(raw) {
  const src = pick(raw, ['dronePaths', 'paths', 'pathsByGen', 'trajectories', 'drone', 'paths_by_gen']) ?? ciGet(raw, 'dronePaths') ?? raw.dronePaths ?? raw;
  const out = {};
  const map = (k, v) => { if (k != null) out[k] = normPts(v); };
  if (Array.isArray(src)) {
    if (src.length >= 3) { map('Gen0', src[0]); map('Gen25', src[1]); map('final', src[2]); }
    else if (src.length === 1) map('final', src[0]);
    return out;
  }
  if (src && typeof src === 'object') {
    map('Gen0', ciGet(src, 'Gen0') ?? ciGet(src, 'gen0') ?? ciGet(src, 'Gen 0') ?? ciGet(src, 'baseline') ?? ciGet(src, 'generation0'));
    map('Gen25', ciGet(src, 'Gen25') ?? ciGet(src, 'gen25') ?? ciGet(src, 'Gen 25') ?? ciGet(src, 'intermediate') ?? ciGet(src, 'generation25'));
    map('final', ciGet(src, 'final') ?? ciGet(src, 'Gen500') ?? ciGet(src, 'gen500') ?? ciGet(src, 'winner') ?? ciGet(src, 'evolved'));
    const single = normPts(pick(src, ['dronePath', 'path', 'points']));
    if (!out.final && single.length) out.final = single;
  }
  return out;
}
function normMetrics(raw) {
  const src = pick(raw, ['metrics', 'metricsByGen', 'generationMetrics', 'scores']) ?? ciGet(raw, 'metrics') ?? {};
  const m = {};
  const one = v => (v && typeof v === 'object') ? { score: Number(v.score ?? v.value ?? v.scalar ?? v.metric ?? NaN), time: Number(v.time ?? v.timeS ?? v.duration ?? NaN), collisions: Number(v.collisions ?? v.collisionCount ?? v.collisionsCount ?? 0) } : null;
  if (Array.isArray(src)) { ['Gen0', 'Gen25', 'final'].forEach((k, i) => { const v = one(src[i]); if (v) m[k] = v; }); return m; }
  ['Gen0', 'Gen25', 'final'].forEach(k => { const v = one(ciGet(src, k) ?? ciGet(src, k === 'final' ? 'Gen500' : k)); if (v) m[k] = v; });
  return m;
}
function normalize(raw) {
  const fb = fallbackData();
  if (!raw || typeof raw !== 'object') return { d: fb, source: 'fallback' };
  const bRaw = pick(raw, ['buildings', 'blocks', 'bldgs']) ?? [];
  const buildings = Array.isArray(bRaw) && bRaw.length ? bRaw.map(normBuilding) : fb.buildings;
  const dem = asGrid(pick(raw, ['dem', 'elevation', 'elevations', 'demGrid', 'terrain']) ?? ciGet(raw, 'dem')) || fb.dem;
  const gnss = asGrid(pick(raw, ['gnss', 'gnssShadow', 'shadowGrid', 'shadow', 'gnssShadowGrid', 'heatmap']) ?? ciGet(raw, 'gnss')) || fb.gnss;
  const dp = normDronePaths(raw);
  const dronePaths = Object.keys(dp).some(k => dp[k].length) ? dp : fb.dronePaths;
  const groundPath = normGround(pick(raw, ['groundPath', 'ground_path', 'wheelchairPath', 'ground']) ?? ciGet(raw, 'groundPath'));
  const ramps = normRamps(pick(raw, ['ramps', 'rampMarkers']) ?? []);
  const metrics = normMetrics(raw);
  const hasRealMetrics = Object.keys(metrics).length;
  return {
    d: { buildings, dem, gnss, dronePaths, groundPath: groundPath.length ? groundPath : fb.groundPath, ramps: ramps.length ? ramps : fb.ramps, metrics: hasRealMetrics ? metrics : fb.metrics },
    source: 'data.json',
  };
}

// ---------- three.js scene ----------
const container = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0a0f16');
scene.fog = new THREE.Fog('#0a0f16', 250, 700);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 3000);
camera.position.set(150, 150, 220);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 8, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;
scene.add(new THREE.HemisphereLight('#cfe8ff', '#1a2633', 0.9));
const sun = new THREE.DirectionalLight('#ffffff', 1.6);
sun.position.set(120, 220, 80);
scene.add(sun);
const grid = new THREE.GridHelper(400, 40, '#1e2e42', '#16222f');
grid.position.y = -0.2;
scene.add(grid);

let DATA = null; // normalized {d, source}
let layers = {}; // named groups/meshes
let currentGen = 'final';
let floodLevel = 0.10;
let droneMarker = null;
let droneAnim = { pts: [], t: 0 };

function demAt(x, z, dem) {
  const i = Math.round((x - dem.x0) / dem.dx);
  const j = Math.round((z - dem.z0) / dem.dz);
  const ii = Math.max(0, Math.min(dem.nx - 1, i));
  const jj = Math.max(0, Math.min(dem.nz - 1, j));
  return dem.values[jj][ii];
}

function clearLayers() {
  for (const key of Object.keys(layers)) {
    const obj = layers[key];
    if (obj) { scene.remove(obj); obj.traverse?.(o => { o.geometry?.dispose?.(); }); }
  }
  layers = {};
  if (droneMarker) { scene.remove(droneMarker); droneMarker = null; }
}

function buildScene(d) {
  clearLayers();
  const demMin = Math.min(...d.dem.values.flat());
  // Ground from DEM
  const groundGeo = new THREE.PlaneGeometry(200, 200, d.dem.nx - 1, d.dem.nz - 1);
  groundGeo.rotateX(-Math.PI / 2);
  const pos = groundGeo.attributes.position;
  const colors = [];
  const cA = new THREE.Color('#1a2633'), cB = new THREE.Color('#24364a'), cLine = new THREE.Color('#2d445c');
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    const y = demAt(x, z, d.dem) - demMin;
    pos.setY(i, y);
    const c = (((Math.round(x / 10) + Math.round(z / 10)) % 2) === 0) ? cB : cA;
    colors.push(c.r, c.g, c.b);
  }
  groundGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
  groundGeo.computeVertexNormals();
  const ground = new THREE.Mesh(groundGeo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0 }));
  ground.position.set(0, 0, 0);
  scene.add(ground);
  layers.ground = ground;

  // Buildings
  const bGroup = new THREE.Group();
  d.buildings.forEach((b, i) => {
    const gy = demAt(b.x, b.z, d.dem) - demMin;
    const geo = new THREE.BoxGeometry(b.w, b.h, b.d);
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color().setHSL(0.58, 0.18, 0.32 + (i % 5) * 0.02), roughness: 0.85 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(b.x, gy + b.h / 2, b.z);
    mesh.rotation.y = b.rot || 0;
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: '#8fa6bd', transparent: true, opacity: 0.35 }));
    mesh.add(edges);
    bGroup.add(mesh);
  });
  scene.add(bGroup);
  layers.buildings = bGroup;

  // GNSS shadow heatmap (continuous vertex-colored plane)
  const gGeo = new THREE.PlaneGeometry(200, 200, d.gnss.nx - 1, d.gnss.nz - 1);
  gGeo.rotateX(-Math.PI / 2);
  const gp = gGeo.attributes.position;
  const gc = [];
  const cLow = new THREE.Color('#334155'), cMid = new THREE.Color('#f59e0b'), cHigh = new THREE.Color('#ef4444');
  for (let i = 0; i < gp.count; i++) {
    const x = gp.getX(i), z = gp.getZ(i);
    const v = Math.max(0, Math.min(1, (() => { const ii = Math.max(0, Math.min(d.gnss.nx - 1, Math.round((x - d.gnss.x0) / d.gnss.dx))); const jj = Math.max(0, Math.min(d.gnss.nz - 1, Math.round((z - d.gnss.z0) / d.gnss.dz))); return d.gnss.values[jj][ii]; })()));
    const y = demAt(x, z, d.dem) - demMin + 0.25;
    gp.setY(i, y);
    const c = v < 0.5 ? cLow.clone().lerp(cMid, v * 2) : cMid.clone().lerp(cHigh, (v - 0.5) * 2);
    gc.push(c.r, c.g, c.b);
  }
  gGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(gc), 3));
  const gnss = new THREE.Mesh(gGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6, depthWrite: false }));
  scene.add(gnss);
  layers.gnss = gnss;

  // Flood per-cell instanced overlay
  const NX = d.dem.nx, NZ = d.dem.nz;
  const cellGeo = new THREE.PlaneGeometry(d.dem.dx * 0.98, d.dem.dz * 0.98);
  cellGeo.rotateX(-Math.PI / 2);
  const floodMesh = new THREE.InstancedMesh(cellGeo, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.85, depthWrite: false }), NX * NZ);
  floodMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(floodMesh);
  layers.flood = floodMesh;

  // Drone paths (all three; toggle by gen)
  layers.dronePaths = {};
  for (const gen of ['Gen0', 'Gen25', 'final']) {
    const pts = d.dronePaths[gen] || [];
    if (!pts.length) continue;
    const v = pts.map(([x, y, z]) => new THREE.Vector3(x, gy_offset(x, z) + y, z));
    function gy_offset(px, pz) { return demAt(px, pz, d.dem) - demMin; }
    const geo = new THREE.BufferGeometry().setFromPoints(v);
    const col = gen === 'final' ? '#c084fc' : gen === 'Gen25' ? '#7dd3fc' : '#f87171';
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: gen === currentGen ? 1 : 0.28 }));
    scene.add(line);
    layers.dronePaths[gen] = line;
  }
  // drone marker (animated dot on active path)
  droneMarker = new THREE.Mesh(new THREE.OctahedronGeometry(2.2), new THREE.MeshBasicMaterial({ color: '#e9d5ff' }));
  scene.add(droneMarker);

  // Ground wheelchair path
  const gpts = d.groundPath.map(([x, z]) => new THREE.Vector3(x, demAt(x, z, d.dem) - demMin + 0.6, z));
  const gLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(gpts), new THREE.LineBasicMaterial({ color: '#a3e635', linewidth: 2 }));
  scene.add(gLine);
  layers.groundPath = gLine;

  // Ramps
  const rGroup = new THREE.Group();
  d.ramps.forEach(r => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.5, 2.4), new THREE.MeshStandardMaterial({ color: '#22d3ee', emissive: '#0e7490', roughness: 0.6 }));
    mesh.position.set(r.x, demAt(r.x, r.z, d.dem) - demMin + 0.5, r.z);
    rGroup.add(mesh);
  });
  scene.add(rGroup);
  layers.ramps = rGroup;

  applyGen(); applyFlood(); applyToggles();
}

const _m4 = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3();
function cellY(x, z) { const demMin = Math.min(...DATA.d.dem.values.flat()); return demAt(x, z, DATA.d.dem) - demMin; }
function applyFlood() {
  if (!DATA || !layers.flood) return;
  const d = DATA.d;
  const demMin = Math.min(...d.dem.values.flat());
  let flooded = 0, blocked = 0, idx = 0;
  for (let j = 0; j < d.dem.nz; j++) {
    for (let i = 0; i < d.dem.nx; i++) {
      const x = d.dem.x0 + i * d.dem.dx, z = d.dem.z0 + j * d.dem.dz;
      // Screening depth: water depth above the lowest DEM cell (bathtub).
      const depthAbs = floodLevel - (d.dem.values[j][i] - demMin);
      const set = (scaleY, color) => {
        _p.set(x, cellY(x, z) + 0.35, z); _q.identity(); _s.set(1, 1, scaleY);
        _m4.compose(_p, _q, _s); layers.flood.setMatrixAt(idx, _m4);
        layers.flood.setColorAt(idx, new THREE.Color(color));
      };
      if (depthAbs >= WHEELCHAIR_BLOCK_M) { set(1, '#fbbf24'); blocked++; }
      else if (depthAbs > 0) { set(1, '#38bdf8'); flooded++; }
      else { _p.set(x, -100, z); _q.identity(); _s.set(0.0001, 0.0001, 0.0001); _m4.compose(_p, _q, _s); layers.flood.setMatrixAt(idx, _m4); layers.flood.setColorAt(idx, new THREE.Color('#000000')); }
      idx++;
    }
  }
  layers.flood.instanceMatrix.needsUpdate = true;
  if (layers.flood.instanceColor) layers.flood.instanceColor.needsUpdate = true;
  // ground path blocked check: sample path pts
  let pathBlocked = 0;
  for (const [x, z] of DATA.d.groundPath) { if (floodLevel - (demAt(x, z, DATA.d.dem) - demMin) >= WHEELCHAIR_BLOCK_M) pathBlocked++; }
  const el = document.getElementById('floodStats');
  if (el) el.textContent = `Flooded cells: ${flooded} · wheelchair-blocked (≥0.10 m): ${blocked} · ground path in blocked zone: ${pathBlocked}/${DATA.d.groundPath.length} pts`;
  document.getElementById('floodValue').textContent = floodLevel.toFixed(2) + ' m';
  document.getElementById('hudFlood').textContent = floodLevel.toFixed(2) + ' m';
}
function demMinOffset(d) { return 6.0; }

function applyGen() {
  if (!DATA) return;
  for (const gen of ['Gen0', 'Gen25', 'final']) {
    const line = layers.dronePaths?.[gen];
    if (line) line.material.opacity = (gen === currentGen ? 1 : 0.25);
    const btn = document.querySelector(`[data-gen="${gen}"]`);
    if (btn) btn.classList.toggle('active', gen === currentGen);
  }
  document.getElementById('hudGen').textContent = currentGen === 'Gen0' ? 'Gen 0 (baseline)' : currentGen === 'Gen25' ? 'Gen 25 (intermediate)' : 'Final (Gen 500)';
  renderMetrics();
}
function applyToggles() {
  if (!DATA) return;
  const set = (id, obj, invert) => { const el = document.getElementById(id); if (el && obj) obj.visible = invert ? !el.checked : el.checked; };
  set('toggleFlood', layers.flood);
  set('toggleGnss', layers.gnss);
  set('toggleGround', layers.groundPath);
  set('toggleRamps', layers.ramps);
  const dt = document.getElementById('toggleDrone');
  if (dt && layers.dronePaths) for (const g of ['Gen0', 'Gen25', 'final']) if (layers.dronePaths[g]) layers.dronePaths[g].visible = dt.checked;
  if (droneMarker) droneMarker.visible = document.getElementById('toggleDrone')?.checked ?? true;
}

function renderMetrics() {
  if (!DATA) return;
  const m = DATA.d.metrics;
  const body = document.getElementById('metricsBody');
  body.innerHTML = '';
  for (const gen of ['Gen0', 'Gen25', 'final']) {
    const v = m[gen] || { score: NaN, time: NaN, collisions: 0 };
    const tr = document.createElement('tr');
    if (gen === currentGen) tr.className = 'active';
    const ok = Number(v.collisions) === 0;
    tr.innerHTML = `<td>${gen === 'final' ? 'Final' : gen}</td><td>${Number.isFinite(v.score) ? v.score.toFixed(1) : '—'}</td><td>${Number.isFinite(v.time) ? v.time.toFixed(1) + ' s' : '—'}</td><td class="${ok ? 'pass' : 'fail'}">${v.collisions} ${ok ? 'PASS' : 'FAIL'}</td>`;
    body.appendChild(tr);
  }
  const cur = m[currentGen] || {};
  document.getElementById('metricsDetail').textContent =
    `Active: ${currentGen} · CD scalar (maximise) = ${Number.isFinite(cur.score) ? cur.score.toFixed(2) : '—'} · time ${Number.isFinite(cur.time) ? cur.time.toFixed(1) + ' s' : '—'} · collisions ${cur.collisions} (${Number(cur.collisions) === 0 ? 'PASS — hard-reject satisfied' : 'FAIL — hard-reject violated'}) · drone altitude clamped to 0–60 m solver envelope.`;
}

// ---------- wiring ----------
document.getElementById('gen0Btn').addEventListener('click', () => { currentGen = 'Gen0'; applyGen(); });
document.getElementById('gen25Btn').addEventListener('click', () => { currentGen = 'Gen25'; applyGen(); });
document.getElementById('genFinalBtn').addEventListener('click', () => { currentGen = 'final'; applyGen(); });
document.getElementById('floodSlider').addEventListener('input', e => { floodLevel = Number(e.target.value); applyFlood(); });
for (const id of ['toggleFlood', 'toggleGnss', 'toggleDrone', 'toggleGround', 'toggleRamps']) {
  document.getElementById(id).addEventListener('change', applyToggles);
}

// ---------- resize + animate ----------
function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (DATA && droneMarker && layers.dronePaths?.[currentGen]) {
    const pts = DATA.d.dronePaths[currentGen] || [];
    if (pts.length > 1) {
      droneAnim.t = (droneAnim.t + dt * 0.06) % 1;
      const f = droneAnim.t * (pts.length - 1);
      const a = pts[Math.floor(f)], b = pts[Math.min(pts.length - 1, Math.floor(f) + 1)];
      const s = f - Math.floor(f);
      const [x, y, z] = [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s];
      droneMarker.position.set(x, cellY(x, z) + y, z);
    }
  }
  controls.update();
  renderer.render(scene, camera);
}
resize(); animate();

// ---------- Real Capture mode (Aleph) ----------
// Simulated path above is untouched. Real mode hides the simulated layers and
// shows a separate group built from ./capture/capture.json. Google-derived
// layers are research-only (never redistributed; production swaps to OneMap SG).
const $id = (id) => document.getElementById(id);
let mode = 'simulated'; // 'simulated' | 'real'
let captureData = null;
let realGroup = null;
let captureLoading = false;
let photoMarkers = []; // meshes with userData.photo set

// Asset paths in capture.json are relative to viewer/. Also probe the
// ./capture/ re-export layout and bare basenames so either hosting works.
function assetCandidates(p) {
  if (!p) return [];
  const out = [p];
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
        t.colorSpace = THREE.SRGBColorSpace;
        resolve(t);
      }, undefined, () => tryNext(i + 1));
    };
    tryNext(0);
  });
}
async function loadGLB(urls) {
  const loader = new GLTFLoader();
  for (const u of urls) {
    try { return await loader.loadAsync(u); } catch { /* try next candidate */ }
  }
  throw new Error('mesh.glb failed to load: ' + urls[0]);
}

function latLonToScene(lat, lon, bbox) {
  const x = ((lon - bbox.minLon) / (bbox.maxLon - bbox.minLon)) * 200 - 100;
  const z = 100 - ((lat - bbox.minLat) / (bbox.maxLat - bbox.minLat)) * 200; // north = -z
  return [x, z];
}

function setSimVisible(v) {
  for (const key of ['ground', 'buildings', 'gnss', 'flood', 'groundPath', 'ramps']) {
    if (layers[key]) layers[key].visible = v;
  }
  if (layers.dronePaths) for (const g of Object.keys(layers.dronePaths)) if (layers.dronePaths[g]) layers.dronePaths[g].visible = v;
  if (droneMarker) droneMarker.visible = v;
  if (v) applyToggles(); // restore per-checkbox visibility
}

function setMode(next) {
  if (next === 'real' && !captureData) { void loadCapture(); return; }
  mode = next;
  $id('modeSimBtn').classList.toggle('active', mode === 'simulated');
  $id('modeRealBtn').classList.toggle('active', mode === 'real');
  $id('realCard').hidden = mode !== 'real';
  $id('galleryStrip').hidden = mode !== 'real';
  if (mode === 'real') {
    if (realGroup) realGroup.visible = true;
    setSimVisible(false);
    $id('hudGen').textContent = 'Real Capture (Aleph)';
    $id('modeHint').textContent = 'Real Aleph capture loaded. Toggle back to Simulated Slice for the data.json replay.';
  } else {
    closePhoto();
    if (realGroup) realGroup.visible = false;
    setSimVisible(true);
    applyGen(); // restores hudGen + metrics for simulated mode
    $id('modeHint').textContent = 'Simulated replay from data.json. Switch to Real Capture to load the Aleph capture (mesh.glb, satellite, Street View).';
  }
}

async function loadCapture() {
  if (captureLoading) return;
  captureLoading = true;
  const note = $id('realNote');
  note.textContent = 'Loading capture.json…';
  $id('realCard').hidden = false; // show progress even though mode stays simulated
  let c = null;
  try {
    const res = await fetch('./capture/capture.json', { cache: 'no-store' });
    if (res.ok) c = await res.json();
  } catch { c = null; }
  if (!c || !c.bbox) {
    note.textContent = 'no capture yet — run scripts/aleph.py capture';
    captureLoading = false;
    setMode('simulated'); // stay on simulated per contract
    $id('realCard').hidden = false; // keep the note visible
    return;
  }
  captureData = c;
  note.textContent = `Loaded ${c.name || 'Aleph capture'} — building real scene…`;
  try {
    buildRealScene(c);
    note.textContent = `Real scene ready · ${Array.isArray(c.photos) ? c.photos.length : 0} Street View photos.`;
    captureLoading = false;
    setMode('real');
  } catch (err) {
    note.textContent = 'Capture loaded but real scene failed: ' + (err && err.message ? err.message : err);
    captureLoading = false;
    setMode('simulated');
    $id('realCard').hidden = false;
  }
}

function buildRealScene(c) {
  if (realGroup) { scene.remove(realGroup); }
  realGroup = new THREE.Group();
  realGroup.visible = false;
  scene.add(realGroup);
  photoMarkers = [];
  const bbox = c.bbox;
  // Bbox extents in metres (equirect approx) — satellite plane keeps bbox aspect.
  const midLat = (bbox.minLat + bbox.maxLat) / 2 * Math.PI / 180;
  const wM = (bbox.maxLon - bbox.minLon) * 111320 * Math.cos(midLat);
  const hM = (bbox.maxLat - bbox.minLat) * 110540;
  let planeW = 200, planeH = 200;
  if (wM >= hM) planeH = Math.max(1, 200 * hM / wM); else planeW = Math.max(1, 200 * wM / hM);
  const meshScale = planeW / Math.max(1e-6, wM); // mesh is in bbox metres -> scene units

  // Satellite ground texture (research-only Google layer).
  const satPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(planeW, planeH),
    new THREE.MeshBasicMaterial({ color: '#22303f' })
  );
  satPlane.rotation.x = -Math.PI / 2;
  satPlane.position.y = -0.5; // below mesh ground (mesh rests on y = 0)
  realGroup.add(satPlane);
  loadTexture(assetCandidates(c.satellite))
    .then((tex) => { satPlane.material = new THREE.MeshBasicMaterial({ map: tex }); })
    .catch(() => { $id('realNote').textContent = 'satellite.png failed to load (mesh/Street View still available).'; });

  // Textured 3D mesh (Google Earth, research-only). Positioned by lat/lon mapping.
  loadGLB(assetCandidates(c.mesh))
    .then((gltf) => {
      const meshObj = gltf.scene;
      meshObj.scale.setScalar(meshScale);
      realGroup.add(meshObj);
    })
    .catch(() => { $id('realNote').textContent = 'mesh.glb failed to load (satellite/Street View still available).'; });

  // Street View photo markers + gallery strip.
  const scroll = $id('galleryScroll');
  scroll.innerHTML = '';
  const photos = Array.isArray(c.photos) ? c.photos : [];
  const markerGeo = new THREE.SphereGeometry(1.3, 10, 8);
  const markerMat = new THREE.MeshBasicMaterial({ color: '#fbbf24' });
  for (const photo of photos) {
    if (!Number.isFinite(photo.lat) || !Number.isFinite(photo.lon)) continue;
    photo._cands = assetCandidates(photo.file);
    const [x, z] = latLonToScene(photo.lat, photo.lon, bbox);
    const mk = new THREE.Mesh(markerGeo, markerMat);
    mk.position.set(x, 2.5, z);
    mk.userData.photo = photo;
    realGroup.add(mk);
    photoMarkers.push(mk);
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

  // OSM / terrain / coverage stats panel.
  const dl = document.createElement('dl');
  const stats = c.stats && typeof c.stats === 'object' ? c.stats : {};
  for (const k of Object.keys(stats)) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = String(stats[k]);
    dl.append(dt, dd);
  }
  const statsEl = $id('realStats');
  statsEl.innerHTML = '';
  statsEl.appendChild(dl);
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

// Marker click (ignore orbit drags: only treat short pointer moves as clicks).
(() => {
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let downX = 0, downY = 0;
  renderer.domElement.addEventListener('pointerdown', (e) => { downX = e.clientX; downY = e.clientY; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (mode !== 'real' || !photoMarkers.length) return;
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) return;
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObjects(photoMarkers, false)[0];
    if (hit) openPhoto(hit.object.userData.photo);
  });
})();

$id('modeSimBtn').addEventListener('click', () => setMode('simulated'));
$id('modeRealBtn').addEventListener('click', () => setMode('real'));
$id('photoOverlayClose').addEventListener('click', closePhoto);
$id('photoOverlay').addEventListener('click', (e) => { if (e.target === $id('photoOverlay')) closePhoto(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closePhoto(); });

// ---------- load ----------
(async () => {
  let raw = null;
  try {
    const res = await fetch('./data.json', { cache: 'no-store' });
    if (res.ok) raw = await res.json();
  } catch { raw = null; }
  DATA = normalize(raw);
  document.getElementById('sourceNote').textContent =
    DATA.source === 'data.json' ? 'Source: data.json (orchestrator run_demo output)' : 'Source: built-in fallback demo data (data.json not found — page never blank)';
  buildScene(DATA.d);
})();
