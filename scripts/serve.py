"""Local AETHER-SG app server (stdlib only, 127.0.0.1).

Static file serving of the project root (viewer at ``/viewer/``) plus
a small JSON API:

  GET  /api/resolve?q=...        -> aleph_ingest.resolve results
  POST /api/capture {place?, bbox?:[S,W,N,E], sources?:[...]}
                                  -> create a capture job (queued)
  GET  /api/job?id=...           -> one job (status + log lines)
  GET  /api/jobs                 -> all jobs

Job pipeline (background thread): resolve place -> 200 m bbox if needed,
``aleph_ingest.capture_create`` (all five sources by default),
``capture_view.convert_capture`` to ``viewer/capture/``, then
``scripts/run_real.py --capture-dir <new>``.

Security: binds 127.0.0.1 only; bbox must be inside lat -90..90 /
lon -180..180 and area <= 4 km2.

Run:  .venv/Scripts/python.exe scripts/serve.py [--port 8100]
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aether import aleph_ingest  # noqa: E402
from aether import capture_view  # noqa: E402

HOST = "127.0.0.1"
MAX_AREA_KM2 = 4.0

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

_CT: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".geojson": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".glb": "model/gltf-binary",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".svg": "image/svg+xml",
    ".osm": "application/octet-stream",
}


def _now() -> float:
    return time.time()


def _log(job: dict, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with _JOBS_LOCK:
        job.setdefault("log", []).append(line)


def bbox_area_km2(bbox) -> float:
    """Approximate bbox area in km2 (equirectangular, mid-lat)."""
    s, w, n, e = (float(bbox[0]), float(bbox[1]),
                  float(bbox[2]), float(bbox[3]))
    h_m = (n - s) * 110540.0
    w_m = (e - w) * 111320.0 * math.cos(math.radians((s + n) / 2.0))
    return (h_m * w_m) / 1e6


def _validate_geo_bbox(bbox):
    """Strict API bbox check. Return (ok, error, area_km2).

    The HTTP API accepts only geographic [S, W, N, E] (lat -90..90,
    lon -180..180, area <= 4 km2). The public ``validate_bbox`` below
    is a frame-tolerant sanity helper; the API uses this function.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "bbox must be [S, W, N, E] (south, west, north, east)", None
    try:
        s, w, n, e = (float(bbox[0]), float(bbox[1]),
                      float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return False, f"bbox must contain four numbers, got {bbox!r}", None
    if not (-90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
        return False, f"latitude outside -90..90: S={s} N={n}", None
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        return False, f"longitude outside -180..180: W={w} E={e}", None
    if not (s < n and w < e):
        return False, f"empty bbox: S={s} W={w} N={n} E={e}", None
    area = bbox_area_km2([s, w, n, e])
    if area > MAX_AREA_KM2:
        return False, f"bbox area {area:.2f} km2 exceeds {MAX_AREA_KM2} km2", area
    return True, None, area


def validate_bbox(bbox) -> bool:
    """Frame-tolerant bbox sanity (backend / viewer / geographic).

    Accepts a list ``[a, b, c, d]`` or a four-key dict (values in key
    order) when it is a non-empty, non-oversized box in ANY of the
    three AETHER frames: backend slice 0..200 m, viewer -100..100 m,
    or geographic lat/lon with area <= 4 km2. The HTTP API itself
    enforces the strict geographic rule via ``_validate_geo_bbox``;
    this helper guards callers that accept any frame.
    """
    vals = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        vals = list(bbox)
    elif isinstance(bbox, dict) and len(bbox) == 4:
        vals = list(bbox.values())
    else:
        return False
    try:
        a, b, c, d = (float(vals[0]), float(vals[1]),
                      float(vals[2]), float(vals[3]))
    except (TypeError, ValueError):
        return False
    if not (a < c and b < d):
        return False
    try:
        if (-90.0 <= a <= 90.0 and -90.0 <= c <= 90.0
                and -180.0 <= b <= 180.0 and -180.0 <= d <= 180.0
                and bbox_area_km2([a, b, c, d]) <= MAX_AREA_KM2):
            return True
    except (TypeError, ValueError):
        pass
    span = (c - a) * (d - b)
    if (0.0 <= a <= 200.0 and 0.0 <= b <= 200.0
            and 0.0 <= c <= 200.0 and 0.0 <= d <= 200.0
            and span <= 4e6):
        return True
    if (-100.0 <= a <= 100.0 and -100.0 <= b <= 100.0
            and -100.0 <= c <= 100.0 and -100.0 <= d <= 100.0
            and span <= 4e6):
        return True
    return False


def bbox_from_latlon(lat: float, lon: float, size_m: float = 200.0):
    """200 m box centred on (lat, lon), as [S, W, N, E]."""
    dlat = (size_m / 2.0) / 110540.0
    dlon = (size_m / 2.0) / (111320.0 * math.cos(math.radians(lat)))
    return [float(lat) - dlat, float(lon) - dlon,
            float(lat) + dlat, float(lon) + dlon]


def _extract_folder(result: dict, since: float) -> Path:
    """Capture run folder from capture_create --json, else newest."""
    folder = (result or {}).get("folder")
    if folder:
        p = Path(str(folder))
        if not p.is_absolute():
            p = ROOT / p
        if p.is_dir():
            return p
        print(f"capture folder from result not found: {p}", flush=True)
    caps = ROOT / "captures"
    best, best_m = None, since - 1
    if caps.is_dir():
        for d in caps.iterdir():
            if d.is_dir():
                try:
                    m = d.stat().st_mtime
                except OSError:
                    continue
                if m > best_m:
                    best, best_m = d, m
    if best is not None:
        return best
    raise RuntimeError(f"capture_create returned no usable folder: {result!r}")


def _set_status(job: dict, status: str, **kw) -> None:
    with _JOBS_LOCK:
        job["status"] = status
        job.update(kw)


def run_job(job: dict, place, bbox, sources) -> None:
    """Queued capture job: ingest -> convert -> run_real (logs each step)."""
    started = _now()
    _set_status(job, "running", started_at=started)
    try:
        if place:
            _log(job, f"resolve place={place!r}")
            res = aleph_ingest.resolve(query=str(place))
            results = res.get("results") or []
            if not results:
                raise RuntimeError(f"no results for place {place!r}")
            top = results[0]
            lat, lon = float(top["lat"]), float(top["lon"])
            bbox = bbox_from_latlon(lat, lon, 200.0)
            _log(job, f"resolved {top.get('label', top.get('name'))} "
                      f"at ({lat:.6f}, {lon:.6f}) -> bbox "
                      f"{[round(v, 6) for v in bbox]}")
            _set_status(job, "running", bbox=bbox, place=str(place))
        ok, err, area = _validate_geo_bbox(bbox)
        if not ok:
            raise ValueError(f"invalid bbox {bbox!r}: {err}")
        _log(job, f"bbox {[round(v, 6) for v in bbox]} "
                  f"area {area:.3f} km2 sources={sources}")
        _log(job, "capture_create ...")
        created = aleph_ingest.capture_create(
            bbox=tuple(float(v) for v in bbox), sources=list(sources))
        folder = _extract_folder(created, started)
        _log(job, f"captured to {folder}")
        _set_status(job, "running", capture_dir=str(folder))
        out_dir = ROOT / "viewer" / "capture"
        _log(job, f"convert_capture -> {out_dir}")
        capture_view.convert_capture(str(folder), str(out_dir))
        _log(job, "convert_capture done")
        cmd = [sys.executable, str(ROOT / "scripts" / "run_real.py"),
               "--capture-dir", str(folder),
               "--out", str(out_dir / "capture.json")]
        _log(job, f"run_real {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=600)
        for line in (proc.stdout or "").strip().splitlines():
            _log(job, f"run_real| {line}")
        for line in (proc.stderr or "").strip().splitlines():
            _log(job, f"run_real stderr| {line}")
        if proc.returncode != 0:
            raise RuntimeError(f"run_real exited {proc.returncode}")
        _log(job, "run_real done")
        _set_status(job, "done", finished_at=_now(), error=None)
        _log(job, "job done")
    except Exception as exc:  # noqa: BLE001 - job error is reported, not raised
        _log(job, f"ERROR: {type(exc).__name__}: {exc}")
        _set_status(job, "error", finished_at=_now(), error=str(exc))


class Handler(BaseHTTPRequestHandler):
    server_version = "AETHER-SG/1.0"

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        ctype = _CT.get(suffix) or mimetypes.guess_type(str(path))[0]
        ctype = ctype or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, url_path: str) -> None:
        rel = unquote(url_path).lstrip("/")
        if rel in ("", "index.html"):
            rel = "viewer/index.html"
        elif rel in ("viewer", "viewer/"):
            rel = "viewer/index.html"
        target = (ROOT / rel).resolve()
        if not str(target).startswith(str(ROOT.resolve())):
            self._send_json({"status": "error",
                             "error": {"code": "forbidden",
                                       "message": "path outside project root"}},
                            status=403)
            return
        if target.is_dir():
            idx = target / "index.html"
            if idx.is_file():
                self._send_file(idx)
                return
            self._send_json({"status": "error",
                             "error": {"code": "not_found",
                                       "message": f"no file at {url_path!r}"}},
                            status=404)
            return
        if target.is_file():
            self._send_file(target)
            return
        self._send_json({"status": "error",
                         "error": {"code": "not_found",
                                   "message": f"no file at {url_path!r}"}},
                        status=404)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        if path == "/api/resolve":
            q = (qs.get("q") or [""])[0].strip()
            if not q:
                self._send_json({"status": "error",
                                 "error": {"code": "invalid_arguments",
                                           "message": "q is required"}},
                                status=400)
                return
            try:
                res = aleph_ingest.resolve(query=q)
            except aleph_ingest.AlephError as exc:
                self._send_json(exc.to_dict(), status=502)
                return
            except ValueError as exc:
                self._send_json({"status": "error",
                                 "error": {"code": "invalid_arguments",
                                           "message": str(exc)}})
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json({"status": "error",
                                 "error": {"code": "request_failed",
                                           "message": str(exc)}},
                                status=502)
                return
            self._send_json({"status": res.get("status", "complete"),
                             "query": q,
                             "results": res.get("results") or [],
                             "cache": res.get("cache")})
            return
        if path == "/api/job":
            jid = (qs.get("id") or [""])[0]
            with _JOBS_LOCK:
                job = _JOBS.get(jid)
                snapshot = dict(job) if job else None
            if not snapshot:
                self._send_json({"status": "error",
                                 "error": {"code": "not_found",
                                           "message": f"no job {jid!r}"}},
                                status=404)
                return
            self._send_json(snapshot)
            return
        if path == "/api/jobs":
            with _JOBS_LOCK:
                jobs = [dict(j) for j in _JOBS.values()]
            jobs.sort(key=lambda j: j.get("created_at", 0))
            self._send_json({"jobs": jobs})
            return
        if path.startswith("/api/"):
            self._send_json({"status": "error",
                             "error": {"code": "not_found",
                                       "message": f"no API at {path!r}"}},
                            status=404)
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path != "/api/capture":
            self._send_json({"status": "error",
                             "error": {"code": "not_found",
                                       "message": f"no API at {parsed.path!r}"}},
                            status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json({"status": "error",
                             "error": {"code": "invalid_arguments",
                                       "message": f"invalid JSON body: {exc}"}},
                            status=400)
            return
        if not isinstance(payload, dict):
            self._send_json({"status": "error",
                             "error": {"code": "invalid_arguments",
                                       "message": "body must be a JSON object"}},
                            status=400)
            return
        place = payload.get("place")
        bbox = payload.get("bbox")
        sources = payload.get("sources")
        if sources is None:
            sources = list(aleph_ingest.CAPTURE_SOURCES)
        elif not isinstance(sources, list) or not sources:
            self._send_json({"status": "error",
                             "error": {"code": "invalid_arguments",
                                       "message": "sources must be a non-empty list"}},
                            status=400)
            return
        else:
            bad = [s for s in sources if s not in aleph_ingest.CAPTURE_SOURCES]
            if bad:
                self._send_json({"status": "error",
                                 "error": {"code": "invalid_arguments",
                                           "message": f"invalid sources {bad!r}; "
                                           f"choose from {list(aleph_ingest.CAPTURE_SOURCES)}"}},
                                status=400)
                return
        if place is None and bbox is None:
            self._send_json({"status": "error",
                             "error": {"code": "invalid_arguments",
                                       "message": "provide place and/or bbox"}},
                            status=400)
            return
        if bbox is not None:
            ok, err, _area = _validate_geo_bbox(bbox)
            if not ok:
                self._send_json({"status": "error",
                                 "error": {"code": "invalid_arguments",
                                           "message": err,
                                           "bbox": bbox}},
                                status=400)
                return
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "status": "queued", "place": place,
               "bbox": bbox, "sources": list(sources),
               "created_at": _now(), "log": [],
               "error": None, "capture_dir": None}
        with _JOBS_LOCK:
            _JOBS[jid] = job
        th = threading.Thread(target=run_job,
                              args=(job, place, bbox, list(sources)),
                              daemon=True, name=f"capture-{jid}")
        th.start()
        self._send_json({"id": jid, "status": "queued", "place": place,
                         "bbox": bbox, "sources": list(sources)},
                        status=202)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AETHER-SG local server (127.0.0.1)")
    p.add_argument("--port", type=int, default=8100, help="port (default 8100)")
    p.add_argument("--host", default=HOST, help="bind host (must be 127.0.0.1)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.host != HOST:
        print(f"refusing to bind {args.host!r}: only {HOST!r} is allowed",
              file=sys.stderr)
        return 2
    if not (1 <= int(args.port) <= 65535):
        print(f"invalid port: {args.port!r}", file=sys.stderr)
        return 2
    httpd = ThreadingHTTPServer((HOST, int(args.port)), Handler)
    print(f"AETHER-SG serving {ROOT} at http://{HOST}:{int(args.port)}/ "
          f"(viewer: /viewer/)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

