"""Quarry Maker - a simple app in your web browser for the pipeline.

Start it with the "Start Quarry Maker" file (double-click), or:
    python app/quarry_app.py

It runs only on this computer (127.0.0.1). Nothing is published online.
The OpenTopography key you type is kept in memory while the app runs and is
never written to a file.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(APP_DIR))
import common  # noqa: E402
import config_io  # noqa: E402

CONFIG = ROOT / "config.yaml"
PROJECT = ROOT  # folder that holds config.yaml, data/, terrain/ ... (set in main)
TOKEN = secrets.token_urlsafe(24)
PIT_FILE = "data/reference/pit_outline.geojson"

STAGE_INFO = {
    "00": ("Check setup", "Checks that Python, Blender, BlenderGIS and your OpenTopography key are ready."),
    "01": ("Download elevation data", "Downloads the 30 m height data (DEM) of your area from OpenTopography."),
    "02": ("Download satellite image", "Downloads the satellite image of your area (Esri World Imagery)."),
    "03": ("Prepare and align", "Puts height data and image in one coordinate system, fills small gaps, "
                                "and fixes the origin point."),
    "04": ("Build the Blender terrain", "Builds the true-scale, textured terrain in Blender and exports "
                                        "files for Rhino."),
    "05": ("Make the Gaea package", "Makes the square heightmap and a guide for setting the right scale in Gaea."),
    "06": ("Check the results", "Measures the results (scale, position, heights) and renders preview images."),
    "07": ("Calculate the benches", "Carves the quarry benches (flat floors, tall walls) into the "
                                    "terrain, following the bench lines in the satellite photo."),
    "09": ("Compare the years", "Finds the quarry area on every different archived satellite photo "
                                "(Esri Wayback, 2014 onwards) and shows how the quarry grew."),
    "08": ("Draw the plan linework", "Draws bench lines, quarry outline, roads, farmland and block yards "
                                     "as a DXF file for Rhino / CAD."),
}

HINTS = [
    ("OpenTopography key is not set", "Enter your OpenTopography key in step 2 (Settings) and press 'Use this key'."),
    ("OPENTOPO_API_KEY", "Enter your OpenTopography key in step 2 (Settings) and press 'Use this key'."),
    ("area of interest is not set", "Draw your area on the map in step 1 and press Save."),
    ("Blender path is not set", "Set the Blender location in step 2 (Settings)."),
    ("Blender executable not found", "The Blender location in step 2 is wrong. Pick one of the detected "
                                     "locations or check where Blender is installed."),
    ("consistent bench pattern", "No part of the pit showed clear benches. Look at the 'bench evidence' "
                                 "image under step 7. If you know the bench height, enter it in step 2. "
                                 "If the quarry was dug mostly after 2015, the height data is too old for "
                                 "this step: untick 'Calculate the benches'; your other results are complete."),
    ("pit_mask not found", "The traced quarry outline file is missing (it stays in the old folder when you "
                           "download a new version). Copy data/reference/pit_outline.geojson from the old "
                           "folder, or trace the outline again in step 1."),
    ("No quarry pit was found", "Trace the pit outline on the map (step 1, 'Trace pit outline') and save."),
    ("API key looks invalid", "Your OpenTopography key was refused. Check it and enter it again in step 2."),
    ("too large", "The area is too large. Draw a smaller rectangle around the quarry."),
    ("needs", "The area needs too many satellite tiles. Draw a smaller rectangle."),
    ("Could not reach", "No internet connection to the data service. Check your connection and try again."),
    ("Python package", "A Python package is missing. Close the app and start it again with the "
                       "'Start Quarry Maker' file; it installs what is missing."),
]


def hint_for(text: str) -> str:
    for key, hint in HINTS:
        if key.lower() in (text or "").lower():
            return hint
    return "See the log below for details. You can send the message to your helper."


# ----------------------------------------------------------------------
# Blender detection
# ----------------------------------------------------------------------

def detect_blender() -> list[str]:
    found = []
    sysname = platform.system()
    cands = []
    if sysname == "Darwin":
        cands += ["/Applications/Blender.app/Contents/MacOS/Blender",
                  str(Path.home() / "Applications/Blender.app/Contents/MacOS/Blender")]
        cands += sorted(glob.glob("/Applications/Blender*.app/Contents/MacOS/Blender"), reverse=True)
    elif sysname == "Windows":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            cands += sorted(glob.glob(os.path.join(base, "Blender Foundation", "Blender*", "blender.exe")),
                            reverse=True)
        cands += sorted(glob.glob(r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"))
    else:
        cands += ["/snap/bin/blender", "/usr/bin/blender", "/usr/local/bin/blender"]
        cands += sorted(glob.glob(str(Path.home() / "blender*/blender")), reverse=True)
        cands += sorted(glob.glob("/opt/blender*/blender"), reverse=True)
    w = shutil.which("blender")
    if w:
        cands.insert(0, w)
    for c in cands:
        if c and Path(c).exists() and c not in found:
            found.append(c)
    return found


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------

class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.stage = None
        self.lines: list[str] = []
        self.last_error = None
        self.last_stage_done = None
        self.api_key: str | None = None

    def log(self, line):
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > 4000:
                self.lines = self.lines[-3000:]


JOB = Job()


def load_cfg():
    return common.load_config(CONFIG)


def current_paths(cfg):
    run_id = common.resolve_run_id(cfg)
    return common.Paths(Path(cfg["_root"]), run_id)


def stage_status(cfg, paths, stage):
    rec_p = paths.stage_record(stage)
    if stage == "00":
        rep = PROJECT / "logs" / "env_report.json"
        if rep.exists():
            r = json.loads(rep.read_text(encoding="utf-8"))
            return {"status": "ok" if not r.get("missing") else "failed", "summary": [],
                    "missing": r.get("missing", []), "warnings": r.get("warnings", []),
                    "check_images": []}
        return {"status": "todo", "summary": [], "check_images": []}
    rec = common.stage_up_to_date(paths, cfg, stage)
    if not rec and rec_p.exists():
        r = json.loads(rec_p.read_text(encoding="utf-8"))
        if r.get("status") == "failed" and r.get("config_hash") == common.config_hash(cfg):
            return {"status": "failed", "summary": r.get("summary", []),
                    "check_images": [i for i in r.get("check_images", []) if (PROJECT / i).exists()]}
    if rec:
        return {"status": "ok", "summary": rec.get("summary", []),
                "check_images": rec.get("check_images", []), "finished": rec.get("finished")}
    if rec_p.exists():
        return {"status": "outdated", "summary": [], "check_images": []}
    return {"status": "todo", "summary": [], "check_images": []}


def results(paths):
    base = paths.base
    items = [
        ("Measured terrain (Blender)", base / "quarry_base.blend", "blend"),
        ("Quarry with calculated benches (Blender)",
         PROJECT / "terrain" / "10_terraces" / paths.run_id / "quarry_terraced.blend", "blend"),
        ("Rhino files", paths.rhino, "folder"),
        ("Plan linework (DXF for Rhino / CAD)", paths.rhino / "plan", "folder"),
        ("Quarry expansion over time", PROJECT / "terrain" / "20_expansion" / paths.run_id, "folder"),
        ("Gaea package", paths.gaea, "folder"),
        ("Report: terrain", base / "report.md", "file"),
        ("Report: benches", PROJECT / "terrain" / "10_terraces" / paths.run_id / "report_terraces.md", "file"),
        ("All check images", paths.checks, "folder"),
    ]
    out = []
    for label, p, kind in items:
        exists = p.exists() and (kind != "folder" or any(p.iterdir()))
        out.append({"label": label, "path": str(p.relative_to(PROJECT)), "kind": kind, "exists": exists})
    return out


def state():
    s = {"token_ok": True, "platform": platform.system()}
    try:
        cfg = load_cfg()
    except Exception as e:
        return {"error": f"config.yaml could not be read: {e}"}
    aoi = cfg.get("aoi") or {}
    t = cfg.get("terraces") or {}
    pit = None
    if t.get("pit_mask"):
        pp = PROJECT / t["pit_mask"]
        if pp.exists():
            try:
                pit = json.loads(pp.read_text(encoding="utf-8"))
            except Exception:
                pit = None
    s["config"] = {
        "aoi": {k: aoi.get(k) for k in ("west", "south", "east", "north")},
        "buffer_m": aoi.get("buffer_m", 500),
        "crs_epsg": cfg.get("crs_epsg"),
        "blender_path": (cfg.get("blender") or {}).get("path"),
        "bench_height_m": t.get("bench_height_m", "auto"),
        "terraces_enabled": t.get("enabled", True),
        "imagery_mode": (cfg.get("imagery") or {}).get("mode", "esri_tiles"),
        "zoom": (cfg.get("imagery") or {}).get("zoom", 17),
        "max_tiles": (cfg.get("imagery") or {}).get("max_tiles", 3000),
        "pit": pit,
    }
    s["blender_detected"] = detect_blender()
    s["key"] = {"in_env": bool(os.environ.get("OPENTOPO_API_KEY")), "in_app": bool(JOB.api_key)}
    paths = current_paths(cfg)
    s["run_id"] = paths.run_id
    s["stages"] = []
    for st in common.STAGES:
        info = stage_status(cfg, paths, st)
        info.update({"id": st, "title": STAGE_INFO[st][0], "about": STAGE_INFO[st][1]})
        s["stages"].append(info)
    s["results"] = results(paths)
    with JOB.lock:
        s["job"] = {"running": JOB.running, "stage": JOB.stage, "n_lines": len(JOB.lines),
                    "last_error": JOB.last_error, "hint": hint_for(JOB.last_error) if JOB.last_error else None,
                    "last_stage_done": JOB.last_stage_done}
    return s


# ----------------------------------------------------------------------
# Running stages
# ----------------------------------------------------------------------

def run_stages(stages: list[str]):
    env = dict(os.environ)
    if JOB.api_key:
        env["OPENTOPO_API_KEY"] = JOB.api_key
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"            # same text encoding on every system
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        for st in stages:
            with JOB.lock:
                JOB.stage = st
            JOB.log(f"===== Step {st}: {STAGE_INFO[st][0]} =====")
            cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--config", str(CONFIG), "--only", st, "--auto"]
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
            tail = []
            for line in proc.stdout:
                line = line.rstrip("\n")
                if JOB.api_key and JOB.api_key in line:
                    line = line.replace(JOB.api_key, "***")
                JOB.log(line)
                tail.append(line)
            rc = proc.wait()
            if rc != 0:
                reason = next((ln.split("Reason:", 1)[-1] for ln in tail if "Reason:" in ln), None)
                errs = [ln for ln in tail if "ERROR" in ln or ln.strip().startswith(("1.", "2.", "3."))]
                if st == "00":
                    reason = "Setup is not complete: " + " ".join(
                        ln.strip() for ln in tail if ln.strip()[:2] in ("1.", "2.", "3.", "4.", "5."))
                elif errs:
                    reason = errs[-1].split("ERROR", 1)[-1].strip(" :")
                with JOB.lock:
                    JOB.last_error = reason or "\n".join(tail[-5:])
                JOB.log(f"===== Step {st} stopped =====")
                return
            with JOB.lock:
                JOB.last_stage_done = st
            JOB.log(f"===== Step {st} finished =====")
    finally:
        with JOB.lock:
            JOB.running = False
            JOB.stage = None


def start_job(mode: str, stage: str | None = None) -> str | None:
    with JOB.lock:
        if JOB.running:
            return "A step is already running."
    cfg = load_cfg()
    paths = current_paths(cfg)
    order = [s for s in common.STAGES]
    if mode == "one" and stage:
        todo = [stage]
    else:
        pending = [s for s in order if s == "00" or not common.stage_up_to_date(paths, cfg, s)]
        if mode == "next":
            # always (re)check setup before the first real step
            first = next((s for s in pending if s != "00"), None)
            rep = PROJECT / "logs" / "env_report.json"
            env_ok = rep.exists() and not json.loads(rep.read_text(encoding="utf-8")).get("missing")
            todo = ["00"] if not env_ok else ([first] if first else [])
        else:  # all
            todo = pending
    if not todo:
        return "Everything is already done for these settings."
    with JOB.lock:
        JOB.running = True
        JOB.last_error = None
        JOB.lines.append(f"----- {time.strftime('%H:%M:%S')} -----")
    threading.Thread(target=run_stages, args=(todo,), daemon=True).start()
    return None


# ----------------------------------------------------------------------
# Saving settings
# ----------------------------------------------------------------------

def save_settings(data: dict) -> str | None:
    upd = {}
    a = data.get("aoi")
    if a:
        try:
            w, s, e, n = (float(a[k]) for k in ("west", "south", "east", "north"))
        except (TypeError, ValueError, KeyError):
            return "Please give all four corners as numbers."
        if not (-180 <= w < e <= 180 and -85 <= s < n <= 85):
            return "The corners look wrong: west must be smaller than east, south smaller than north."
        upd.update({"aoi.west": round(w, 6), "aoi.south": round(s, 6), "aoi.east": round(e, 6),
                    "aoi.north": round(n, 6)})
        upd["crs_epsg"] = config_io.utm_epsg_for((w + e) / 2, (s + n) / 2)
    if "buffer_m" in data:
        upd["aoi.buffer_m"] = max(0, int(float(data["buffer_m"])))
    if "blender_path" in data:
        bp = (data["blender_path"] or "").strip().strip('"')
        if bp and not Path(bp).exists():
            return f"Blender was not found at: {bp}"
        upd["blender.path"] = bp or None
    if "bench_height_m" in data:
        bh = data["bench_height_m"]
        if bh in (None, "", "auto"):
            upd["terraces.bench_height_m"] = "auto"
        else:
            try:
                v = float(bh)
            except ValueError:
                return "Bench height must be a number in metres, or empty for automatic."
            if not 2 <= v <= 40:
                return "Bench height should be between 2 and 40 m."
            upd["terraces.bench_height_m"] = v
    if "terraces_enabled" in data:
        upd["terraces.enabled"] = bool(data["terraces_enabled"])
    if "pit" in data:
        if data["pit"]:
            p = PROJECT / PIT_FILE
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data["pit"]), encoding="utf-8")
            upd["terraces.pit_mask"] = PIT_FILE
        else:
            upd["terraces.pit_mask"] = None
    if "api_key" in data:
        JOB.api_key = (data["api_key"] or "").strip() or None
    if upd:
        config_io.update_config(CONFIG, upd)
        # a changed setup must be re-checked
        rep = PROJECT / "logs" / "env_report.json"
        if rep.exists() and any(k.startswith(("blender", "aoi")) for k in upd):
            rep.unlink()
    elif "api_key" in data:
        rep = PROJECT / "logs" / "env_report.json"
        if rep.exists():
            rep.unlink()
    return None


def open_path(rel: str) -> str | None:
    p = (PROJECT / rel).resolve()
    try:
        p.relative_to(PROJECT)
    except ValueError:
        return "Not allowed."
    if not p.exists():
        return "That file does not exist yet."
    if p.suffix == ".blend":
        cfg = load_cfg()
        b, _ = common.find_blender(cfg)
        if not b:
            return "Set the Blender location first."
        subprocess.Popen([b, str(p)], cwd=str(p.parent))
        return None
    sysname = platform.system()
    if sysname == "Darwin":
        subprocess.Popen(["open", str(p)])
    elif sysname == "Windows":
        os.startfile(str(p))  # noqa: S606  (local file only)
    else:
        subprocess.Popen(["xdg-open", str(p)])
    return None


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------

SERVE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".md": "text/plain; charset=utf-8", ".json": "application/json", ".geojson": "application/json"}


class Handler(BaseHTTPRequestHandler):
    server_version = "QuarryMaker"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _token_ok(self, qs=None):
        tok = self.headers.get("X-Token") or ((qs or {}).get("t") or [None])[0]
        return secrets.compare_digest(str(tok or ""), TOKEN)

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "forbidden"})
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            if not self._token_ok(qs):
                return self._send(403, "Open Quarry Maker with the link shown in its window.", "text/plain")
            html = (APP_DIR / "index.html").read_text(encoding="utf-8")
            return self._send(200, html, "text/html; charset=utf-8")
        if not self._token_ok(qs):
            return self._send(403, {"error": "forbidden"})
        if u.path == "/api/state":
            return self._send(200, state())
        if u.path == "/api/log":
            since = int((qs.get("since") or ["0"])[0])
            with JOB.lock:
                return self._send(200, {"lines": JOB.lines[since:], "n": len(JOB.lines)})
        if u.path.startswith("/files/"):
            rel = u.path[len("/files/"):]
            p = (PROJECT / rel).resolve()
            try:
                p.relative_to(PROJECT)
            except ValueError:
                return self._send(403, {"error": "forbidden"})
            if p.suffix.lower() not in SERVE_EXT or not p.is_file():
                return self._send(404, {"error": "not found"})
            return self._send(200, p.read_bytes(), SERVE_EXT[p.suffix.lower()])
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._host_ok() or not self._token_ok():
            return self._send(403, {"error": "forbidden"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad request"})
        u = urlparse(self.path)
        try:
            if u.path == "/api/settings":
                err = save_settings(data)
            elif u.path == "/api/run":
                err = start_job(data.get("mode", "next"), data.get("stage"))
            elif u.path == "/api/open":
                err = open_path(data.get("path", ""))
            else:
                return self._send(404, {"error": "not found"})
        except common.PipelineError as e:
            err = str(e)
        except Exception as e:  # show, don't crash the app
            err = f"{type(e).__name__}: {e}"
        return self._send(200 if not err else 400, {"ok": not err, "error": err})


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Quarry Maker app")
    ap.add_argument("--port", type=int, default=0, help="port (default: pick a free one)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"), help="project config (default: config.yaml here)")
    a = ap.parse_args()
    global CONFIG, PROJECT
    CONFIG = Path(a.config).resolve()
    PROJECT = CONFIG.parent
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t={TOKEN}"
    print("\n  Quarry Maker is running.")
    print(f"  If your browser did not open, copy this link into it:\n\n    {url}\n")
    print("  Keep this window open while you work. Close it to stop Quarry Maker.\n", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
