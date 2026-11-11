"""Shared helpers for the quarry base-terrain pipeline.

This module is imported by the plain-Python stages AND by the scripts that run
inside Blender, so it must only import the standard library at module level.
Heavy or optional dependencies (yaml, numpy, PIL, rasterio, pyproj) are
imported lazily inside the functions that need them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# BlenderGIS scene keys, taken from BlenderGIS geoscene.py (class SK):
#   LAT = "latitude", LON = "longitude", CRS = "SRID",
#   CRSX = "crs x", CRSY = "crs y", SCALE = "scale", ZOOM = "zoom"
BGIS_KEYS = {
    "crs": "SRID",
    "crsx": "crs x",
    "crsy": "crs y",
    "lat": "latitude",
    "lon": "longitude",
}

# Config keys that do not change any output (machine-specific). They are left
# out of the config hash so that moving machines does not start a new run.
HASH_EXCLUDE = {("run_id",), ("blender", "path")}

STAGES = ["00", "01", "02", "03", "04", "05", "06"]
STAGE_SCRIPTS = {
    "00": "check_env.py",
    "01": "01_fetch_dem.py",
    "02": "02_fetch_imagery.py",
    "03": "03_preprocess.py",
    "04": "04_build_blender.py",
    "05": "05_export_gaea.py",
    "06": "06_verify.py",
}
STAGE_TITLES = {
    "00": "Environment check",
    "01": "Fetch DEM",
    "02": "Fetch imagery",
    "03": "Preprocess",
    "04": "Build Blender terrain",
    "05": "Gaea package",
    "06": "Verify",
}


class PipelineError(RuntimeError):
    """An expected, user-explainable failure. The message is shown as-is."""


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def load_config(path: Path | str | None = None) -> dict:
    import yaml  # lazy: not available inside Blender's Python

    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise PipelineError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["_config_path"] = str(path.resolve())
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def _strip_for_hash(obj, prefix=()):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).startswith("_"):
                continue
            key = prefix + (k,)
            if key in HASH_EXCLUDE:
                continue
            out[k] = _strip_for_hash(v, key)
        return out
    if isinstance(obj, list):
        return [_strip_for_hash(v, prefix) for v in obj]
    return obj


def config_hash(cfg: dict) -> str:
    canon = json.dumps(_strip_for_hash(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def require_aoi(cfg: dict) -> dict:
    """Return the AOI dict, or stop with a request to fill it in (never guess)."""
    aoi = cfg.get("aoi") or {}
    missing = [k for k in ("west", "south", "east", "north") if aoi.get(k) is None]
    if missing:
        raise PipelineError(
            "The area of interest is not set in config.yaml (missing: "
            + ", ".join(missing)
            + "). Please fill in aoi.west/south/east/north as WGS84 lon/lat. "
            "The pipeline will not guess coordinates."
        )
    w, s, e, n = (float(aoi[k]) for k in ("west", "south", "east", "north"))
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise PipelineError(
            f"AOI looks invalid: west={w}, south={s}, east={e}, north={n}. "
            "Expected west < east and south < north, in decimal degrees."
        )
    return {"west": w, "south": s, "east": e, "north": n,
            "buffer_m": float(aoi.get("buffer_m") or 0.0)}


def crs_string(cfg: dict) -> str:
    epsg = cfg.get("crs_epsg")
    if epsg is None:
        raise PipelineError("crs_epsg is not set in config.yaml. Please set it (e.g. 32635).")
    return f"EPSG:{int(epsg)}"


def find_blender(cfg: dict) -> tuple[str | None, str]:
    """Return (path, where_found). Config wins, then BLENDER_PATH, then PATH."""
    p = (cfg.get("blender") or {}).get("path")
    if p:
        return str(p), "config.yaml blender.path"
    p = os.environ.get("BLENDER_PATH")
    if p:
        return p, "BLENDER_PATH environment variable"
    p = shutil.which("blender")
    if p:
        return p, "PATH"
    return None, ""


# --------------------------------------------------------------------------
# Paths and run IDs
# --------------------------------------------------------------------------

@dataclass
class Paths:
    root: Path
    run_id: str

    @property
    def raw(self) -> Path: return self.root / "data" / "raw"
    @property
    def processed(self) -> Path: return self.root / "data" / "processed"
    @property
    def reference(self) -> Path: return self.root / "data" / "reference"
    @property
    def base(self) -> Path: return self.root / "terrain" / "00_base" / self.run_id
    @property
    def gaea(self) -> Path: return self.root / "exchange" / "gaea" / self.run_id
    @property
    def rhino(self) -> Path: return self.root / "exchange" / "rhino" / self.run_id
    @property
    def checks(self) -> Path: return self.root / "checks" / self.run_id
    @property
    def logs(self) -> Path: return self.root / "logs"
    @property
    def log_file(self) -> Path: return self.logs / f"run_{self.run_id}.log"
    @property
    def run_state(self) -> Path: return self.base / "run.json"

    def stage_record(self, stage: str) -> Path:
        return self.checks / f"stage_{stage}.json"

    def ensure(self):
        for d in (self.raw, self.processed, self.reference, self.base,
                  self.gaea, self.rhino, self.checks, self.logs):
            d.mkdir(parents=True, exist_ok=True)


RUN_RE = re.compile(r"^v(\d{3,})$")


def _existing_runs(root: Path) -> list[str]:
    base = root / "terrain" / "00_base"
    if not base.exists():
        return []
    runs = [d.name for d in base.iterdir() if d.is_dir() and RUN_RE.match(d.name)]
    return sorted(runs, key=lambda r: int(r[1:]))


def resolve_run_id(cfg: dict, new_run: bool = False) -> str:
    """Pick the run ID for this config.

    run_id: auto  -> reuse the latest run if it was made with the same config
                     hash (so interrupted runs resume), else start vNNN+1.
    run_id: v003  -> use exactly that run; refuse if it exists with a
                     different config hash (base outputs are write-once).
    """
    root = Path(cfg["_root"])
    h = config_hash(cfg)
    wanted = str(cfg.get("run_id") or "auto")
    runs = _existing_runs(root)

    if wanted != "auto":
        if not RUN_RE.match(wanted):
            raise PipelineError(f"run_id must be 'auto' or like 'v001', got {wanted!r}")
        state = root / "terrain" / "00_base" / wanted / "run.json"
        if state.exists():
            old = json.loads(state.read_text())
            if old.get("config_hash") != h:
                raise PipelineError(
                    f"Run {wanted} already exists with a different config "
                    f"(hash {old.get('config_hash')} vs {h}). Base outputs are "
                    "write-once: set run_id: auto (or a new vNNN) to start a new run."
                )
        return wanted

    if runs and not new_run:
        latest = runs[-1]
        state = root / "terrain" / "00_base" / latest / "run.json"
        if state.exists() and json.loads(state.read_text()).get("config_hash") == h:
            return latest
    n = int(runs[-1][1:]) + 1 if runs else 1
    return f"v{n:03d}"


def check_run_matches(cfg: dict, run_id: str) -> None:
    """Refuse to write into an existing run made with a different config."""
    if not RUN_RE.match(run_id):
        raise PipelineError(f"Run ID must look like v001, got {run_id!r}")
    state = Path(cfg["_root"]) / "terrain" / "00_base" / run_id / "run.json"
    if state.exists():
        old = json.loads(state.read_text()).get("config_hash")
        if old != config_hash(cfg):
            raise PipelineError(
                f"Run {run_id} was made with a different config (hash {old} vs "
                f"{config_hash(cfg)}). Base outputs are write-once; use a new run.")


def init_run(cfg: dict, run_id: str) -> Paths:
    paths = Paths(Path(cfg["_root"]), run_id)
    paths.ensure()
    if not paths.run_state.exists():
        write_json(paths.run_state, {
            "run_id": run_id,
            "config_hash": config_hash(cfg),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": _strip_for_hash(cfg),
        })
    return paths


# --------------------------------------------------------------------------
# Logging, records, CLI
# --------------------------------------------------------------------------

def setup_logging(paths: Paths | None, name: str) -> logging.Logger:
    logger = logging.getLogger("quarry")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(f"%(asctime)s [{name}] %(levelname)s %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if paths is not None:
            paths.logs.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(paths.log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def stage_args(description: str, argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--run", default=None, help="run ID (vNNN); resolved from config if omitted")
    ap.add_argument("--force", action="store_true",
                    help="redo work even if outputs are up to date (never overwrites terrain/00_base)")
    return ap.parse_args(argv)


def stage_context(description: str, stage: str, argv=None):
    """Common start-up for plain-Python stages: args, config, run, logger."""
    args = stage_args(description, argv)
    cfg = load_config(args.config)
    if args.run:
        check_run_matches(cfg, args.run)
    run_id = args.run or resolve_run_id(cfg)
    paths = init_run(cfg, run_id)
    log = setup_logging(paths, f"{stage} {STAGE_TITLES[stage]}")
    return args, cfg, paths, log


def run_main(main, stage: str) -> None:
    """Run a stage's main(); turn expected failures into a short message and a
    non-zero exit code (2), and log it to the run log when one is open."""
    try:
        code = main()
    except PipelineError as e:
        msg = f"Stage {stage} failed: {e}"
        lg = logging.getLogger("quarry")
        if lg.handlers:
            lg.error(msg)
        else:
            print("ERROR " + msg, file=sys.stderr)
        sys.exit(2)
    sys.exit(code or 0)


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def rel(paths: Paths, p) -> str:
    try:
        return str(Path(p).resolve().relative_to(paths.root.resolve()))
    except ValueError:
        return str(p)


def stage_up_to_date(paths: Paths, cfg: dict, stage: str) -> dict | None:
    """Return the stage record if the stage already succeeded for this config
    and all its outputs still exist; else None."""
    rec_path = paths.stage_record(stage)
    if not rec_path.exists():
        return None
    rec = read_json(rec_path)
    if rec.get("status") != "ok" or rec.get("config_hash") != config_hash(cfg):
        return None
    for p in rec.get("outputs", []) + rec.get("check_images", []):
        if not (paths.root / p).exists():
            return None
    return rec


def finish_stage(paths: Paths, cfg: dict, stage: str, summary: list[str],
                 outputs: list, check_images: list, log: logging.Logger,
                 extra: dict | None = None) -> dict:
    rec = {
        "stage": stage,
        "title": STAGE_TITLES[stage],
        "status": "ok",
        "run_id": paths.run_id,
        "config_hash": config_hash(cfg),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "outputs": [rel(paths, p) for p in outputs],
        "check_images": [rel(paths, p) for p in check_images],
    }
    if extra:
        rec.update(extra)
    write_json(paths.stage_record(stage), rec)
    log.info("Stage %s done. Summary:", stage)
    for line in summary:
        log.info("  - %s", line)
    for p in rec["check_images"]:
        log.info("  check image: %s", p)
    return rec


def write_once_guard(path: Path) -> None:
    """terrain/00_base/vNNN is write-once: refuse to replace an existing file."""
    if Path(path).exists():
        raise PipelineError(
            f"{path} already exists. terrain/00_base is write-once per run; "
            "start a new run (run_id: auto with a changed config, or --new-run)."
        )


def bbox_hash(*values) -> str:
    s = ",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# --------------------------------------------------------------------------
# Geometry helpers (lazy imports)
# --------------------------------------------------------------------------

def buffered_extent_utm(cfg: dict) -> tuple[float, float, float, float]:
    """AOI (lon/lat) -> projected CRS, buffered by buffer_m. Returns
    (xmin, ymin, xmax, ymax) in the projected CRS, not yet grid-snapped."""
    from pyproj import Transformer

    aoi = require_aoi(cfg)
    tr = Transformer.from_crs("EPSG:4326", crs_string(cfg), always_xy=True)
    xmin, ymin, xmax, ymax = tr.transform_bounds(
        aoi["west"], aoi["south"], aoi["east"], aoi["north"], densify_pts=21)
    b = aoi["buffer_m"]
    return xmin - b, ymin - b, xmax + b, ymax + b


def target_resolution(cfg: dict) -> float:
    r = float((cfg.get("dem") or {}).get("target_resolution_m") or 30)
    if r <= 0:
        raise PipelineError("dem.target_resolution_m must be > 0")
    return r


def target_extent(cfg: dict) -> tuple[float, float, float, float]:
    """Buffered AOI in the projected CRS, snapped OUTWARD to a grid of
    multiples of the target resolution. This is the shared extent of every
    processed raster, the mesh and the Gaea package."""
    import math

    r = target_resolution(cfg)
    xmin, ymin, xmax, ymax = buffered_extent_utm(cfg)
    return (math.floor(xmin / r) * r, math.floor(ymin / r) * r,
            math.ceil(xmax / r) * r, math.ceil(ymax / r) * r)


def utm_extent_to_lonlat(cfg: dict, extent, margin_deg: float = 0.0):
    from pyproj import Transformer

    tr = Transformer.from_crs(crs_string(cfg), "EPSG:4326", always_xy=True)
    w, s, e, n = tr.transform_bounds(*extent, densify_pts=21)
    return w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg


def hillshade(z, dx: float, dy: float, azimuth: float = 315.0, altitude: float = 45.0):
    """Classic Horn-style hillshade, 0..1. z is a 2-D array with row 0 = north."""
    import numpy as np

    z = np.asarray(z, dtype="float64")
    gy, gx = np.gradient(z, dy, dx)
    # rows go south, so north-facing gradient is -gy
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def save_gray_png(arr01, path: Path, max_side: int | None = 2048):
    import numpy as np
    from PIL import Image

    img = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype("uint8"), "L")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
