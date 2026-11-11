"""Synthetic open-pit quarry with KNOWN ground truth, to measure how well
stage 07 recovers benches.

  make      write a test project: config.yaml, a 30 m-like DEM (area-averaged
            truth + noise, EPSG:4326 1 arc-second) and zoom-17 "satellite"
            tiles rendered from the true 1 m surface (sun shading, rock /
            vegetation colours). Caches are pre-seeded, so no network is used.
  evaluate  after the pipeline ran: compare the interpreted terrace DEM and
            the plain upsampled DEM against the truth.

    python tests/synthetic_quarry.py make /tmp/q --blender /path/to/blender
    python run_pipeline.py --config /tmp/q/config.yaml --auto
    python tests/synthetic_quarry.py evaluate /tmp/q

Nothing here is real data; the AOI is an arbitrary test box.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

# --- truth parameters ---------------------------------------------------
LON0, LAT0 = 27.4100, 40.3100          # site centre (test box)
BENCH_H = 12.0                          # true bench height (m)
BENCH_W = 10.0                          # true bench floor width (m)
FACE_DEG = 70.0                         # true face angle
FLOOR_Z = 294.0                         # pit floor; benches at FLOOR_Z + k*H (fixed RL)
FLOOR_R = 140.0                         # floor radius (m, before shape modulation)
PIT_OFFSET = (-120.0, 80.0)             # pit centre relative to site centre (m)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def site_centre_utm():
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True).transform(LON0, LAT0)


def ground(dx, dy):
    """Natural ground around the pit (local metres from site centre)."""
    z = 420 + 0.025 * dx + 0.012 * dy
    z += 45 * np.exp(-(((dx - 650) / 420) ** 2 + ((dy + 380) / 380) ** 2))
    z += 6 * np.sin(dx / 170.0) * np.cos(dy / 230.0)
    return z


def pit_radius(dx, dy):
    """Irregular pit 'radius' coordinate: benches follow its iso-lines."""
    px, py = dx - PIT_OFFSET[0], dy - PIT_OFFSET[1]
    th = np.arctan2(py, px)
    r = np.hypot(px * 1.0, py * 1.25)
    return r / (1 + 0.13 * np.sin(3 * th + 0.6) + 0.06 * np.sin(5 * th - 1.1))


def pit_surface(dx, dy):
    face_w = BENCH_H / math.tan(math.radians(FACE_DEG))
    period = BENCH_W + face_w
    s = (pit_radius(dx, dy) - FLOOR_R) / period
    k = np.floor(s)
    f = s - k
    rise = np.clip((f - BENCH_W / period) / (face_w / period), 0, 1)
    z = FLOOR_Z + BENCH_H * (k + rise)
    return np.where(s < 0, FLOOR_Z, z)


def truth(dx, dy):
    """True surface (m). dx, dy: local metres from the site centre."""
    return np.minimum(ground(dx, dy), pit_surface(dx, dy))


def truth_utm(x, y):
    cx, cy = site_centre_utm()
    return truth(np.asarray(x) - cx, np.asarray(y) - cy)


# --- synthetic satellite colour -------------------------------------------

def _noise(dx, dy, scale, seed):
    # cheap smooth value noise from a few sinusoids
    rng = np.random.default_rng(seed)
    out = 0
    for _ in range(6):
        a, b, p = rng.uniform(-1, 1, 3)
        k = rng.uniform(0.5, 2.0) / scale
        out = out + np.sin((a * dx + b * dy) * k * 2 * np.pi + p * 6)
    return out / 6


def colour(dx, dy, px_m, rng):
    z = truth(dx, dy)
    g = ground(dx, dy)
    e = 0.5
    zx = (truth(dx + e, dy) - truth(dx - e, dy)) / (2 * e)
    zy = (truth(dx, dy + e) - truth(dx, dy - e)) / (2 * e)
    n = np.stack([-zx, -zy, np.ones_like(zx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    az, el = math.radians(150), math.radians(48)   # late-morning sun from SSE
    s = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    lam = np.clip(n @ s, 0, None) / s[2]
    in_pit = (g - z) > 0.3
    rock = np.array([0.80, 0.77, 0.71])
    veg_a, veg_b = np.array([0.33, 0.40, 0.24]), np.array([0.62, 0.55, 0.42])
    t = np.clip(0.5 + 0.9 * _noise(dx, dy, 180, 1), 0, 1)[..., None]
    veg = veg_a * (1 - t) + veg_b * t
    alb = np.where(in_pit[..., None], rock * (0.92 + 0.08 * _noise(dx, dy, 40, 2))[..., None], veg)
    rgb = alb * (0.30 + 0.70 * lam[..., None])
    rgb += rng.normal(0, 0.02, rgb.shape)
    return (np.clip(rgb, 0, 1) * 255).astype("uint8")


CONFIG = """run_id: auto
aoi:   # SYNTHETIC TEST BOX - not a real site
  west: {w:.6f}
  south: {s:.6f}
  east: {e:.6f}
  north: {n:.6f}
  buffer_m: 300
crs_epsg: 32635
dem:
  source: COP30
  target_resolution_m: 30
  resampling: bilinear
imagery:
  mode: esri_tiles
  zoom: 17
  user_geotiff: null
  tile_url: "http://offline.invalid/{{z}}/{{y}}/{{x}}"
  throttle_s: 0
  max_tiles: 1000
  resampling: bilinear
blender:
  path: {blender}
  mesh_decimate_ratio: 1.0
  check_render_px: 480
  check_render_samples: 8
gaea:
  size_px: 512
  bit_depth: 16
rhino:
  formats: [obj, fbx]
terraces:
  enabled: true
"""


def make(dest: Path, blender: str):
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from rasterio.transform import from_origin

    import common

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "data" / "reference").mkdir(parents=True, exist_ok=True)
    dl, dt = 0.0085, 0.0065
    (dest / "config.yaml").write_text(CONFIG.format(w=LON0 - dl, s=LAT0 - dt, e=LON0 + dl,
                                                    n=LAT0 + dt, blender=blender))
    cfg = common.load_config(dest / "config.yaml")
    cx, cy = site_centre_utm()
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
    rng = np.random.default_rng(7)

    # DEM: 1 arc-second cells, each the mean of 6x6 truth samples (+ noise)
    fetch = _load("fetch_dem", "01_fetch_dem.py")
    W_, S_, E_, N_ = fetch.request_bbox(cfg)
    res = 1 / 3600
    cols, rows = math.ceil((E_ - W_) / res), math.ceil((N_ - S_) / res)
    sub = (np.arange(6) + 0.5) / 6
    acc = np.zeros((rows, cols))
    for a in sub:
        for b in sub:
            lon = W_ + (np.arange(cols) + a) * res
            lat = N_ - (np.arange(rows) + b) * res
            LON, LAT = np.meshgrid(lon, lat)
            x, y = to_utm.transform(LON, LAT)
            acc += truth(x - cx, y - cy)
    dem = (acc / 36 + rng.normal(0, 0.8, acc.shape)).astype("float32")
    out = fetch.dem_cache_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", width=cols, height=rows, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(W_, N_, res, res),
                       nodata=-32767) as ds:
        ds.write(dem, 1)

    imag = _load("fetch_imagery", "02_fetch_imagery.py")
    zl, x0, y0, x1, y1 = imag.tile_range(cfg)
    cache = dest / "data" / "raw" / "tiles" / "esri_world_imagery" / str(zl)
    tr = Transformer.from_crs("EPSG:3857", "EPSG:32635", always_xy=True)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            bx0, by0, bx1, by1 = imag.tile_bounds_merc(x, y, zl)
            px = (bx1 - bx0) / 256
            MX, MY = np.meshgrid(bx0 + (np.arange(256) + 0.5) * px, by1 - (np.arange(256) + 0.5) * px)
            ux, uy = tr.transform(MX, MY)
            p = cache / str(x) / f"{y}.img"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(colour(ux - cx, uy - cy, px, rng)).save(p, format="JPEG", quality=88)
    print(f"truth: bench height {BENCH_H} m, width {BENCH_W} m, face {FACE_DEG} deg; "
          f"DEM {rows}x{cols}; tiles {(x1 - x0 + 1) * (y1 - y0 + 1)}")
    print("config:", dest / "config.yaml")


def evaluate(dest: Path) -> dict:
    import rasterio

    runs = sorted((dest / "terrain" / "10_terraces").glob("v*"))
    if not runs:
        raise SystemExit("no terrain/10_terraces run found")
    run = runs[-1]
    params = json.loads((run / "terraces_params.json").read_text())
    with rasterio.open(run / "terraces_dem.tif") as ds:
        zt = ds.read(1).astype("float64")
        T = ds.transform
    with rasterio.open(run / "measured_upsampled.tif") as ds:
        zm = ds.read(1).astype("float64")
    with rasterio.open(run / "classes.tif") as ds:
        cls = ds.read(1)
    h, w = zt.shape
    xs = T.c + (np.arange(w) + 0.5) * T.a
    ys = T.f + (np.arange(h) + 0.5) * T.e
    X, Y = np.meshgrid(xs, ys)
    cx, cy = site_centre_utm()
    ztrue = truth(X - cx, Y - cy)
    in_pit_true = (ground(X - cx, Y - cy) - ztrue) > 0.3
    zone = in_pit_true & np.isin(cls, (2, 3))  # bench zone where truth has benches

    def rmse(a):
        return float(np.sqrt(np.mean((a[zone] - ztrue[zone]) ** 2)))

    # bench-scale error: remove a 60 m local mean so only the step pattern counts
    import terraces as t
    k = max(1, int(round(30 / abs(T.a))))
    hp = lambda a: a - t.box_blur(a, k)  # noqa: E731

    res = {
        "true_bench_height_m": BENCH_H,
        "estimated_bench_height_m": params["bench_height_m"],
        "bench_height_error_m": params["bench_height_m"] - BENCH_H,
        "est_bench_floors_m": params["bench_floor_levels_m"],
        "bench_floor_level_error_m": [round(min(abs(f - (FLOOR_Z + BENCH_H * n)) for n in range(-2, 20)), 2)
                                      for f in params["bench_floor_levels_m"]],
        "zone_cells": int(zone.sum()),
        "rmse_measured_upsampled_m": rmse(zm),
        "rmse_terraced_m": rmse(zt),
        "bench_scale_rmse_measured_m": float(np.sqrt(np.mean((hp(zm) - hp(ztrue))[zone] ** 2))),
        "bench_scale_rmse_terraced_m": float(np.sqrt(np.mean((hp(zt) - hp(ztrue))[zone] ** 2))),
        "corr_bench_pattern_terraced": float(np.corrcoef(hp(zt)[zone], hp(ztrue)[zone])[0, 1]),
        "corr_bench_pattern_measured": float(np.corrcoef(hp(zm)[zone], hp(ztrue)[zone])[0, 1]),
        "pct_within_3m_measured": float(100 * np.mean(np.abs(zm - ztrue)[zone] <= 3)),
        "pct_within_3m_terraced": float(100 * np.mean(np.abs(zt - ztrue)[zone] <= 3)),
        "pit_mask_iou": float((in_pit_true & (cls > 0)).sum() / max(1, (in_pit_true | (cls > 0)).sum())),
    }
    print(json.dumps(res, indent=2))
    # visual: hillshade of measured | interpreted | truth over the pit
    import common
    from PIL import Image
    rows, cols = np.where(zone)
    sl = (slice(max(0, rows.min() - 40), rows.max() + 40), slice(max(0, cols.min() - 40), cols.max() + 40))
    ims = [Image.fromarray((common.hillshade(a[sl], abs(T.a), abs(T.a)) * 255).astype("uint8"))
           for a in (zm, zt, ztrue)]
    sheet = Image.new("L", (sum(i.width for i in ims) + 20, ims[0].height), 255)
    x = 0
    for i in ims:
        sheet.paste(i, (x, 0))
        x += i.width + 10
    out = dest / "eval_measured_terraced_truth.png"
    sheet.save(out)
    print("visual:", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("dest")
    m.add_argument("--blender", required=True)
    ev = sp.add_parser("evaluate")
    ev.add_argument("dest")
    a = ap.parse_args()
    if a.cmd == "make":
        make(Path(a.dest), a.blender)
    else:
        evaluate(Path(a.dest))
