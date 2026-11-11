"""Build a SYNTHETIC test project to exercise the pipeline offline.

Creates <dest>/config.yaml and pre-seeds the caches that stages 01/02 look
for (a fake 1-arc-second DEM in EPSG:4326 and fake Esri tiles), so the full
pipeline runs without network access or an API key. Nothing here is real
data - the AOI is an arbitrary test box and must never be used for the
project itself.

    python tests/make_synthetic_fixture.py /tmp/quarry_test --blender /path/to/blender
    python run_pipeline.py --config /tmp/quarry_test/config.yaml --auto
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CONFIG = """run_id: auto
aoi:   # SYNTHETIC TEST BOX - not a real site
  west: {w}
  south: {s}
  east: {e}
  north: {n}
  buffer_m: 300
crs_epsg: 32635
dem:
  source: COP30
  target_resolution_m: 30
  resampling: bilinear
imagery:
  mode: {mode}
  zoom: 15
  user_geotiff: null
  tile_url: "http://offline.invalid/{{z}}/{{y}}/{{x}}"
  throttle_s: 0
  max_tiles: 500
  resampling: bilinear
blender:
  path: {blender}
  mesh_decimate_ratio: 1.0
  check_render_px: 480
  check_render_samples: 8
gaea:
  size_px: {gaea_px}
  bit_depth: 16
rhino:
  formats: [obj, fbx]
"""


def main():
    import numpy as np
    import rasterio
    from PIL import Image, ImageDraw
    from rasterio.transform import from_origin

    import common

    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--blender", required=True)
    ap.add_argument("--mode", default="esri_tiles")
    ap.add_argument("--gaea-px", type=int, default=512)
    a = ap.parse_args()
    dest = Path(a.dest)
    (dest / "data" / "reference").mkdir(parents=True, exist_ok=True)
    w, s, e, n = 27.40, 40.30, 27.43, 40.32
    (dest / "config.yaml").write_text(CONFIG.format(w=w, s=s, e=e, n=n, mode=a.mode,
                                                    blender=a.blender, gaea_px=a.gaea_px))
    cfg = common.load_config(dest / "config.yaml")

    # --- fake DEM (EPSG:4326, 1 arc-second) with a terraced pit and a hill ---
    fetch = _load("fetch_dem", "01_fetch_dem.py")
    W_, S_, E_, N_ = fetch.request_bbox(cfg)
    res = 1 / 3600
    cols, rows = math.ceil((E_ - W_) / res), math.ceil((N_ - S_) / res)
    lon = W_ + (np.arange(cols) + 0.5) * res
    lat = N_ - (np.arange(rows) + 0.5) * res
    LON, LAT = np.meshgrid(lon, lat)
    xm = (LON - (w + e) / 2) * 111320 * math.cos(math.radians(40.31))
    ym = (LAT - (s + n) / 2) * 110540
    z = 420 + 0.02 * xm + 0.015 * ym
    z += 120 * np.exp(-(((xm - 900) / 700) ** 2 + ((ym + 500) / 600) ** 2))
    rr = np.hypot(xm + 300, ym - 200)
    pit = np.clip((700 - rr) / 700, 0, 1) * 150
    z -= np.floor(pit / 15) * 15  # 15 m benches
    z = z.astype("float32")
    z[rows // 3:rows // 3 + 3, cols // 2:cols // 2 + 4] = -32767  # a nodata hole
    out = fetch.dem_cache_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", width=cols, height=rows, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(W_, N_, res, res),
                       nodata=-32767) as ds:
        ds.write(z, 1)
    print("fake DEM:", out, z.shape)

    # --- fake Esri tiles: colour from position + grid lines every 250 m ---
    if a.mode == "esri_tiles":
        imag = _load("fetch_imagery", "02_fetch_imagery.py")
        zl, x0, y0, x1, y1 = imag.tile_range(cfg)
        cache = dest / "data" / "raw" / "tiles" / "esri_world_imagery" / str(zl)
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:3857", "EPSG:32635", always_xy=True)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                bx0, by0, bx1, by1 = imag.tile_bounds_merc(x, y, zl)
                px = (bx1 - bx0) / 256
                mx = bx0 + (np.arange(256) + 0.5) * px
                my = by1 - (np.arange(256) + 0.5) * px
                MX, MY = np.meshgrid(mx, my)
                ux, uy = tr.transform(MX, MY)
                rgb = np.zeros((256, 256, 3), dtype="uint8")
                rgb[..., 0] = (ux / 7) % 200 + 40
                rgb[..., 1] = (uy / 9) % 200 + 40
                rgb[..., 2] = 120
                grid = ((ux % 250) < 6) | ((uy % 250) < 6)
                rgb[grid] = 255
                p = cache / str(x) / f"{y}.img"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(p, format="JPEG", quality=90)
        print("fake tiles:", (x1 - x0 + 1) * (y1 - y0 + 1))

    ref = Image.new("RGB", (800, 600), (90, 110, 80))
    ImageDraw.Draw(ref).text((20, 20), "SYNTHETIC REFERENCE IMAGE", fill="white")
    ref.save(dest / "data" / "reference" / "google_earth_reference.jpg")
    print("config:", dest / "config.yaml")


if __name__ == "__main__":
    main()
