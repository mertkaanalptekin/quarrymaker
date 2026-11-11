"""Stage 02 - fetch imagery.

esri_tiles   : download Esri World Imagery XYZ tiles that cover the buffered
               AOI, cache each tile, mosaic them and write a georeferenced
               GeoTIFF in Web Mercator (EPSG:3857). Reprojection is stage 03.
user_geotiff : copy the user's georeferenced GeoTIFF into data/raw/.
none         : no imagery; stage 03 makes a neutral hillshade texture.

The Google Earth image in data/reference/ has no georeferencing and is used
only for visual comparison in stage 06.
"""
from __future__ import annotations

import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ESRI_ATTRIBUTION = ("Esri World Imagery - Source: Esri, Maxar, Earthstar Geographics, "
                    "and the GIS User Community. Powered by Esri.")
ESRI_TERMS = ("Esri World Imagery is licensed under the Esri Master License Agreement / "
              "ArcGIS Online terms of use. Attribution is required on every map or image "
              "that shows it. Esri states the standard World Imagery layer is not intended "
              "for exporting tiles for offline use (it offers a separate 'World Imagery "
              "(for Export)' layer for that). Check the current terms on the World Imagery "
              "item page before publishing.")
WEBM = 20037508.342789244
MARGIN_DEG = 0.001  # ~100 m extra so reprojection never reads past the mosaic
USER_AGENT = "quarry-terrain-pipeline/1.0"


def imagery_cfg(cfg) -> dict:
    return cfg.get("imagery") or {}


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_bounds_merc(x, y, z):
    size = 2 * WEBM / 2 ** z
    xmin = -WEBM + x * size
    ymax = WEBM - y * size
    return xmin, ymax - size, xmin + size, ymax


def tile_range(cfg):
    z = int(imagery_cfg(cfg).get("zoom", 17))
    w, s, e, n = common.utm_extent_to_lonlat(cfg, common.target_extent(cfg), MARGIN_DEG)
    x0, y0 = lonlat_to_tile(w, n, z)
    x1, y1 = lonlat_to_tile(e, s, z)
    return z, x0, y0, x1, y1


def ortho_raw_path(cfg) -> Path:
    raw = Path(cfg["_root"]) / "data" / "raw"
    ic = imagery_cfg(cfg)
    mode = ic.get("mode", "esri_tiles")
    if mode == "esri_tiles":
        z, x0, y0, x1, y1 = tile_range(cfg)
        h = common.bbox_hash(ic.get("tile_url", ""), z, x0, y0, x1, y1)
        return raw / f"ortho_esri_z{z}_{h}.tif"
    if mode == "user_geotiff":
        src = _user_src(cfg)
        h = common.bbox_hash(str(src.resolve()), src.stat().st_size, int(src.stat().st_mtime))
        return raw / f"ortho_user_{h}.tif"
    return raw / "ortho_none.tif"


def _user_src(cfg) -> Path:
    p = imagery_cfg(cfg).get("user_geotiff")
    if not p:
        raise common.PipelineError("imagery.mode is user_geotiff but imagery.user_geotiff is empty.")
    src = Path(p)
    if not src.is_absolute():
        src = Path(cfg["_root"]) / src
    if not src.exists():
        raise common.PipelineError(f"imagery.user_geotiff not found: {src}")
    return src


def fetch_tiles(cfg, log) -> tuple[Path, dict]:
    import numpy as np
    import rasterio
    import requests
    from PIL import Image
    from rasterio.transform import from_origin

    ic = imagery_cfg(cfg)
    url_t = ic.get("tile_url") or \
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    throttle = float(ic.get("throttle_s", 0.15))
    max_tiles = int(ic.get("max_tiles", 3000))
    z, x0, y0, x1, y1 = tile_range(cfg)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    if nx * ny > max_tiles:
        raise common.PipelineError(
            f"The AOI needs {nx * ny} tiles at zoom {z} (cap imagery.max_tiles = {max_tiles}). "
            "Lower imagery.zoom or raise the cap.")
    log.info("Imagery: %d x %d = %d tiles at zoom %d", nx, ny, nx * ny, z)

    cache = Path(cfg["_root"]) / "data" / "raw" / "tiles" / "esri_world_imagery" / str(z)
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    mosaic = np.zeros((ny * 256, nx * 256, 3), dtype="uint8")
    downloaded = cached = 0
    for j, y in enumerate(range(y0, y1 + 1)):
        for i, x in enumerate(range(x0, x1 + 1)):
            tp = cache / str(x) / f"{y}.img"
            if tp.exists():
                cached += 1
            else:
                url = url_t.format(z=z, x=x, y=y)
                for attempt in range(4):
                    try:
                        r = sess.get(url, timeout=30)
                        if r.status_code == 200 and r.content:
                            break
                        err = f"HTTP {r.status_code}"
                    except requests.RequestException as ex:
                        err = str(ex)
                    time.sleep(2 ** attempt)
                else:
                    raise common.PipelineError(f"Tile download failed ({err}): z{z} x{x} y{y}")
                tp.parent.mkdir(parents=True, exist_ok=True)
                tp.write_bytes(r.content)
                downloaded += 1
                time.sleep(throttle)
            with Image.open(tp) as im:
                mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = np.asarray(im.convert("RGB"))
        log.info("  row %d/%d", j + 1, ny)

    xmin, _, _, ymax = tile_bounds_merc(x0, y0, z)
    px = 2 * WEBM / 2 ** z / 256
    out = ortho_raw_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", driver="GTiff", width=nx * 256, height=ny * 256, count=3,
                       dtype="uint8", crs="EPSG:3857", transform=from_origin(xmin, ymax, px, px),
                       compress="deflate", tiled=True, photometric="RGB") as ds:
        ds.write(mosaic.transpose(2, 0, 1))
    tmp.replace(out)
    info = {"mode": "esri_tiles", "zoom": z, "tiles": nx * ny, "tiles_downloaded": downloaded,
            "tiles_from_cache": cached, "tile_url": url_t, "tile_range": [x0, y0, x1, y1],
            "pixel_size_webmercator_m": px, "attribution": ESRI_ATTRIBUTION,
            "terms": ESRI_TERMS, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return out, info


def main(argv=None) -> int:
    args, cfg, paths, log = common.stage_context(__doc__, "02", argv)
    common.require_aoi(cfg)
    mode = imagery_cfg(cfg).get("mode", "esri_tiles")
    preview = paths.checks / "02_imagery_preview.jpg"

    if not args.force and common.stage_up_to_date(paths, cfg, "02"):
        log.info("Stage 02 up to date - skipping.")
        return 0

    if mode == "none":
        info = {"mode": "none", "attribution": "none",
                "note": "No imagery; stage 03 writes a neutral hillshade texture."}
        from PIL import Image
        Image.new("RGB", (256, 256), (128, 128, 128)).save(preview)
        common.finish_stage(paths, cfg, "02", ["Imagery mode 'none' - no imagery fetched."],
                            [], [preview], log, extra={"imagery": info, "ortho_raw": None})
        return 0

    out = ortho_raw_path(cfg)
    side = out.with_suffix(".info.json")
    if out.exists() and side.exists():
        log.info("Imagery cache hit: %s", out)
        info = common.read_json(side)
    elif mode == "esri_tiles":
        log.info("Esri usage terms: %s", ESRI_TERMS)
        out, info = fetch_tiles(cfg, log)
        common.write_json(side, info)
    elif mode == "user_geotiff":
        import rasterio
        src = _user_src(cfg)
        with rasterio.open(src) as ds:
            if ds.crs is None:
                raise common.PipelineError(
                    f"{src} has no CRS - it is not georeferenced. Use a georeferenced GeoTIFF; "
                    "un-georeferenced images belong in data/reference/ for visual comparison only.")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        info = {"mode": "user_geotiff", "source_file": str(src),
                "attribution": "User-supplied GeoTIFF (add its credit line here)"}
        common.write_json(side, info)
    else:
        raise common.PipelineError(f"Unknown imagery.mode {mode!r}")

    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.open(out) as ds:
        scale = max(ds.width, ds.height) / 1600
        h, w = max(1, int(ds.height / max(scale, 1))), max(1, int(ds.width / max(scale, 1)))
        arr = ds.read(indexes=[1, 2, 3] if ds.count >= 3 else [1, 1, 1], out_shape=(3, h, w))
        desc = f"{ds.width} x {ds.height} px, {ds.crs}, pixel {ds.res[0]:.3f} {ds.crs.linear_units}"
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype("uint8")
    Image.fromarray(arr.transpose(1, 2, 0)).save(preview, quality=88)

    summary = [f"Mode: {mode}", f"Mosaic: {common.rel(paths, out)} - {desc}",
               f"Attribution: {info.get('attribution')}"]
    if mode == "esri_tiles":
        summary.append(f"Tiles: {info.get('tiles')} at zoom {info.get('zoom')} "
                       f"({info.get('tiles_downloaded', 0)} downloaded, "
                       f"{info.get('tiles_from_cache', 0)} cached)")
        summary.append("Terms: " + ESRI_TERMS)
    common.finish_stage(paths, cfg, "02", summary, [out], [preview], log,
                        extra={"imagery": info, "ortho_raw": common.rel(paths, out)})
    return 0


if __name__ == "__main__":
    common.run_main(main, "02")
