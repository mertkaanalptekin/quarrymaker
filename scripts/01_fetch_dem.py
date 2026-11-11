"""Stage 01 - fetch the DEM from the OpenTopography Global DEM API.

Requests the buffered AOI (plus a few cells of margin for reprojection) from
the `globaldem` endpoint and caches the untouched GeoTIFF in data/raw/.
The API key is read from OPENTOPO_API_KEY and never written anywhere.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

API_URL = "https://portal.opentopography.org/API/globaldem"
# Margin around the request so bilinear/cubic reprojection never reads
# outside the downloaded data (~6 arc-seconds, about 6 COP30 cells).
MARGIN_DEG = 6 / 3600.0

DEM_NOTES = {
    "COP30": "Copernicus GLO-30 (TanDEM-X, acquired 2011-2015; edited DSM). 1 arc-second grid.",
    "COP90": "Copernicus GLO-90 (TanDEM-X, acquired 2011-2015). 3 arc-second grid.",
    "SRTMGL1": "SRTM GL1 (Shuttle Radar Topography Mission, February 2000). 1 arc-second grid.",
    "SRTMGL3": "SRTM GL3 (February 2000). 3 arc-second grid.",
    "NASADEM": "NASADEM (reprocessed SRTM, February 2000). 1 arc-second grid.",
    "AW3D30": "ALOS World 3D 30 m (PRISM, acquired 2006-2011). 1 arc-second grid.",
}


def dem_source(cfg) -> str:
    return str((cfg.get("dem") or {}).get("source") or "COP30")


def request_bbox(cfg):
    """Lon/lat request box (west, south, east, north)."""
    return common.utm_extent_to_lonlat(cfg, common.target_extent(cfg), MARGIN_DEG)


def dem_cache_path(cfg) -> Path:
    w, s, e, n = request_bbox(cfg)
    h = common.bbox_hash(dem_source(cfg), round(w, 6), round(s, 6), round(e, 6), round(n, 6))
    return Path(cfg["_root"]) / "data" / "raw" / f"dem_{dem_source(cfg)}_{h}.tif"


def download(cfg, out: Path, log) -> dict:
    import requests

    key = os.environ.get("OPENTOPO_API_KEY")
    if not key:
        raise common.PipelineError(
            "OPENTOPO_API_KEY is not set and the DEM is not cached. Please export your "
            "OpenTopography API key in the environment.")
    w, s, e, n = request_bbox(cfg)
    params = {"demtype": dem_source(cfg), "south": f"{s:.6f}", "north": f"{n:.6f}",
              "west": f"{w:.6f}", "east": f"{e:.6f}", "outputFormat": "GTiff"}
    log.info("Requesting %s from OpenTopography globaldem, bbox W%.6f S%.6f E%.6f N%.6f",
             params["demtype"], w, s, e, n)
    t0 = time.time()
    try:
        r = requests.get(API_URL, params={**params, "API_Key": key}, timeout=300, stream=True)
    except requests.RequestException as ex:
        raise common.PipelineError(f"Could not reach OpenTopography: {ex}") from None

    if r.status_code != 200:
        body = r.text[:500].strip()
        hint = ""
        if r.status_code in (401, 403) or "api key" in body.lower() or "apikey" in body.lower():
            hint = " The API key looks invalid or not authorised for this dataset."
        elif "area" in body.lower() or "too large" in body.lower() or "exceed" in body.lower():
            hint = " The bbox is too large for this demtype; shrink the AOI or the buffer."
        elif r.status_code == 204:
            hint = " No data for this area."
        raise common.PipelineError(
            f"OpenTopography returned HTTP {r.status_code}.{hint} Response: {body}")

    tmp = out.with_suffix(".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    head = tmp.read_bytes()[:4]
    if head not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        txt = tmp.read_bytes()[:500].decode("utf-8", "replace")
        tmp.unlink()
        raise common.PipelineError(f"OpenTopography did not return a GeoTIFF: {txt}")
    os.replace(tmp, out)
    dt = time.time() - t0
    log.info("Downloaded %.1f MB in %.1f s -> %s", out.stat().st_size / 1e6, dt, out)
    return {"request_bbox_lonlat": [w, s, e, n], "demtype": params["demtype"],
            "request_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "download_seconds": round(dt, 1), "endpoint": API_URL}


def main(argv=None) -> int:
    args, cfg, paths, log = common.stage_context(__doc__, "01", argv)
    common.require_aoi(cfg)
    out = dem_cache_path(cfg)
    hs_png = paths.checks / "01_dem_raw_hillshade.png"
    side = out.with_suffix(".request.json")

    if not args.force and common.stage_up_to_date(paths, cfg, "01"):
        log.info("Stage 01 up to date - skipping.")
        return 0

    if out.exists():
        log.info("DEM cache hit: %s (download skipped)", out)
        req = common.read_json(side) if side.exists() else {"note": "cached file, request log missing"}
    else:
        req = download(cfg, out, log)
        common.write_json(side, req)

    import numpy as np
    import rasterio

    with rasterio.open(out) as ds:
        z = ds.read(1, masked=True).astype("float64")
        crs = ds.crs.to_string() if ds.crs else "unknown"
        res = ds.res
        b = ds.bounds
    nodata = int(np.ma.count_masked(z))
    zf = z.filled(np.nan)
    lat = 0.5 * (b.top + b.bottom)
    if crs.upper() in ("EPSG:4326", "OGC:CRS84") or abs(res[0]) < 0.1:
        dx = res[0] * 111320.0 * np.cos(np.radians(lat))
        dy = res[1] * 110540.0
    else:
        dx, dy = res
    fill = np.where(np.isnan(zf), np.nanmin(zf) if np.isfinite(zf).any() else 0, zf)
    common.save_gray_png(common.hillshade(fill, dx, dy), hs_png)

    summary = [
        f"Source: {dem_source(cfg)} - {DEM_NOTES.get(dem_source(cfg), 'see OpenTopography docs')}",
        f"File: {common.rel(paths, out)} ({out.stat().st_size / 1e6:.1f} MB)",
        f"Raster: {z.shape[1]} x {z.shape[0]} cells, CRS {crs}, res {res[0]:.6g} x {res[1]:.6g}",
        f"Elevation (raw): {np.nanmin(zf):.2f} .. {np.nanmax(zf):.2f} m, nodata cells: {nodata}",
        f"Request: {req.get('request_bbox_lonlat', 'cached')}",
    ]
    common.finish_stage(paths, cfg, "01", summary, [out], [hs_png], log,
                        extra={"dem_raw": common.rel(paths, out), "request": req})
    return 0


if __name__ == "__main__":
    common.run_main(main, "01")
