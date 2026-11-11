"""Stage 03 - preprocess.

Reprojects the raw DEM and imagery to crs_epsg, clips both to one shared
extent (buffered AOI snapped to the DEM grid), fills DEM nodata by
interpolation, computes the local origin and writes:

  data/processed/dem_utm.tif   float32, metres, cell-centre samples
  data/processed/dem_utm.npy   same array for Blender (row 0 = north)
  data/processed/ortho_utm.tif RGB, finer resolution, same extent
  data/processed/ortho_utm.jpg same, for Rhino
  data/processed/meta.json     (also copied to terrain/00_base/vNNN/meta.json)

No smoothing, benches or erosion: the only change to measured values is the
configured reprojection resampling and the nodata fill, both logged.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MAX_ORTHO_PX = 16384
VERTICAL_DATUM = {
    "COP30": "EGM2008 geoid (orthometric heights)",
    "COP90": "EGM2008 geoid (orthometric heights)",
    "SRTMGL1": "EGM96 geoid", "SRTMGL3": "EGM96 geoid", "NASADEM": "EGM96 geoid",
    "AW3D30": "EGM96 geoid",
}
CAPTURE_NOTE = {
    "COP30": "Copernicus GLO-30 is derived from TanDEM-X acquisitions made 2011-2015.",
    "COP90": "Copernicus GLO-90 is derived from TanDEM-X acquisitions made 2011-2015.",
    "SRTMGL1": "SRTM was flown in February 2000.",
    "SRTMGL3": "SRTM was flown in February 2000.",
    "NASADEM": "NASADEM is reprocessed SRTM data from February 2000.",
    "AW3D30": "ALOS AW3D30 uses PRISM imagery acquired 2006-2011.",
}


def _resampling(name: str):
    from rasterio.enums import Resampling
    try:
        return getattr(Resampling, name)
    except AttributeError:
        raise common.PipelineError(f"Unknown resampling method {name!r}") from None


def _native_res_m(ds) -> float:
    """Approximate native pixel size of a dataset in metres."""
    import math as m
    b = ds.bounds
    if ds.crs is None:
        raise common.PipelineError(f"{ds.name} has no CRS.")
    if ds.crs.is_geographic:
        lat = 0.5 * (b.top + b.bottom)
        return min(abs(ds.res[0]) * 111320 * m.cos(m.radians(lat)), abs(ds.res[1]) * 110540)
    if ds.crs.to_epsg() == 3857:
        from pyproj import Transformer
        lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(
            0.5 * (b.left + b.right), 0.5 * (b.top + b.bottom))[1]
        return abs(ds.res[0]) * m.cos(m.radians(lat))
    return abs(ds.res[0]) * (ds.crs.linear_units_factor[1] if ds.crs.linear_units_factor else 1)


def stage_input(paths, key: str, stage: str):
    rec = paths.stage_record(stage)
    if not rec.exists():
        raise common.PipelineError(f"Stage {stage} has not run for {paths.run_id}; run it first.")
    v = common.read_json(rec).get(key)
    return (paths.root / v) if v else None, common.read_json(rec)


def main(argv=None) -> int:
    import numpy as np
    import rasterio
    from rasterio.fill import fillnodata
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
    from pyproj import Transformer

    args, cfg, paths, log = common.stage_context(__doc__, "03", argv)
    if not args.force and common.stage_up_to_date(paths, cfg, "03"):
        log.info("Stage 03 up to date - skipping.")
        return 0

    crs = common.crs_string(cfg)
    r = common.target_resolution(cfg)
    dcfg = cfg.get("dem") or {}
    icfg = cfg.get("imagery") or {}
    dem_method = str(dcfg.get("resampling", "bilinear"))
    ortho_method = str(icfg.get("resampling", "bilinear"))
    source = str(dcfg.get("source", "COP30"))

    dem_raw, _ = stage_input(paths, "dem_raw", "01")
    ortho_raw, rec02 = stage_input(paths, "ortho_raw", "02")
    imagery_info = rec02.get("imagery", {})

    xmin, ymin, xmax, ymax = common.target_extent(cfg)
    W, H = int(round((xmax - xmin) / r)), int(round((ymax - ymin) / r))
    transform = from_origin(xmin, ymax, r, r)
    log.info("Target grid %s: %d x %d cells at %.3f m, extent %.1f %.1f %.1f %.1f",
             crs, W, H, r, xmin, ymin, xmax, ymax)

    # --- DEM -----------------------------------------------------------
    with rasterio.open(dem_raw) as src:
        native_m = _native_res_m(src)
        src_desc = f"{src.crs.to_string()}, {src.res[0]:.6g} x {src.res[1]:.6g}"
        dem = np.full((H, W), np.nan, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=dem,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=_resampling(dem_method))
    log.info("DEM reprojected from %s (~%.1f m native) to %s at %.1f m using %s",
             src_desc, native_m, crs, r, dem_method)
    if r < 0.9 * native_m:
        log.warning("Target resolution %.1f m is finer than native ~%.1f m: the DEM is "
                    "UPSAMPLED (interpolated, no new detail).", r, native_m)

    valid = np.isfinite(dem)
    n_fill = int((~valid).sum())
    if valid.sum() == 0:
        raise common.PipelineError("The reprojected DEM has no valid cells - check AOI and DEM source.")
    if n_fill:
        filled = fillnodata(np.where(valid, dem, 0).astype("float32"),
                            mask=valid.astype("uint8"), max_search_distance=max(W, H))
        dem = np.where(valid, dem, filled).astype("float32")
        log.info("Filled %d nodata cells (%.3f%%) by inverse-distance interpolation "
                 "(rasterio.fill.fillnodata).", n_fill, 100.0 * n_fill / dem.size)
    else:
        log.info("No nodata cells to fill.")
    if not np.isfinite(dem).all():
        raise common.PipelineError("DEM still has nodata after filling.")

    zmin, zmax = float(dem.min()), float(dem.max())
    paths.processed.mkdir(parents=True, exist_ok=True)
    dem_tif = paths.processed / "dem_utm.tif"
    dem_npy = paths.processed / "dem_utm.npy"
    with rasterio.open(dem_tif, "w", driver="GTiff", width=W, height=H, count=1,
                       dtype="float32", crs=crs, transform=transform, compress="deflate",
                       predictor=3) as ds:
        ds.write(dem, 1)
        ds.update_tags(AREA_OR_POINT="Area", units="m", source=source)
    np.save(dem_npy, dem)

    # --- Imagery -------------------------------------------------------
    mode = icfg.get("mode", "esri_tiles")
    ortho_tif = paths.processed / "ortho_utm.tif"
    ortho_jpg = paths.processed / "ortho_utm.jpg"
    hs = common.hillshade(dem, r, r)
    if ortho_raw is None or mode == "none":
        k = 1
        ortho = np.repeat((hs * 200 + 30).astype("uint8")[None], 3, axis=0)
        ortho_native_m = None
        empty = 0
        log.info("No imagery: writing a neutral hillshade texture at DEM resolution.")
    else:
        with rasterio.open(ortho_raw) as src:
            ortho_native_m = _native_res_m(src)
            k = max(1, math.ceil(r / ortho_native_m))
            if max(W, H) * k > MAX_ORTHO_PX:
                k_cap = max(1, MAX_ORTHO_PX // max(W, H))
                log.warning("Ortho would be %d px wide; capping to %d px (pixel %.2f m instead "
                            "of %.2f m).", max(W, H) * k, max(W, H) * k_cap, r / k_cap, r / k)
                k = k_cap
            bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
            ortho = np.zeros((3, H * k, W * k), dtype="uint8")
            for i, b in enumerate(bands):
                reproject(source=rasterio.band(src, b), destination=ortho[i],
                          src_transform=src.transform, src_crs=src.crs, src_nodata=None,
                          dst_transform=from_origin(xmin, ymax, r / k, r / k), dst_crs=crs,
                          dst_nodata=0, resampling=_resampling(ortho_method))
        empty = int((ortho.max(axis=0) == 0).sum())
        log.info("Ortho reprojected from ~%.2f m native to %.3f m (DEM cell / %d) using %s; "
                 "%dx%d px", ortho_native_m, r / k, k, ortho_method, W * k, H * k)
        if empty > 0.001 * ortho[0].size:
            log.warning("%d ortho pixels (%.2f%%) are empty (outside the imagery).",
                        empty, 100.0 * empty / ortho[0].size)
    with rasterio.open(ortho_tif, "w", driver="GTiff", width=W * k, height=H * k, count=3,
                       dtype="uint8", crs=crs, transform=from_origin(xmin, ymax, r / k, r / k),
                       compress="deflate", photometric="RGB") as ds:
        ds.write(ortho)
    from PIL import Image
    Image.fromarray(ortho.transpose(1, 2, 0)).save(ortho_jpg, quality=92)

    # --- Origin & meta -------------------------------------------------
    origin_x = round(xmin / 100.0) * 100.0
    origin_y = round(ymin / 100.0) * 100.0
    olon, olat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(origin_x, origin_y)
    aoi = common.require_aoi(cfg)
    meta = {
        "run_id": paths.run_id,
        "config_hash": common.config_hash(cfg),
        "crs": crs,
        "crs_epsg": int(cfg["crs_epsg"]),
        "units": "metre",
        "z_up": True,
        "extent": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                   "width_m": xmax - xmin, "height_m": ymax - ymin},
        "origin_x": origin_x,
        "origin_y": origin_y,
        "origin_lon": olon,
        "origin_lat": olat,
        "origin_rule": "lower-left corner of the extent, rounded to the nearest 100 m",
        "aoi_wgs84": aoi,
        "grid": {
            "cols": W, "rows": H, "cell_size_m": r,
            "convention": "Each value is the cell-centre elevation. Cell (row 0, col 0) "
                          "centre = (xmin + cell/2, ymax - cell/2). Row 0 is north.",
        },
        "dem": {
            "source": source,
            "raw_file": common.rel(paths, dem_raw),
            "native_resolution_m_approx": round(native_m, 2),
            "native_grid": src_desc,
            "resolution_m": r,
            "vertical_datum": VERTICAL_DATUM.get(source, "see source documentation"),
            "capture_date_note": CAPTURE_NOTE.get(source, "see source documentation"),
            "nodata_cells_filled": n_fill,
            "nodata_fill_method": "rasterio.fill.fillnodata (inverse distance weighting)",
        },
        "elevation": {"min": zmin, "max": zmax, "mean": float(dem.mean()),
                      "range": zmax - zmin},
        "resampling": {
            "dem_method": dem_method,
            "dem_from": f"~{native_m:.1f} m ({src_desc})",
            "dem_to": f"{r} m ({crs})",
            "dem_upsampled": bool(r < 0.9 * native_m),
            "ortho_method": ortho_method if ortho_native_m else None,
            "ortho_from_m": round(ortho_native_m, 3) if ortho_native_m else None,
            "ortho_to_m": r / k,
            "note": "Reprojection from the source grid to the projected CRS always resamples; "
                    "no smoothing or other edits were applied.",
        },
        "imagery": {
            "mode": mode,
            "raw_file": common.rel(paths, ortho_raw) if ortho_raw else None,
            "pixel_size_m": r / k,
            "pixels": [W * k, H * k],
            "empty_pixels": empty,
            "attribution": imagery_info.get("attribution"),
            "terms": imagery_info.get("terms"),
            "zoom": imagery_info.get("zoom"),
        },
        "files": {
            "dem_utm_tif": common.rel(paths, dem_tif),
            "dem_utm_npy": common.rel(paths, dem_npy),
            "ortho_utm_tif": common.rel(paths, ortho_tif),
            "ortho_utm_jpg": common.rel(paths, ortho_jpg),
        },
    }
    meta_proc = paths.processed / "meta.json"
    common.write_json(meta_proc, meta)
    meta_base = paths.base / "meta.json"
    if meta_base.exists():
        old = common.read_json(meta_base)
        if old.get("config_hash") != meta["config_hash"]:
            common.write_once_guard(meta_base)
        log.info("terrain meta.json already exists for this run (same config) - kept.")
    else:
        common.write_json(meta_base, meta)

    # --- Check images --------------------------------------------------
    blend_png = paths.checks / "03_hillshade_over_ortho.png"
    hist_png = paths.checks / "03_elevation_histogram.png"
    img = Image.fromarray(ortho.transpose(1, 2, 0))
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048), Image.LANCZOS)
    hs_img = Image.fromarray((hs * 255).astype("uint8"), "L").resize(img.size, Image.BILINEAR)
    Image.blend(img.convert("RGB"), hs_img.convert("RGB"), 0.5).save(blend_png)
    draw_histogram(dem, hist_png, f"{source} elevation, {r:g} m cells ({W}x{H})")

    summary = [
        f"Grid: {W} x {H} cells at {r:g} m in {crs}; extent {xmax - xmin:.0f} x {ymax - ymin:.0f} m",
        f"Origin (lower-left, nearest 100 m): x={origin_x:.0f}, y={origin_y:.0f} "
        f"(lon {olon:.6f}, lat {olat:.6f})",
        f"Elevation: {zmin:.2f} .. {zmax:.2f} m (range {zmax - zmin:.2f} m)",
        f"DEM resampled with {dem_method} from ~{native_m:.1f} m; nodata cells filled: {n_fill}",
        f"Ortho: {W * k} x {H * k} px at {r / k:.3f} m ({mode})",
    ]
    common.finish_stage(paths, cfg, "03", summary,
                        [dem_tif, dem_npy, ortho_tif, ortho_jpg, meta_proc, meta_base],
                        [blend_png, hist_png], log)
    return 0


def draw_histogram(z, path: Path, title: str, bins: int = 60):
    import numpy as np
    from PIL import Image, ImageDraw

    counts, edges = np.histogram(z, bins=bins)
    Wp, Hp, m = 900, 420, 50
    im = Image.new("RGB", (Wp, Hp), "white")
    d = ImageDraw.Draw(im)
    d.text((m, 12), title, fill="black")
    bw = (Wp - 2 * m) / bins
    top = counts.max() or 1
    for i, c in enumerate(counts):
        h = (Hp - 2 * m) * c / top
        d.rectangle([m + i * bw, Hp - m - h, m + (i + 1) * bw - 1, Hp - m], fill=(70, 110, 160))
    d.line([m, Hp - m, Wp - m, Hp - m], fill="black")
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        x = m + frac * (Wp - 2 * m)
        v = edges[0] + frac * (edges[-1] - edges[0])
        d.line([x, Hp - m, x, Hp - m + 5], fill="black")
        d.text((x - 20, Hp - m + 8), f"{v:.0f} m", fill="black")
    d.text((Wp - m - 140, 12), f"max count {top}", fill="black")
    im.save(path)


if __name__ == "__main__":
    common.run_main(main, "03")
