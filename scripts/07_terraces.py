"""Stage 07 - interpret quarry terraces (benches) from imagery + DEM.

INTERPRETED data. Reads the measured base (data/processed/dem_utm.tif,
ortho_utm.tif, terrain/00_base/vNNN/meta.json) and never modifies it.
Writes to terrain/10_terraces/vNNN/ and exchange/rhino/vNNN/terraced/:

  terraces_dem.tif        interpreted elevation (m) on a fine grid
  measured_upsampled.tif  the measured DEM, cubic-upsampled to the same grid
  dz.tif                  terraces - measured (m)
  classes.tif             0 outside pit, 1 pit floor, 2 bench, 3 face, 4 water
  confidence.tif          0..1, how consistently the imagery supports the levels
  edge_score.tif          image edges running along the DEM contours
  pit_mask.tif, pit_outline.geojson
  terraces_params.json    bench height, levels, face angle, statistics
  quarry_terraced.blend   terraced mesh (+ measured context), same BlenderGIS
                          georef and origin as the base; vertex attributes
                          confidence / terrace_class / dz_interp
  report_terraces.md

Method (see scripts/terraces.py): pit mask from imagery + DEM relief; edges
parallel to DEM contours; bench height from the periodicity of edge strength
over elevation; bench levels from its phase (refined locally); a
volume-preserving staircase built on the smooth DEM surface.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    "enabled": True,
    "resolution_m": 2.0,            # interpreted grid; must divide the DEM cell
    "bench_height_m": "auto",       # or a number (m)
    "bench_height_range_m": [5.0, 25.0],
    "face_angle_deg": 70.0,         # typical hard-rock bench face; not measurable from imagery
    "min_slope_deg": 8.0,           # below this (pit floor, flats) no benches are built
    "surface_smoothing_m": 20.0,    # smoothing of the DEM surface the benches are built on
    "local_window_m": 30.0,         # window for the local DEM correction
    "pit_mask": None,               # optional GeoJSON polygon(s) (lon/lat or project CRS)
    "pit_threshold": None,          # optional fixed threshold for the imagery quarry index
    "pit_min_area_m2": 20000,
    "pit_min_local_relief_m": 3.0,
    "feather_m": 8.0,
    "margin_m": 150.0,              # mesh extent = pit bbox + margin
    "mesh_max_vertices": 3000000,
    "require_significant": True,    # stop (ask) when no bench pattern is found and H is auto
}

CLASS_NAMES = {0: "outside", 1: "pit floor", 2: "bench", 3: "face", 4: "water"}


def tcfg(cfg) -> dict:
    t = dict(DEFAULTS)
    t.update(cfg.get("terraces") or {})
    return t


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def out_dir(paths) -> Path:
    return paths.root / "terrain" / "10_terraces" / paths.run_id


def load_user_mask(path, cfg, transform, shape):
    """Rasterise a user GeoJSON (lon/lat or project CRS) onto the grid."""
    from pyproj import Transformer
    from rasterio import features
    import numpy as np
    import shapely
    from shapely.geometry import shape as shp

    p = Path(path)
    if not p.is_absolute():
        p = Path(cfg["_root"]) / p
    if not p.exists():
        raise common.PipelineError(f"terraces.pit_mask not found: {p}")
    gj = json.loads(p.read_text())
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    geoms = [shp(f["geometry"] if "geometry" in f else f) for f in feats]
    minx = min(g.bounds[0] for g in geoms)
    if abs(minx) <= 180:  # lon/lat
        tr = Transformer.from_crs("EPSG:4326", common.crs_string(cfg), always_xy=True)
        geoms = [shapely.transform(g, lambda c: np.c_[tr.transform(c[:, 0], c[:, 1])]) for g in geoms]
    mask = features.rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                              dtype="uint8").astype(bool)
    return mask, geoms


def main(argv=None) -> int:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
    from pyproj import Transformer
    import shapely
    from shapely.geometry import box, mapping

    import terraces as T

    args, cfg, paths, log = common.stage_context(__doc__, "07", argv)
    tc = tcfg(cfg)
    if not args.force and common.stage_up_to_date(paths, cfg, "07"):
        log.info("Stage 07 up to date - skipping.")
        return 0
    if not tc["enabled"]:
        common.finish_stage(paths, cfg, "07", ["Terrace interpretation disabled (terraces.enabled: false)."],
                            [], [], log)
        return 0

    meta = common.read_json(paths.base / "meta.json")
    proc_meta = common.read_json(paths.processed / "meta.json")
    if proc_meta.get("config_hash") != meta.get("config_hash"):
        raise common.PipelineError("data/processed belongs to a different config; re-run stage 03.")
    dem_tif = paths.processed / "dem_utm.tif"
    ortho_tif = paths.processed / "ortho_utm.tif"
    base_blend = paths.base / "quarry_base.blend"
    base_hashes = {"dem_utm.tif": sha256(dem_tif)}
    if base_blend.exists():
        base_hashes["quarry_base.blend"] = sha256(base_blend)

    od = out_dir(paths)
    od.mkdir(parents=True, exist_ok=True)

    # ---------------- grid -------------------------------------------------
    e = meta["extent"]
    r = float(meta["grid"]["cell_size_m"])
    k = max(1, int(round(r / float(tc["resolution_m"]))))
    res = r / k
    W, H = meta["grid"]["cols"] * k, meta["grid"]["rows"] * k
    tf = from_origin(e["xmin"], e["ymax"], res, res)
    crs = meta["crs"]
    log.info("Interpretation grid: %d x %d at %.3f m (DEM cell / %d)", W, H, res, k)

    z0 = np.zeros((H, W), "float32")
    with rasterio.open(dem_tif) as src:
        reproject(rasterio.band(src, 1), z0, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=tf, dst_crs=crs, resampling=Resampling.cubic)
    z0 = z0.astype("float64")
    rgb = np.zeros((3, H, W), "uint8")
    with rasterio.open(ortho_tif) as src:
        rs = Resampling.average if src.res[0] < res else Resampling.bilinear
        for i in range(3):
            reproject(rasterio.band(src, i + 1), rgb[i], src_transform=src.transform,
                      src_crs=src.crs, dst_transform=tf, dst_crs=crs, resampling=rs)
    rgb = rgb.transpose(1, 2, 0)
    if meta["imagery"]["mode"] == "none":
        raise common.PipelineError("Terrace interpretation needs imagery (imagery.mode is 'none').")
    lum = rgb.astype("float64").mean(-1) / 255.0

    zs = T.gauss(z0, float(tc["surface_smoothing_m"]) / res)
    slope = T.overall_slope_deg(zs, res)

    # ---------------- 1. pit mask -------------------------------------------
    aoi = common.require_aoi(cfg)
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    aoi_poly = shapely.transform(box(aoi["west"], aoi["south"], aoi["east"], aoi["north"]),
                                 lambda c: np.c_[to_crs.transform(c[:, 0], c[:, 1])])
    if tc["pit_mask"]:
        mask, polys = load_user_mask(tc["pit_mask"], cfg, tf, (H, W))
        mask_info = {"source": f"user polygon {tc['pit_mask']}", "area_m2": float(mask.sum() * res * res)}
    else:
        mask, polys, mask_info = T.detect_pit(
            rgb, slope, z0, tf, res, threshold=tc["pit_threshold"],
            min_area_m2=float(tc["pit_min_area_m2"]),
            min_local_relief_m=float(tc["pit_min_local_relief_m"]), keep_polygon=aoi_poly)
        mask_info["source"] = "imagery quarry index + DEM relief (automatic)"
    if mask.sum() == 0:
        raise common.PipelineError(
            "No quarry pit was found in the imagery. Draw the pit outline as a GeoJSON polygon "
            "and set terraces.pit_mask, or set terraces.pit_threshold.")
    log.info("Pit mask: %.1f ha (%s)", mask_info["area_m2"] / 1e4, mask_info["source"])

    feather = T.smoothstep(T.gauss(mask.astype(float), float(tc["feather_m"]) / res), 0.5, 0.95)
    min_slope = float(tc["min_slope_deg"])
    zone = mask & (slope >= min_slope)

    # ---------------- 2. edges ----------------------------------------------
    E = T.contour_edge_score(lum, zs, res)
    cap = np.percentile(E[zone], 99.5) if zone.any() else 1.0
    En = np.clip(E / (cap + 1e-12), 0, 1)
    w = np.where(zone, En ** 2, 0.0)

    # ---------------- 3. bench height & levels --------------------------------
    zl = T.gauss(z0, 1.0)
    hmin, hmax = (float(v) for v in tc["bench_height_range_m"])
    est = T.estimate_bench_height(zl[zone], w[zone], np.ones(int(zone.sum())), hmin, hmax)
    auto = str(tc["bench_height_m"]).lower() == "auto"
    if auto:
        Hb = est["H"]
    else:
        Hb = float(tc["bench_height_m"])
        est_fixed = T.estimate_bench_height(zl[zone], w[zone], np.ones(int(zone.sum())), Hb, Hb)
        est["edge_level_m"] = est_fixed["edge_level_m"]
    # significance: real score vs scores after shifting the edge map against the DEM
    nulls = T.null_scores(zl, w, zone, hmin, hmax)
    null_max = max(nulls) if nulls else 0.0
    significance = est["score"] / max(null_max, 1e-6)
    weak = significance < 1.5 or est["score"] < 0.05
    log.info("Bench signal score %.3f vs best of %d shifted (null) maps %.3f -> x%.1f",
             est["score"], len(nulls), null_max, significance)
    if weak:
        msg = (f"No consistent bench pattern in the imagery (score {est['score']:.3f} vs "
               f"{null_max:.3f} for shifted maps, x{significance:.1f}; need x1.5).")
        if auto and tc.get("require_significant", True):
            raise common.PipelineError(
                msg + " Benches are not visible enough to calculate them. If you know the bench "
                "height, set terraces.bench_height_m (levels are then still taken from the "
                "imagery), check the pit mask (terraces.pit_mask), or set terraces.enabled: false.")
        log.warning(msg + " Continuing because the bench height was given / require_significant is off; "
                    "the geometry is LOW CONFIDENCE.")
    log.info("Bench height %.2f m (%s); coherence R=%.3f vs baseline %.3f; edge level %.2f m mod H",
             Hb, "estimated" if auto else "from config", est["R"], est["R_baseline"],
             est["edge_level_m"])
    # Benches are mined at fixed elevations, so the bench levels stay GLOBAL
    # (G + n*H). Where the imagery shows the edges off those levels, the DEM is
    # locally biased: that offset (wrapped to +-H/2) corrects the surface.
    G = est["edge_level_m"]
    level, conf = T.local_levels(zl, w, Hb, float(tc["local_window_m"]) / res, G)
    delta = (level - G + Hb / 2) % Hb - Hb / 2

    # ---------------- 4. staircase --------------------------------------------
    alpha = math.radians(float(tc["face_angle_deg"]))
    phi = np.clip(np.tan(np.radians(slope)) / math.tan(alpha), 0.05, 0.95)
    # steps are built on the measured surface itself (lightly filtered) so the
    # result is volume-neutral; the smoothed surface only gives slope/direction
    z_st, face, bench_n = T.staircase(zl - delta, G, Hb, phi)
    wb = feather * T.smoothstep(slope, min_slope, min_slope + 4.0)
    zt = z0 + wb * (z_st - z0)

    # water (dark, bluish ponds inside the pit) -> flat surface
    water = mask & T.water_index(rgb)
    water = T.box_blur(water.astype(float), max(1, int(round(3 / res)))) > 0.5
    labels, wpolys = T.label_components(water, tf)
    n_water = 0
    for i, p in enumerate(wpolys):
        if p.area < 500:
            continue
        sel = labels == i + 1
        zt[sel] = np.percentile(z0[sel], 10)
        n_water += 1

    classes = np.zeros((H, W), "uint8")
    classes[mask] = 1
    classes[mask & (wb > 0.5) & ~face] = 2
    classes[mask & (wb > 0.5) & face] = 3
    if n_water:
        classes[mask & (labels > 0)] = 4
    dz = zt - z0
    confidence = np.where(mask, conf, 0.0)

    # ---------------- work extent for the mesh ---------------------------------
    rows_any = np.where(mask.any(1))[0]
    cols_any = np.where(mask.any(0))[0]
    m_px = int(round(float(tc["margin_m"]) / res))
    r0, r1 = max(0, rows_any[0] - m_px), min(H, rows_any[-1] + 1 + m_px)
    c0, c1 = max(0, cols_any[0] - m_px), min(W, cols_any[-1] + 1 + m_px)
    # align to DEM cells so the context mesh hole lines up
    r0, c0 = (r0 // k) * k, (c0 // k) * k
    r1, c1 = min(H, -(-r1 // k) * k), min(W, -(-c1 // k) * k)
    nverts = (r1 - r0) * (c1 - c0)
    stride = max(1, int(math.ceil(math.sqrt(nverts / float(tc["mesh_max_vertices"])))))
    if stride > 1:
        log.warning("Mesh would have %d vertices; using every %d-th sample (limit "
                    "terraces.mesh_max_vertices).", nverts, stride)

    # ---------------- write rasters --------------------------------------------
    def wr(name, arr, dtype, nodata=None):
        p = od / name
        if p.exists():
            return p
        with rasterio.open(p, "w", driver="GTiff", width=W, height=H, count=1, dtype=dtype,
                           crs=crs, transform=tf, compress="deflate", nodata=nodata) as ds:
            ds.write(arr.astype(dtype), 1)
            ds.update_tags(interpreted="true", run_id=paths.run_id)
        return p

    files = [wr("terraces_dem.tif", zt, "float32"), wr("measured_upsampled.tif", z0, "float32"),
             wr("dz.tif", dz, "float32"), wr("classes.tif", classes, "uint8"),
             wr("confidence.tif", confidence, "float32"), wr("edge_score.tif", En, "float32"),
             wr("pit_mask.tif", mask, "uint8"),
             wr("surface_correction.tif", np.where(wb > 0, -delta * wb, 0.0), "float32")]
    ox, oy = meta["origin_x"], meta["origin_y"]
    gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}},
          "features": [{"type": "Feature", "properties": {"id": i, "area_m2": round(p.area)},
                        "geometry": mapping(p)} for i, p in enumerate(polys)]}
    outline = od / "pit_outline.geojson"
    if not outline.exists():
        outline.write_text(json.dumps(gj))
    files.append(outline)

    patch = od / "blender_patch.npz"
    sl = (slice(r0, r1, stride), slice(c0, c1, stride))
    if not patch.exists():
        np.savez(patch, z=zt[sl].astype("float32"), confidence=confidence[sl].astype("float32"),
                 classes=classes[sl].astype("int32"), dz=dz[sl].astype("float32"))
    zone_dz = dz[mask & (wb > 0.5)]
    zpit = zt[mask]
    n0 = int(math.floor((float(zpit.min()) - G) / Hb - 0.5))
    n1 = int(math.ceil((float(zpit.max()) - G) / Hb - 0.5))
    floors = [round(G + (n + 0.5) * Hb, 2) for n in range(n0, n1 + 1)
              if zpit.min() - Hb / 2 <= G + (n + 0.5) * Hb <= zpit.max() + Hb / 2]
    params = {
        "run_id": paths.run_id,
        "config_hash": common.config_hash(cfg),
        "interpreted": True,
        "crs": crs, "origin_x": ox, "origin_y": oy,
        "grid": {"resolution_m": res, "cols": W, "rows": H, "transform": list(tf)[:6]},
        "patch": {"row0": r0, "row1": r1, "col0": c0, "col1": c1, "stride": stride,
                  "xmin": e["xmin"] + c0 * res, "xmax": e["xmin"] + c1 * res,
                  "ymax": e["ymax"] - r0 * res, "ymin": e["ymax"] - r1 * res},
        "bench_height_m": Hb,
        "bench_height_source": "estimated from imagery+DEM" if auto else "config",
        "bench_level_mod_H_m": G,
        "bench_floor_levels_m": floors,
        "bench_level_note": ("Bench floors are flat at G + (n + 0.5) * H; face centres at G + n * H. "
                             "Levels are global (fixed elevations); the DEM surface is shifted locally "
                             "by up to +-H/2 to meet them (surface_correction.tif)."),
        "surface_correction_m": {"mean": float(-(delta * wb)[zone].mean()) if zone.any() else 0.0,
                                 "abs_mean": float(np.abs(delta * wb)[zone].mean()) if zone.any() else 0.0},
        "coherence_R": est["R"], "coherence_baseline": est["R_baseline"],
        "signal_score": est["score"], "null_scores": nulls,
        "coherence_significance": significance, "low_confidence": bool(weak),
        "face_angle_deg": float(tc["face_angle_deg"]),
        "face_angle_source": "config (not measurable from imagery)",
        "mean_overall_slope_deg": float(slope[zone].mean()) if zone.any() else None,
        "implied_bench_width_m_mean": float(np.mean(
            Hb / np.tan(np.radians(np.clip(slope[zone], 1, 89))) - Hb / math.tan(alpha)))
        if zone.any() else None,
        "pit": {k2: v for k2, v in mask_info.items() if k2 != "rejected"},
        "pit_rejected_candidates": mask_info.get("rejected", [])[:20],
        "water_bodies": n_water,
        "class_cells": {CLASS_NAMES[c]: int((classes == c).sum()) for c in CLASS_NAMES},
        "dz_in_bench_zone": {"mean": float(zone_dz.mean()) if zone_dz.size else 0.0,
                             "std": float(zone_dz.std()) if zone_dz.size else 0.0,
                             "min": float(zone_dz.min()) if zone_dz.size else 0.0,
                             "max": float(zone_dz.max()) if zone_dz.size else 0.0},
        "mean_confidence_in_pit": float(conf[mask].mean()),
        "settings": {k2: tc[k2] for k2 in DEFAULTS},
        "base_hashes_before": base_hashes,
        "spectrum": {"h_values": est["h_values"].round(3).tolist(),
                     "R_edges": est["R_edges"].round(5).tolist(),
                     "R_baseline": est["R_base"].round(5).tolist()},
    }
    params_p = od / "terraces_params.json"
    common.write_json(params_p, params)
    files.append(params_p)

    # ---------------- check images ------------------------------------------
    imgs = check_images(paths, od, rgb, mask, En, z0, zt, zs, classes, confidence, est, Hb,
                        res, (r0, r1, c0, c1), log)

    # ---------------- Blender ------------------------------------------------
    blender, _ = common.find_blender(cfg)
    if not blender:
        raise common.PipelineError("Blender path is not set (config blender.path or BLENDER_PATH).")
    rhino_t = paths.rhino / "terraced"
    job = {
        "run_id": paths.run_id, "log_file": str(paths.log_file),
        "meta": str(paths.base / "meta.json"), "params": str(params_p),
        "dem_npy": str(paths.processed / "dem_utm.npy"), "patch": str(patch),
        "outline": str(outline), "ortho_tif": str(ortho_tif),
        "ortho_jpg": str(paths.processed / "ortho_utm.jpg"),
        "blend": str(od / "quarry_terraced.blend"), "textures_dir": str(od / "textures"),
        "rhino_dir": str(rhino_t), "checks_dir": str(paths.checks),
        "formats": [f.lower() for f in ((cfg.get("rhino") or {}).get("formats") or ["obj"])],
        "result": str(paths.checks / "blender_07_result.json"),
    }
    job_path = paths.checks / "blender_job_07.json"
    common.write_json(job_path, job)
    b04 = _load("build_blender", "04_build_blender.py")
    rc = b04.run_blender(blender, HERE / "blender_helpers" / "build_terraces.py", job_path, log)
    if rc != 0 or not Path(job["result"]).exists():
        raise common.PipelineError(f"Blender terrace build failed (exit {rc}); see {paths.log_file}")
    br = common.read_json(job["result"])
    if br.get("error"):
        raise common.PipelineError("Blender terrace build failed: " + br["error"])

    # ---------------- checks --------------------------------------------------
    checks = []
    after = {"dem_utm.tif": sha256(dem_tif)}
    if base_blend.exists():
        after["quarry_base.blend"] = sha256(base_blend)
    checks.append(("Measured base untouched (hashes before = after)", after == base_hashes,
                   ", ".join(f"{k2} {v[:12]}" for k2, v in after.items())))
    out_zone = feather == 0
    d_out = float(np.abs(dz[out_zone]).max()) if out_zone.any() else 0.0
    checks.append(("Outside the pit, interpreted = measured", d_out == 0.0, f"max |dz| outside {d_out}"))
    mdz = params["dz_in_bench_zone"]["mean"]
    checks.append(("No systematic shift vs measured (mean dz within 25% of H)", abs(mdz) <= 0.25 * Hb,
                   f"mean dz {mdz:+.2f} m, std {params['dz_in_bench_zone']['std']:.2f} m, H {Hb:.2f} m"))
    g = br["georef"]
    ok_geo = (g.get("SRID") == meta["crs"] and abs(g.get("crs x", 1e18) - ox) < 1e-6
              and abs(g.get("crs y", 1e18) - oy) < 1e-6)
    checks.append(("Terrace .blend georef equals base meta.json", ok_geo,
                   f"SRID={g.get('SRID')}, crs x={g.get('crs x')}, crs y={g.get('crs y')}"))
    pz = np.load(patch)["z"]
    dzr = max(abs(br["patch_bbox"][2] - float(pz.min())), abs(br["patch_bbox"][5] - float(pz.max())))
    checks.append(("Terraced mesh Z range equals interpreted raster", dzr <= 0.01, f"max diff {dzr:.4f} m"))
    pb = params["patch"]
    half = res * stride / 2 + 1e-3
    exp = [pb["xmin"] - ox + half, pb["ymin"] - oy + half, pb["xmax"] - ox - half, pb["ymax"] - oy - half]
    got = [br["patch_bbox"][0], br["patch_bbox"][1], br["patch_bbox"][3], br["patch_bbox"][4]]
    dxy = max(abs(a - b) for a, b in zip(exp, got))
    checks.append(("Terraced mesh sits at the right place (bbox)", dxy <= res * stride,
                   f"max diff {dxy:.3f} m"))
    for name, ok, det in checks:
        log.info("%s: %s - %s", "PASS" if ok else "FAIL", name, det)

    report = od / "report_terraces.md"
    report.write_text(make_report(paths, meta, params, checks, br, imgs, files))
    summary = [
        f"Pit: {mask_info['area_m2'] / 1e4:.1f} ha ({mask_info['source']}); water bodies {n_water}",
        f"Bench floors at {', '.join(f'{v:.1f}' for v in floors)} m",
        f"Bench height {Hb:.2f} m ({params['bench_height_source']}), coherence {est['R']:.3f} "
        f"(baseline {est['R_baseline']:.3f}){' - LOW CONFIDENCE' if weak else ''}",
        f"Face angle {tc['face_angle_deg']} deg (config); mean overall pit slope "
        f"{params['mean_overall_slope_deg'] or 0:.1f} deg; implied bench width "
        f"{params['implied_bench_width_m_mean'] or 0:.1f} m",
        f"dz in bench zone: mean {mdz:+.2f} m, range {params['dz_in_bench_zone']['min']:.1f}.."
        f"{params['dz_in_bench_zone']['max']:.1f} m; mean confidence {params['mean_confidence_in_pit']:.2f}",
        f"Blender: {common.rel(paths, od / 'quarry_terraced.blend')} ({br['patch_verts']} vertices)",
    ] + [f"{'PASS' if ok else 'FAIL'}: {n} - {d}" for n, ok, d in checks]
    if not all(ok for _, ok, _ in checks):
        for s_ in summary:
            log.error(s_)
        raise common.PipelineError("Terrace checks failed: " + "; ".join(n for n, ok, _ in checks if not ok))
    outs = files + [od / "quarry_terraced.blend", report] + [paths.root / p for p in br["exports"]]
    common.finish_stage(paths, cfg, "07", summary, outs, imgs + [paths.root / p for p in br["check_images"]],
                        log, extra={"terraces": {k2: params[k2] for k2 in
                                                 ("bench_height_m", "coherence_R", "low_confidence")}})
    return 0


def _load(name, file):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_images(paths, od, rgb, mask, En, z0, zt, zs, classes, conf, est, Hb, res, win, log):
    import numpy as np
    from PIL import Image, ImageDraw

    import terraces as T

    r0, r1, c0, c1 = win
    out = []
    crop = (slice(r0, r1), slice(c0, c1))
    base = Image.fromarray(rgb[crop])

    def fit(im, side=900):
        im = im.copy()
        im.thumbnail((side, side), Image.LANCZOS)
        return im

    # 1. pit mask + classes over the imagery
    pal = np.array([[0, 0, 0], [80, 160, 255], [255, 210, 60], [230, 60, 40], [40, 90, 255]], "uint8")
    cl = classes[crop]
    over = rgb[crop].astype(float)
    m = cl > 0
    over[m] = over[m] * 0.55 + pal[cl[m]] * 0.45
    p1 = paths.checks / "07_pit_mask_and_classes.png"
    fit(Image.fromarray(over.astype("uint8"))).save(p1)
    out.append(p1)

    # 2. contour-aligned edges
    p2 = paths.checks / "07_contour_edges.png"
    ed = (np.clip(En[crop], 0, 1) ** 0.5 * 255).astype("uint8")
    two = Image.new("RGB", (2 * base.width + 10, base.height), "white")
    two.paste(base, (0, 0))
    two.paste(Image.fromarray(ed).convert("RGB"), (base.width + 10, 0))
    fit(two, 1600).save(p2)
    out.append(p2)

    # 3. bench-height spectrum
    p3 = paths.checks / "07_bench_height_spectrum.png"
    hs, Re, Rb = est["h_values"], est["R_edges"], est["R_base"]
    Wp, Hp, mg = 900, 380, 50
    im = Image.new("RGB", (Wp, Hp), "white")
    d = ImageDraw.Draw(im)
    top = max(Re.max(), Rb.max(), 1e-6)

    def pt(h, v):
        return (mg + (h - hs[0]) / (hs[-1] - hs[0]) * (Wp - 2 * mg), Hp - mg - v / top * (Hp - 2 * mg))
    d.line([pt(h, v) for h, v in zip(hs, Rb)], fill=(160, 160, 160), width=2)
    d.line([pt(h, v) for h, v in zip(hs, Re)], fill=(200, 50, 40), width=2)
    x, _ = pt(Hb, 0)
    d.line([x, mg, x, Hp - mg], fill=(40, 90, 200), width=1)
    d.text((x + 4, mg), f"H = {Hb:.2f} m", fill=(40, 90, 200))
    for hv in range(int(math.ceil(hs[0])), int(hs[-1]) + 1, 5):
        xx, _ = pt(hv, 0)
        d.text((xx - 6, Hp - mg + 6), str(hv), fill="black")
    d.text((mg, 10), "edge coherence over elevation (red) vs baseline (grey), by bench height H [m]",
           fill="black")
    im.save(p3)
    out.append(p3)

    # 4. hillshade measured vs interpreted
    p4 = paths.checks / "07_hillshade_measured_vs_terraced.png"
    a = Image.fromarray((common.hillshade(z0[crop], res, res) * 255).astype("uint8"))
    b = Image.fromarray((common.hillshade(zt[crop], res, res) * 255).astype("uint8"))
    two = Image.new("L", (2 * a.width + 10, a.height), 255)
    two.paste(a, (0, 0))
    two.paste(b, (a.width + 10, 0))
    fit(two, 1600).save(p4)
    out.append(p4)

    # 5. profile through the pit centre (W-E)
    p5 = paths.checks / "07_profile.png"
    rows = np.where(mask.any(1))[0]
    rc = int(np.median(rows))
    cols = np.arange(c0, c1)
    prof = [(z0[rc, cols], (90, 90, 90)), (zs[rc, cols], (170, 170, 220)), (zt[rc, cols], (200, 50, 40))]
    lo = min(p.min() for p, _ in prof)
    hi = max(p.max() for p, _ in prof)
    Wp, Hp = 1200, 420
    im = Image.new("RGB", (Wp, Hp), "white")
    d = ImageDraw.Draw(im)
    for arr, col in prof:
        xs = mg + np.arange(len(arr)) / max(1, len(arr) - 1) * (Wp - 2 * mg)
        ys = Hp - mg - (arr - lo) / max(hi - lo, 1e-6) * (Hp - 2 * mg)
        d.line(list(zip(xs, ys)), fill=col, width=2)
    d.text((mg, 10), f"W-E profile through the pit: measured (grey), smooth surface (blue), "
                     f"interpreted terraces (red). Vertical {lo:.0f}..{hi:.0f} m, "
                     f"horizontal {len(cols) * res:.0f} m (exaggerated)", fill="black")
    im.save(p5)
    out.append(p5)

    # 6. confidence
    p6 = paths.checks / "07_confidence.png"
    c = (np.clip(conf[crop], 0, 1) * 255).astype("uint8")
    fit(Image.fromarray(np.stack([255 - c, c, np.zeros_like(c)], -1))).save(p6)
    out.append(p6)
    _ = T  # keep import for symmetry
    return out


def make_report(paths, meta, p, checks, br, imgs, files) -> str:
    L = [
        f"# Terrace interpretation report - run {paths.run_id}",
        "",
        "> **Interpreted data.** These benches are inferred from satellite imagery and the 30 m "
        "DEM. They are a plausible, imagery-consistent reconstruction, not a survey. The measured "
        "base terrain in `terrain/00_base/` is unchanged.",
        "",
        "## Result",
        f"- Bench height: **{p['bench_height_m']:.2f} m** ({p['bench_height_source']}); coherence "
        f"{p['coherence_R']:.3f} vs baseline {p['coherence_baseline']:.3f} "
        f"; signal x{p['coherence_significance']:.1f} above shifted-map null"
        f"{' - **low confidence**' if p['low_confidence'] else ''}",
        f"- Bench floor elevations (global): {', '.join(f'{v:.1f}' for v in p['bench_floor_levels_m'])} m",
        f"- Local surface correction to meet the levels: mean {p['surface_correction_m']['mean']:+.2f} m, "
        f"mean abs {p['surface_correction_m']['abs_mean']:.2f} m (`surface_correction.tif`)",
        f"- Face angle: {p['face_angle_deg']:g} deg ({p['face_angle_source']})",
        f"- Mean overall pit-wall slope {p['mean_overall_slope_deg'] or 0:.1f} deg -> implied bench "
        f"floor width {p['implied_bench_width_m_mean'] or 0:.1f} m",
        f"- Pit: {p['pit']['area_m2'] / 1e4:.1f} ha, {p['pit']['source']}; water bodies: {p['water_bodies']}",
        f"- Change vs measured in the bench zone: mean {p['dz_in_bench_zone']['mean']:+.2f} m, std "
        f"{p['dz_in_bench_zone']['std']:.2f} m, range {p['dz_in_bench_zone']['min']:.1f}.."
        f"{p['dz_in_bench_zone']['max']:.1f} m",
        f"- Mean confidence in pit: {p['mean_confidence_in_pit']:.2f}",
        f"- Grid {p['grid']['cols']} x {p['grid']['rows']} at {p['grid']['resolution_m']:.3f} m; "
        f"mesh {br['patch_verts']} vertices (stride {p['patch']['stride']})",
        "",
        "## Checks",
        "| | check | detail |",
        "|---|---|---|",
    ]
    for n, ok, d in checks:
        L.append(f"| {'PASS' if ok else '**FAIL**'} | {n} | {d} |")
    L += [
        "",
        "## In Blender",
        "`quarry_terraced.blend` uses the same BlenderGIS georeference (SRID, crs x/y) and origin as "
        "the base, so it lines up with `quarry_base.blend` and with BlenderGIS imports.",
        "- `10_terraces/TERRAIN_terraced`: interpreted mesh. Point attributes: `confidence` (0..1), "
        "`terrace_class` (0 outside, 1 floor, 2 bench, 3 face, 4 water), `dz_interp` (m).",
        "- Materials: `MAT_terrain_ortho` (imagery), `MAT_terrace_confidence` and "
        "`MAT_terrace_class` (switch the material to inspect).",
        "- `00_base_context/TERRAIN_base_context`: the measured 30 m terrain with a hole under the patch.",
        "- `PIT_outline`: pit boundary curve(s).",
        "",
        "## How it works",
        "1. Pit mask: bright, unsaturated, non-vegetated imagery, kept only where the DEM shows local "
        "relief (or the user polygon `terraces.pit_mask`).",
        "2. Edge score: band-pass image edges whose direction is parallel to the DEM gradient, i.e. "
        "lines running along the contours like bench crests and toes.",
        "3. Bench height: benches are mined at fixed elevations, so edge strength projected onto "
        "elevation repeats every H. H maximises the coherence |sum w exp(2 pi i z/H)| / sum w "
        "above the baseline of the elevation distribution itself. Significance: the same score "
        "after shifting the edge map against the DEM (8 shifts) must be clearly lower (x1.5).",
        "4. Levels: the phase of that sum gives the bench elevations G + n*H. They stay global (benches "
        "are mined at fixed elevations); where the imagery puts the edges off these levels, the DEM "
        "surface is shifted locally (by at most H/2) to meet them.",
        "5. Staircase: on the smoothed DEM surface, faces at the face angle take a fraction "
        "tan(slope)/tan(face angle) of each bench period; floors are flat. It is volume-neutral "
        "relative to the smooth surface.",
        "",
        "## Limitations",
        "- Horizontal bench positions follow the smoothed 30 m DEM contours, shifted by the image-derived "
        "levels. Benches that curve more tightly than ~30 m in plan are simplified.",
        "- The DEM predates the imagery: where the pit has expanded since the DEM, the imagery shows "
        "benches but the DEM has no depth there. The reconstruction there is unreliable (see "
        "the confidence map and dz).",
        "- Face angle and haul ramps are not measured; ramps are not modelled.",
        "- Flat pit floors keep the measured (smoothed) shape.",
        "",
        "## Check images",
        *[f"- `{common.rel(paths, i)}`" for i in imgs],
        *[f"- `{x}`" for x in br["check_images"]],
        "",
        "## Files",
        *[f"- `{common.rel(paths, f)}`" for f in files],
        *[f"- `{x}`" for x in br["exports"]],
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    common.run_main(main, "07")
