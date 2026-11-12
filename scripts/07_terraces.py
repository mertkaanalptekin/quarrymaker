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
    "require_significant": True,    # stop (ask) when no region shows a bench pattern
    "region_size_m": 400.0,         # benches are tested region by region (50% overlap)
    "min_significance": 2.0,        # region evidence must beat shifted edge maps by this factor
    "bare_ground_only": True,       # look for benches only on bare rock/soil, not vegetation
    "max_bench_width_m": 60.0,      # wider implied bench floors are not quarry benches
    "method": "auto",               # auto | measured | traced  (see the report)
    "quarry_min_area_ha": 5.0,      # automatic quarry area: smallest body kept
    "line_min_length_m": 18.0,      # bench lines shorter than this are ignored (rubble, blocks)
    "bench_smoothing_m": 30.0,      # how strongly the ground between lines is flattened
    "max_cut_m": 12.0,              # largest change allowed by the cut benches
    "sun_azimuth_deg": 135.0,       # where the sun is (135 = southeast); shadow ends are not benches
}

CLASS_NAMES = {0: "outside", 1: "pit floor", 2: "bench", 3: "face", 4: "water",
               5: "pit, no clear benches (measured kept)"}


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
    gj = json.loads(p.read_text(encoding="utf-8"))
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
    user_mask = tc["pit_mask"]
    if user_mask:
        um = Path(user_mask) if Path(user_mask).is_absolute() else Path(cfg["_root"]) / user_mask
        if not um.exists():
            # e.g. config.yaml copied to a new folder without the traced outline file
            log.warning("The traced quarry outline %s was not found (was it left in the old folder?). "
                        "Using the automatic quarry area instead. To use your outline, copy "
                        "pit_outline.geojson into data/reference/ or trace it again in the app.", um)
            user_mask = None
    if user_mask:
        mask, polys = load_user_mask(user_mask, cfg, tf, (H, W))
        mask_info = {"source": f"user polygon {user_mask}", "area_m2": float(mask.sum() * res * res)}
    else:
        mask, polys, mask_info = T.quarry_area(rgb, z0, res, tf, keep_polygon=aoi_poly,
                                               min_area_ha=float(tc["quarry_min_area_ha"]))
        mask_info["source"] = "quarry area from the photo + DEM relief (automatic)"
    if mask.sum() == 0:
        raise common.PipelineError(
            "No quarry pit was found in the imagery. Draw the pit outline as a GeoJSON polygon "
            "and set terraces.pit_mask, or set terraces.pit_threshold.")
    log.info("Pit mask: %.1f ha (%s)", mask_info["area_m2"] / 1e4, mask_info["source"])

    feather = T.smoothstep(T.gauss(mask.astype(float), float(tc["feather_m"]) / res), 0.5, 0.95)
    min_slope = float(tc["min_slope_deg"])
    zone = mask & (slope >= min_slope)
    if tc.get("bare_ground_only", True):
        bare = T.bare_ground(rgb, res, mask)
        n_before = int(zone.sum())
        zone &= bare
        log.info("Bench analysis on bare ground only: %.0f%% of the pit's slopes (vegetated slopes "
                 "inside the outline are left out)", 100.0 * zone.sum() / max(1, n_before))

    # ---------------- 2. edges ----------------------------------------------
    E = T.contour_edge_score(lum, zs, res)
    cap = np.percentile(E[zone], 99.5) if zone.any() else 1.0
    En = np.clip(E / (cap + 1e-12), 0, 1)
    w = np.where(zone, En ** 2, 0.0)

    # ---------------- 3. bench height & levels (region by region) ---------------
    zl = T.gauss(z0, 1.0)
    hmin, hmax = (float(v) for v in tc["bench_height_range_m"])
    auto = str(tc["bench_height_m"]).lower() == "auto"
    fixed_h = None if auto else float(tc["bench_height_m"])
    sig_min = float(tc["min_significance"])
    # height and level estimates trust the straight middle of the walls most
    wp = T.planar_weights(w, zs, zone, res)
    tiles = T.regional_scan(zl, wp, zone, res, float(tc["region_size_m"]), hmin, hmax, fixed_h=fixed_h)
    for i_, t_ in enumerate(tiles):
        t_["_i"] = i_
    n_local = sum(1 for t in tiles if T.tile_is_significant(t, sig_min))
    rstar = T.apply_sitewide_threshold(tiles, sig_min)
    sig_tiles = [t for t in tiles if T.tile_is_significant(t, sig_min)]
    log.info("Site-wide chance level (many regions tested): %d region(s) passed alone, %d also beat "
             "the site-wide chance level (x%.1f of a typical null)", n_local, len(sig_tiles), rstar)
    log.info("Tested %d regions of %.0f m: %d show a clear bench pattern (x%.1f or more above "
             "shifted edge maps)", len(tiles), float(tc["region_size_m"]), len(sig_tiles), sig_min)
    od.mkdir(parents=True, exist_ok=True)
    ev_img = evidence_image(paths, rgb, mask, tiles, sig_min)
    forced = False
    method = str(tc.get("method", "auto")).lower()
    if method == "traced":
        sig_tiles = []
    if not sig_tiles and method != "measured":
        log.info("No benches confirmed by the elevation data%s; tracing benches from the photo "
                 "(illustrative).", "" if method == "auto" else " (method: traced)")
    elif not sig_tiles:
        diag = diagnostic_images(paths, rgb, mask, En, zl, w, zone, hmin, hmax, fixed_h)
        msg = (f"No region of the pit shows a consistent bench pattern ({len(tiles)} regions tested; "
               f"best x{max([t['sig'] for t in tiles], default=0):.1f}, need x{sig_min:.1f}).")
        if tc.get("require_significant", True):
            common.fail_stage(paths, cfg, "07", [msg, "See the bench evidence map (red = no pattern)."],
                              [ev_img] + diag, log)
            raise common.PipelineError(
                msg + " Look at the bench evidence map. Common reasons: the benches are not sharp "
                "in the satellite image, or the area was dug after the elevation data was captured "
                "(2011-2015 for COP30). If you know the bench height, enter it. You can also set "
                "terraces.require_significant: false to build benches anyway (low confidence), or "
                "terraces.enabled: false to skip this step.")
        log.warning(msg + " Building anyway because require_significant is off: LOW CONFIDENCE.")
        forced = True
        for t in tiles:
            t["sig"] = max(t["sig"], sig_min)  # treat all tested regions as evidence
        sig_tiles = tiles
    # Bench zones = connected regions with matching heights and levels (usually
    # one pit or dump each). Each zone gets ONE bench height and ONE level
    # system, estimated from all its regions together and tested again.
    groups = T.tile_zones(tiles, sig_min) if sig_tiles else []
    n_first = len(groups)
    groups = T.merge_zones(zl, wp, zone, tiles, groups, fixed_h=fixed_h) if groups else []
    if len(groups) < n_first:
        log.info("Merged %d candidate zones into %d (same bench levels)", n_first, len(groups))
    if forced and not groups:
        groups = [list(range(len(tiles)))]
    zones, zone_info, dropped = [], [], 0
    for g in groups:
        sol = T.zone_solution(zl, wp, zone, tiles, g, fixed_h=fixed_h)
        if sol["sig"] < sig_min and not forced:
            dropped += 1
            log.info("Candidate bench zone of %d regions dropped: zone test x%.1f < x%.1f",
                     len(g), sol["sig"], sig_min)
            continue
        if len(g) == 1 and sol["sig"] < 1.5 * sig_min and not forced:
            dropped += 1
            log.info("Isolated single region dropped: needs x%.1f on its own, has x%.1f",
                     1.5 * sig_min, sol["sig"])
            continue
        # plausibility: real quarry benches have floors a few to some tens of
        # metres wide, and a height inside the search range (not at its edge)
        zsel = np.zeros(zone.shape, bool)
        for ti in g:
            t_ = tiles[ti]
            zsel[t_["r0"]:t_["r1"], t_["c0"]:t_["c1"]] = True
        zsel &= zone
        wall = float(np.median(slope[zsel])) if zsel.any() else 0.0
        width = sol["H"] / math.tan(math.radians(max(wall, 1.0))) - sol["H"] / math.tan(
            math.radians(float(tc["face_angle_deg"])))
        at_edge = auto and (sol["H"] <= hmin + 0.3 or sol["H"] >= hmax - 0.3)
        if (width > float(tc["max_bench_width_m"]) or at_edge) and not forced:
            dropped += 1
            log.info("Candidate bench zone of %d regions dropped as implausible: height %.1f m%s, "
                     "implied bench width %.0f m on %.0f deg walls", len(g), sol["H"],
                     " (edge of the search range)" if at_edge else "", width, wall)
            continue
        zones.append((g, sol))
    ev_img = evidence_image(paths, rgb, mask, tiles, sig_min,
                            accepted={i_ for g_, _ in zones for i_ in g_})
    if not zones and method == "measured":
        diag = diagnostic_images(paths, rgb, mask, En, zl, w, zone, hmin, hmax, fixed_h)
        msg = ("Some regions looked promising, but no pit-wide bench pattern passed the test "
               f"({dropped} candidate zone(s) dropped).")
        common.fail_stage(paths, cfg, "07", [msg], [ev_img] + diag, log)
        raise common.PipelineError(
            msg + " There is no consistent bench pattern to build from. Look at the bench evidence map. "
            "If you know the bench height, enter it; or set terraces.enabled: false.")
    zone_info = []
    est = None
    floors = []
    Hb = None
    if zones:
        zidx, evidence = T.zone_fields(tiles, [g for g, _ in zones], (H, W))
        Hz_arr = np.array([sol["H"] for _, sol in zones])
        Gz_arr = np.array([sol["level"] for _, sol in zones])
        Hf = np.where(zidx >= 0, Hz_arr[np.maximum(zidx, 0)], Hz_arr[0]).astype("float64")
        Gz = np.where(zidx >= 0, Gz_arr[np.maximum(zidx, 0)], Gz_arr[0]).astype("float64")
        # fine local correction of the DEM towards each zone's fixed levels (+-H/2)
        L, conf = T.local_levels(zl, w, Hf, float(tc["local_window_m"]) / res, Gz)
        delta = (L - Gz + Hf / 2) % Hf - Hf / 2
        G = Gz
        weak = forced
        for zi, (g, sol) in enumerate(zones):
            Hz, Gzv = sol["H"], sol["level"]
            sub = mask & (zidx == zi) & (evidence > 0.5)
            zz = z0[sub]
            fl = []
            if zz.size:
                n0 = int(math.floor((float(zz.min()) - Gzv) / Hz - 0.5))
                n1 = int(math.ceil((float(zz.max()) - Gzv) / Hz - 0.5))
                fl = [round(Gzv + (n + 0.5) * Hz, 1) for n in range(n0, n1 + 1)
                      if zz.min() - Hz / 2 <= Gzv + (n + 0.5) * Hz <= zz.max() + Hz / 2]
            rows_, cols_ = np.where(sub) if zz.size else (np.array([0]), np.array([0]))
            zone_info.append({"zone": zi + 1, "regions": len(g), "bench_height_m": round(Hz, 2),
                              "level_mod_H_m": round(Gzv, 2), "floor_levels_m": fl,
                              "significance": round(sol["sig"], 1), "area_ha": round(float(sub.sum()) * res * res / 1e4, 1),
                              "bbox_crs": [round(e["xmin"] + cols_.min() * res), round(e["ymax"] - (rows_.max() + 1) * res),
                                           round(e["xmin"] + (cols_.max() + 1) * res), round(e["ymax"] - rows_.min() * res)]})
            log.info("Bench zone %d: %d regions, %.1f ha, bench height %.2f m, zone test x%.1f, floors %s",
                     zi + 1, len(g), zone_info[-1]["area_ha"], Hz, sol["sig"],
                     ", ".join(f"{v:.1f}" for v in fl[:12]) + (" ..." if len(fl) > 12 else ""))
        main = zones[0][0]
        Hmain = zones[0][1]["H"]
        floors = zone_info[0]["floor_levels_m"]
        est = main_spectrum(zl, wp, zone, tiles, main, hmin, hmax)
        Hb = Hmain
    else:
        evidence = np.zeros((H, W), "float32")
        conf = np.zeros((H, W), "float32")
        delta = np.zeros((H, W))
        Hf = np.ones((H, W))
        G = np.zeros((H, W))
        weak = False

    # ---------------- 4. staircase --------------------------------------------
    alpha = math.radians(float(tc["face_angle_deg"]))
    phi = np.clip(np.tan(np.radians(slope)) / math.tan(alpha), 0.05, 0.95)
    # steps are built on the measured surface itself (lightly filtered) so the
    # result is volume-neutral; the smoothed surface only gives slope/direction
    z_st, face, bench_n = T.staircase(zl - delta, G, Hf, phi)
    # benches only where the surrounding regions show evidence; elsewhere measured
    wb = feather * T.smoothstep(slope, min_slope, min_slope + 4.0) * evidence
    zt = z0 + wb * (z_st - z0)

    # ---------------- 4b. benches cut along the photo's bench lines -------------
    trc = None
    if method in ("auto", "traced"):
        within_tr = mask & (evidence <= 0.5)
        if tc.get("bare_ground_only", True):
            within_tr &= T.bare_ground(rgb, res, mask)
        log.info("Tracing bench lines in the photo at full resolution ...")
        lines = T.bench_lines_from_file(ortho_tif, tf, (H, W), crs, within_tr,
                                        min_len_m=float(tc["line_min_length_m"]),
                                        sun_azimuth_deg=float(tc["sun_azimuth_deg"]), log=log)
        log.info("Cutting benches along %.1f km of bench lines ...", lines.sum() * res / 1000.0)
        dz_l, linfo = T.cut_benches(z0, lines, within_tr, res, ell_m=float(tc["bench_smoothing_m"]),
                                    dz_max=float(tc["max_cut_m"]), transform=tf)
        weight_l = feather * (mask & (evidence <= 0.5))
        zt = zt + weight_l * dz_l
        affected = (np.abs(dz_l) > 0.25) & (weight_l > 0.5)
        gz = np.hypot(*np.gradient(zt, res))
        trc = {"lines": lines, "info": linfo, "affected": affected,
               "face": affected & lines & (gz > 0.5), "weight": weight_l * affected}
        log.info("Benches cut along the photo's bench lines: %.1f ha changed; step height at the lines "
                 "median %s m (75%%: %s m, 90%%: %s m), from the elevation data", affected.sum() * res * res / 1e4,
                 linfo.get("step_height_median_m"), linfo.get("step_height_p75_m"), linfo.get("step_height_p90_m"))
        if Hb is None and linfo.get("step_height_median_m"):
            Hb = float(linfo["step_height_median_m"])
    traced_any = bool(trc is not None and trc["affected"].any())
    if not zones and not traced_any:
        diag = diagnostic_images(paths, rgb, mask, En, zl, w, zone, hmin, hmax, fixed_h)
        msg = "Neither the elevation data nor the photo's bench lines give benches in this quarry area."
        common.fail_stage(paths, cfg, "07", [msg], [ev_img] + diag, log)
        raise common.PipelineError(
            msg + " Check the quarry outline, or set terraces.enabled: false; your other results "
            "are complete.")
    if Hb is None:
        Hb = 1.0

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
    classes[mask & (slope >= min_slope) & (evidence <= 0.5)] = 5
    method_map = np.zeros((H, W), "uint8")   # 1 = confirmed by elevation data, 2 = traced (illustrative)
    method_map[mask & (wb > 0.5)] = 1
    if traced_any:
        tr = mask & trc["affected"]
        classes[tr & ~trc["face"]] = 2
        classes[tr & trc["face"]] = 3
        method_map[tr] = 2
    if n_water:
        classes[mask & (labels > 0)] = 4
    dz = zt - z0
    confidence = np.where(mask, conf * evidence, 0.0)
    built = mask & ((wb > 0.5) | (method_map == 2))

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
             wr("surface_correction.tif", np.where(wb > 0, -delta * wb, 0.0), "float32"),
             wr("bench_method.tif", method_map, "uint8")]
    ox, oy = meta["origin_x"], meta["origin_y"]
    gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}},
          "features": [{"type": "Feature", "properties": {"id": i, "area_m2": round(p.area)},
                        "geometry": mapping(p)} for i, p in enumerate(polys)]}
    outline = od / "pit_outline.geojson"
    if not outline.exists():
        outline.write_text(json.dumps(gj), encoding="utf-8")
    files.append(outline)

    patch = od / "blender_patch.npz"
    sl = (slice(r0, r1, stride), slice(c0, c1, stride))
    if not patch.exists():
        np.savez(patch, z=zt[sl].astype("float32"), confidence=confidence[sl].astype("float32"),
                 classes=classes[sl].astype("int32"), dz=dz[sl].astype("float32"),
                 method=method_map[sl].astype("int32"))
    zone_dz = dz[built]
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
        "bench_level_mod_H_m": zone_info[0]["level_mod_H_m"] if zone_info else None,
        "bench_floor_levels_m": floors,
        "bench_zones": zone_info,
        "bench_level_note": ("Each bench zone (connected regions with evidence, usually one pit or "
                             "dump) has its own fixed levels: floors at G + (n + 0.5) * H. The DEM "
                             "surface is shifted locally by up to +-H/2 to meet them "
                             "(surface_correction.tif). Values above are for the largest zone."),
        "regions": {"size_m": float(tc["region_size_m"]), "tested": len(tiles),
                    "with_benches": sum(1 for t in tiles if T.tile_is_significant(t, sig_min)),
                    "min_significance": sig_min, "sitewide_chance_ratio": rstar,
                    "pit_area_with_benches_pct": float(100 * (evidence[zone] > 0.5).mean()) if zone.any() else 0.0,
                    "list": [{k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in t.items()
                              if k2 not in ("curve", "hs", "_i", "nulls")} for t in tiles]},
        "surface_correction_m": {"mean": float(-(delta * wb)[zone].mean()) if zone.any() else 0.0,
                                 "abs_mean": float(np.abs(delta * wb)[zone].mean()) if zone.any() else 0.0},
        "method": method,
        "traced": {**(trc["info"] if trc else {}),
                   "area_ha": round(float(trc["affected"].sum()) * res * res / 1e4, 1) if trc else 0.0,
                   "note": ("Illustrative: bench lines traced in the photo (light/dark edges of faces and "
                            "their shadows); the ground between lines becomes flat benches and each line "
                            "a face whose height is the elevation data's drop there. Hills and low points "
                            "stay as measured.")},
        "coherence_R": est["R"] if est else None, "coherence_baseline": est["R_baseline"] if est else None,
        "signal_score": est["score"] if est else None,
        "coherence_significance": float(max([t["sig"] for t in tiles], default=0.0)),
        "low_confidence": bool(weak),
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
        "mean_confidence_in_pit": float((conf * evidence)[mask].mean()),
        "settings": {k2: tc[k2] for k2 in DEFAULTS},
        "base_hashes_before": base_hashes,
        "spectrum": {"h_values": est["h_values"].round(3).tolist(),
                     "R_edges": est["R_edges"].round(5).tolist(),
                     "R_baseline": est["R_base"].round(5).tolist()} if est else None,
    }
    params_p = od / "terraces_params.json"
    common.write_json(params_p, params)
    files.append(params_p)

    # ---------------- check images ------------------------------------------
    imgs = [ev_img, traced_image(paths, rgb, mask, method_map)] + (
        [lines_image(paths, rgb, mask, trc["lines"])] if trc is not None else []) + check_images(paths, od, rgb, mask, En, z0, zt, zs, classes, confidence, est, Hb,
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
    report.write_text(make_report(paths, meta, params, checks, br, imgs, files), encoding="utf-8")
    summary = [
        f"Pit: {mask_info['area_m2'] / 1e4:.1f} ha ({mask_info['source']}); water bodies {n_water}",
        f"Benches found in {params['regions']['with_benches']} of {params['regions']['tested']} regions "
        f"({params['regions']['pit_area_with_benches_pct']:.0f}% of the pit's slopes); the rest keeps the measured terrain",
    ] + [f"Bench zone {z['zone']}: height {z['bench_height_m']:.1f} m, floors at "
         f"{', '.join(f'{v:.1f}' for v in z['floor_levels_m'][:15])}{' ...' if len(z['floor_levels_m']) > 15 else ''} m"
         for z in zone_info[:6]] + [
        f"Main bench height {Hb:.2f} m ({params['bench_height_source']}){' - LOW CONFIDENCE' if weak else ''}",
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


def main_spectrum(zl, w, zone, tiles, main, hmin, hmax):
    """Spectrum over the main bench zone (for the check image and the report)."""
    import numpy as np

    import terraces as T
    sel = np.zeros(zone.shape, bool)
    for i in main:
        t = tiles[i]
        sel[t["r0"]:t["r1"], t["c0"]:t["c1"]] = True
    sel &= zone
    if not sel.any():
        sel = zone
    return T.estimate_bench_height(zl[sel], w[sel], np.ones(int(sel.sum())), hmin, hmax)


def lines_image(paths, rgb, mask, lines):
    """Photo with the traced bench lines (red) and the quarry outline (white)."""
    import numpy as np
    from PIL import Image
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    a = rgb[crop].copy()
    a[lines[crop]] = (255, 30, 30)
    m = mask[crop]
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    a[edge] = (255, 255, 255)
    img = Image.fromarray(a)
    img.thumbnail((2400, 2400))
    p = paths.checks / "07_bench_lines.png"
    img.save(p)
    return p


def traced_image(paths, rgb, mask, method_map):
    """Photo with the benches: green = confirmed by the elevation data,
    orange = traced from the photo (illustrative)."""
    import numpy as np
    from PIL import Image, ImageDraw
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    a = rgb[crop].astype(float)
    mm = method_map[crop]
    a[mm == 1] = a[mm == 1] * 0.45 + np.array([40, 200, 90]) * 0.55
    a[mm == 2] = a[mm == 2] * 0.45 + np.array([255, 120, 20]) * 0.55
    img = Image.fromarray(a.astype("uint8"))
    img.thumbnail((1600, 1600))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, 520, 44], fill=(255, 255, 255))
    d.text((14, 12), "green = benches confirmed by elevation data", fill=(0, 0, 0))
    d.text((14, 26), "orange = benches cut along the photo's bench lines (illustrative)", fill=(0, 0, 0))
    p = paths.checks / "07_benches_measured_and_traced.png"
    img.save(p)
    return p


def evidence_image(paths, rgb, mask, tiles, sig_min, accepted=None):
    """Map of the tested regions, written whether or not the stage succeeds.
    green = benches built, amber = a possible pattern that did not pass the
    pit-wide test (or a weak one), red = no pattern. `accepted`: indices of the
    regions in accepted bench zones (None = before the zone test)."""
    import numpy as np
    from PIL import Image, ImageDraw

    import terraces as T
    H, W = mask.shape
    s = min(1.0, 1400 / max(H, W))
    img = Image.fromarray((rgb.astype(float) * 0.55 + 40).astype("uint8")).resize(
        (max(1, int(W * s)), max(1, int(H * s))))
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for t in sorted(tiles, key=lambda t: t["sig"]):
        ok = T.tile_is_significant(t, sig_min)
        if ok and (accepted is None or t["_i"] in accepted):
            col = (40, 200, 90)
        elif ok or t["sig"] >= sig_min * 0.75:
            col = (240, 180, 40)
        else:
            col = (220, 60, 50)
        # draw the central half of each region (regions overlap by 50%)
        h4, w4 = (t["r1"] - t["r0"]) / 4, (t["c1"] - t["c0"]) / 4
        box = [(t["c0"] + w4) * s, (t["r0"] + h4) * s, (t["c1"] - w4) * s, (t["r1"] - h4) * s]
        d.rectangle(box, fill=col + (110,), outline=col + (255,))
    m = Image.fromarray((mask * 255).astype("uint8")).resize(img.size)
    edge = np.asarray(m.filter(__import__("PIL.ImageFilter", fromlist=["FIND_EDGES"]).FIND_EDGES)) > 60
    o = np.asarray(over).copy()
    o[edge] = (255, 255, 255, 255)
    img = Image.alpha_composite(img.convert("RGBA"), Image.fromarray(o)).convert("RGB")
    dd = ImageDraw.Draw(img)
    dd.rectangle([8, 8, 470, 62], fill=(255, 255, 255))
    dd.text((14, 12), "Bench evidence per region:", fill=(0, 0, 0))
    dd.text((14, 28), "green = benches built, amber = possible but not confirmed, red = none", fill=(0, 0, 0))
    dd.text((14, 44), "white line = pit outline", fill=(0, 0, 0))
    p = paths.checks / "07_bench_evidence.png"
    img.save(p)
    return p


def diagnostic_images(paths, rgb, mask, En, zl, w, zone, hmin, hmax, fixed_h):
    """Images written when no region shows benches, to help decide what to do."""
    import numpy as np
    from PIL import Image

    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    a = Image.fromarray(rgb[crop])
    b = Image.fromarray((np.clip(En[crop], 0, 1) ** 0.5 * 255).astype("uint8")).convert("RGB")
    two = Image.new("RGB", (2 * a.width + 10, a.height), "white")
    two.paste(a, (0, 0))
    two.paste(b, (a.width + 10, 0))
    two.thumbnail((1600, 1600))
    p = paths.checks / "07_contour_edges.png"
    two.save(p)
    return [p]


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
    pal = np.array([[0, 0, 0], [80, 160, 255], [255, 210, 60], [230, 60, 40], [40, 90, 255],
                    [190, 190, 200]], "uint8")
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
    if est is None:
        est = {"h_values": np.array([5.0, 25.0]), "R_edges": np.zeros(2), "R_base": np.zeros(2)}
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
    col = np.stack([255 - c, c, np.zeros_like(c)], -1)
    dim = (rgb[crop].astype(float) * 0.45 + 60).astype("uint8")  # outside the pit: dimmed photo
    col = np.where(mask[crop][..., None], col, dim)
    fit(Image.fromarray(col.astype("uint8"))).save(p6)
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
        f"- Regions of {p['regions']['size_m']:.0f} m tested: {p['regions']['tested']}; with a clear bench "
        f"pattern: {p['regions']['with_benches']}. Benches are built on "
        f"{p['regions']['pit_area_with_benches_pct']:.0f}% of the pit's slopes; the rest keeps the measured "
        f"terrain (`07_bench_evidence.png`).{' **Low confidence (forced build).**' if p['low_confidence'] else ''}",
        "",
        "| bench zone | area | bench height | zone test | floor elevations (m) |",
        "|---|---|---|---|---|",
        *[f"| {z['zone']} | {z['area_ha']:.1f} ha | {z['bench_height_m']:.2f} m ({p['bench_height_source']}) | "
          f"x{z['significance']:.1f} | {', '.join(f'{v:.1f}' for v in z['floor_levels_m'])} |"
          for z in p["bench_zones"]],
        "",
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
        *(["### Benches cut along the photo's bench lines (illustrative)",
           f"{p['traced'].get('line_length_km', 0)} km of bench lines; {p['traced']['area_ha']:.1f} ha changed. "
           f"Step height at the lines (from the elevation data): median {p['traced'].get('step_height_median_m')} m, "
           f"75% {p['traced'].get('step_height_p75_m')} m, 90% {p['traced'].get('step_height_p90_m')} m. "
           f"{p['traced']['note']}",
           ""] if p.get("traced", {}).get("area_ha") else []),
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
        "`terrace_class` (0 outside, 1 floor, 2 bench, 3 face, 4 water, 5 pit without clear "
        "benches = measured kept), `dz_interp` (m), `bench_method` (1 confirmed by elevation data, "
        "2 traced from the photo).",
        "- Materials: `MAT_terrain_ortho` (imagery), `MAT_terrace_confidence`, `MAT_terrace_class` "
        "and `MAT_bench_method` (switch the material to inspect).",
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
        "above the baseline of the elevation distribution itself (edges on straight wall parts "
        "weigh more).",
        "4. Region by region: the pattern is tested in overlapping regions and must beat shifted "
        "edge maps (circular and spatial shifts) by the configured factor. Agreeing regions form "
        "bench zones (usually one pit or dump each), which are re-tested as a whole and checked for "
        "plausible bench widths. Each zone gets one bench height and one set of fixed floor "
        "elevations G + (n + 0.5)*H; the DEM surface is shifted locally (by at most H/2) to meet "
        "them. Regions without evidence keep the measured terrain.",
        "5. Staircase: on the smoothed DEM surface, faces at the face angle take a fraction "
        "tan(slope)/tan(face angle) of each bench period; floors are flat. It is volume-neutral "
        "relative to the smooth surface.",
        "",
        "## Limitations",
        "- Benches cut along the photo's bench lines (orange in `07_benches_measured_and_traced.png`, "
        "`bench_method` = 2) are illustrative: their plan position follows the photo, their height "
        "is the elevation data's drop. Where the quarry was dug deeper after the elevation data "
        "was captured, the steps come out too shallow. Roads and yard edges can make small steps.",
        "- Small pits, whose walls are only a few 30 m DEM cells wide, are less reliable (the DEM "
        "squeezes their elevations; the bench height can come out a few percent low).",
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
