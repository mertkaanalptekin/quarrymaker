"""Stage 08 - plan linework for Rhino / CAD.

Writes exchange/rhino/vNNN/plan/:
  quarry_plan.dxf   layers QUARRY_OUTLINE, BENCH_LINES (3D, at the terraced
                    surface), ROADS (centrelines), AGRICULTURE, BLOCK_YARDS;
                    local metres (real = local + origin), like the Rhino OBJ
  *.geojson         the same layers in the project CRS, for GIS
  README_plan.md    layers, units, origin, how each layer was made
Check image: checks/vNNN/08_plan_linework.png

All layers are traced automatically from the photo (+ DEM) and are
interpreted, not surveyed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

DEFAULTS = {
    "enabled": True,
    "bench_min_length_m": 18.0,
    "bench_simplify_m": 0.5,
    "road_min_length_m": 80.0,
    "simplify_m": 2.0,
    "sun_azimuth_deg": 135.0,   # sun in the southeast (shadows fall to the northwest)
    "bench_lines_3d": True,
}


def main(argv=None) -> int:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds

    import linework as lw
    import terraces as T

    args, cfg, paths, log = common.stage_context(__doc__, "08", argv)
    pc = dict(DEFAULTS)
    pc.update(cfg.get("plan") or {})
    pc["sun_azimuth_deg"] = float((cfg.get("terraces") or {}).get("sun_azimuth_deg", pc["sun_azimuth_deg"]))
    if not args.force and common.stage_up_to_date(paths, cfg, "08"):
        log.info("Stage 08 up to date - skipping.")
        return 0
    if not pc["enabled"]:
        common.finish_stage(paths, cfg, "08", ["Plan linework disabled (plan.enabled: false)."], [], [], log)
        return 0

    meta = common.read_json(paths.base / "meta.json")
    e, crs = meta["extent"], meta["crs"]
    ox, oy = meta["origin_x"], meta["origin_y"]
    res = 2.0
    W, H = int(round((e["xmax"] - e["xmin"]) / res)), int(round((e["ymax"] - e["ymin"]) / res))
    tf = from_origin(e["xmin"], e["ymax"], res, res)
    ortho = paths.processed / "ortho_utm.tif"
    terr = paths.root / "terrain" / "10_terraces" / paths.run_id

    # surface for the 3D bench lines: the terraced surface when step 07 ran
    z = np.zeros((H, W), "float32")
    src_z = terr / "terraces_dem.tif" if (terr / "terraces_dem.tif").exists() else paths.processed / "dem_utm.tif"
    with rasterio.open(src_z) as s:
        reproject(rasterio.band(s, 1), z, src_transform=s.transform, src_crs=s.crs, dst_transform=tf,
                  dst_crs=crs, resampling=Resampling.bilinear)
    rgb = np.zeros((3, H, W), "uint8")
    with rasterio.open(ortho) as s:
        for i in range(3):
            reproject(rasterio.band(s, i + 1), rgb[i], src_transform=s.transform, src_crs=s.crs,
                      dst_transform=tf, dst_crs=crs, resampling=Resampling.average)
    rgb = rgb.transpose(1, 2, 0)
    lum = rgb.astype("float32").mean(-1) / 255.0
    zs = T.gauss(z.astype(float), 5)
    slope = T.overall_slope_deg(zs, res)

    # quarry area: from step 07 when available
    qm = terr / "pit_mask.tif"
    if qm.exists():
        quarry = np.zeros((H, W), "uint8")
        with rasterio.open(qm) as s:
            reproject(rasterio.band(s, 1), quarry, src_transform=s.transform, src_crs=s.crs,
                      dst_transform=tf, dst_crs=crs, resampling=Resampling.nearest)
        quarry = quarry > 0
        log.info("Quarry area from step 07")
    else:
        quarry, _, _ = T.quarry_area(rgb, z.astype(float), res, tf)
        log.info("Quarry area computed here (step 07 not run)")
    _, _, sat, exg = T.quarry_index(rgb)

    # ---- bench lines: full-resolution photo, block by block, vectorised ----
    log.info("Tracing bench lines (full resolution) ...")
    bench = []
    block_m, margin = 500.0, 40.0
    bpx = int(block_m / res)
    bare = T.bare_ground(rgb, res, quarry)
    with rasterio.open(ortho) as src:
        rs_ = src.res[0]
        blocks = [(r0, c0) for r0 in range(0, H, bpx) for c0 in range(0, W, bpx)
                  if bare[r0:r0 + bpx, c0:c0 + bpx].any()]
        for n, (r0, c0) in enumerate(blocks):
            r1, c1 = min(H, r0 + bpx), min(W, c0 + bpx)
            bx0, bx1 = e["xmin"] + c0 * res, e["xmin"] + c1 * res
            by1, by0 = e["ymax"] - r0 * res, e["ymax"] - r1 * res
            win = from_bounds(bx0 - margin, by0 - margin, bx1 + margin, by1 + margin,
                              src.transform).round_offsets().round_lengths()
            blk = src.read([1, 2, 3], window=win, boundless=True, fill_value=0).astype("float32")
            wtf = src.window_transform(win)
            lines = T.canny_lines(blk.mean(0) / 255.0, rs_, min_len_m=float(pc["bench_min_length_m"]),
                                  within=blk.max(0) > 0, sun_azimuth_deg=pc["sun_azimuth_deg"])
            # keep only pixels in this block's core and on bare quarry ground
            bare_hi = np.zeros(lines.shape, "uint8")
            reproject(bare[r0:r1, c0:c1].astype("uint8"), bare_hi,
                      src_transform=from_origin(bx0, by1, res, res), src_crs=crs,
                      dst_transform=wtf, dst_crs=crs, resampling=Resampling.nearest)
            lines &= bare_hi > 0
            bench += lw.trace_polylines(lines, wtf, min_len_m=float(pc["bench_min_length_m"]) * 0.6,
                                        simplify_m=float(pc["bench_simplify_m"]))
            if (n + 1) % 20 == 0:
                log.info("  %d/%d blocks", n + 1, len(blocks))
    log.info("Bench lines: %d polylines, %.1f km", len(bench), sum(g.length for g in bench) / 1000)

    # ---- roads: narrow bright, unsaturated strips -> centrelines -------------
    th = lw.white_tophat(T.gauss(lum, 1.0).astype("float32"), 6)        # narrower than ~26 m
    road = (th > 0.05) & (T.gauss(sat, 1.5) < 0.16) & (slope < 25)
    road = T.box_blur(road.astype(float), 1) > 0.5
    roads = lw.trace_polylines(lw.thin(road), tf, min_len_m=float(pc["road_min_length_m"]),
                               simplify_m=float(pc["simplify_m"]))
    log.info("Roads: %d centrelines, %.1f km", len(roads), sum(g.length for g in roads) / 1000)

    # ---- agriculture: flat, vegetated or saturated ground outside the quarry -
    veg = (T.gauss(exg, 3) > 0.01) | (T.gauss(sat, 3) > 0.12)
    farm = ~quarry & (slope < 8) & veg & (T.gauss(lum, 3) > 0.25)
    farm = T.gauss(farm.astype(float), 5) > 0.5
    agri = lw.mask_polygons(farm, tf, min_area_m2=10000, simplify_m=3.0)
    log.info("Agriculture: %d areas, %.0f ha", len(agri), sum(p.area for p in agri) / 1e4)

    # ---- block yards: dense fields of small bright blocks inside the quarry --
    blobs = lw.white_tophat(lum, 2) > 0.08                                # bright objects < ~10 m
    dens = T.gauss(blobs.astype(float), 10 / res)
    yard = quarry & (dens > 0.30) & (slope < 15)
    yard = T.gauss(yard.astype(float), 3) > 0.5
    yards = lw.mask_polygons(yard, tf, min_area_m2=2000, simplify_m=2.0)
    log.info("Block yards: %d areas, %.1f ha", len(yards), sum(p.area for p in yards) / 1e4)

    outline = lw.mask_polygons(quarry, tf, min_area_m2=20000, simplify_m=float(pc["simplify_m"]))

    # ---- write ---------------------------------------------------------------
    out = paths.rhino / "plan"
    out.mkdir(parents=True, exist_ok=True)

    def z_at(xs, ys):
        cc = np.clip(((np.asarray(xs) - e["xmin"]) / res - 0.5), 0, W - 1)
        rr = np.clip(((e["ymax"] - np.asarray(ys)) / res - 0.5), 0, H - 1)
        c0_, r0_ = np.floor(cc).astype(int), np.floor(rr).astype(int)
        c1_, r1_ = np.minimum(c0_ + 1, W - 1), np.minimum(r0_ + 1, H - 1)
        fc, fr = cc - c0_, rr - r0_
        return ((z[r0_, c0_] * (1 - fc) + z[r0_, c1_] * fc) * (1 - fr)
                + (z[r1_, c0_] * (1 - fc) + z[r1_, c1_] * fc) * fr)

    layers = {"QUARRY_OUTLINE": outline, "BENCH_LINES": bench, "ROADS": roads,
              "AGRICULTURE": agri, "BLOCK_YARDS": yards}
    dxf = out / "quarry_plan.dxf"
    counts = lw.write_dxf(dxf, layers, origin=(ox, oy),
                          z_of={"BENCH_LINES": z_at} if pc["bench_lines_3d"] else {})
    files = [dxf]
    for name, geoms in layers.items():
        p = out / f"{name.lower()}.geojson"
        lw.write_geojson(p, geoms, crs, {"layer": name})
        files.append(p)
    readme = out / "README_plan.md"
    readme.write_text(f"""# Plan linework - run {paths.run_id}

`quarry_plan.dxf` (DXF R12; opens in Rhino, AutoCAD, QGIS). **Units: metres.**
Coordinates are local, like the Rhino OBJ: real {crs} coordinates = local + origin
(origin_x = {ox:.1f}, origin_y = {oy:.1f}).

| layer | content | count |
|---|---|---|
| QUARRY_OUTLINE | the quarry area (closed polylines) | {counts['QUARRY_OUTLINE']} |
| BENCH_LINES | bench edges traced in the photo (shadow ends removed; sun from {pc['sun_azimuth_deg']:.0f} deg){', 3D at the terraced surface' if pc['bench_lines_3d'] else ''} | {counts['BENCH_LINES']} |
| ROADS | road and haul-road centrelines | {counts['ROADS']} |
| AGRICULTURE | farmland outside the quarry (closed polylines) | {counts['AGRICULTURE']} |
| BLOCK_YARDS | block stone depositing / staging areas (closed polylines) | {counts['BLOCK_YARDS']} |

All layers are traced automatically from the satellite photo (and the elevation data) and
are interpreted, not surveyed. Lines are simplified (bench lines to {pc['bench_simplify_m']} m,
the rest to {pc['simplify_m']} m) for small files. In Rhino: set units to Meters before
importing; use ProjectToCPlane on BENCH_LINES for a flat plan.
The same layers are in the *.geojson files (project CRS) for GIS.
""", encoding="utf-8")
    files.append(readme)

    # ---- check image -----------------------------------------------------------
    from PIL import Image, ImageDraw
    f = min(1.0, 2400 / W)
    img = Image.fromarray((rgb.astype(float) * 0.7 + 50).astype("uint8")).resize((int(W * f), int(H * f)))
    d = ImageDraw.Draw(img)

    def px(g):
        xy = np.asarray(g.coords if g.geom_type == "LineString" else g.exterior.coords)
        return [((x - e["xmin"]) / res * f, (e["ymax"] - y) / res * f) for x, y in xy]
    for g in agri:
        d.line(px(g), fill=(60, 170, 60), width=2)
    for g in yards:
        d.line(px(g), fill=(40, 200, 230), width=2)
    for g in outline:
        d.line(px(g), fill=(255, 255, 255), width=2)
    for g in roads:
        d.line(px(g), fill=(120, 120, 120), width=2)
    for g in bench:
        d.line(px(g), fill=(230, 30, 30), width=1)
    d.rectangle([8, 8, 640, 30], fill=(255, 255, 255))
    d.text((14, 12), "red bench lines | grey roads | white quarry | green agriculture | cyan block yards",
           fill=(0, 0, 0))
    chk = paths.checks / "08_plan_linework.png"
    img.save(chk)

    size_mb = dxf.stat().st_size / 1e6
    summary = [f"DXF {common.rel(paths, dxf)} ({size_mb:.1f} MB), local metres, origin ({ox:.0f}, {oy:.0f})",
               f"Bench lines {counts['BENCH_LINES']} ({sum(g.length for g in bench) / 1000:.1f} km), "
               f"roads {counts['ROADS']} ({sum(g.length for g in roads) / 1000:.1f} km), "
               f"agriculture {counts['AGRICULTURE']}, block yards {counts['BLOCK_YARDS']}, "
               f"quarry outline {counts['QUARRY_OUTLINE']}"]
    common.finish_stage(paths, cfg, "08", summary, files, [chk], log)
    return 0


if __name__ == "__main__":
    common.run_main(main, "08")
