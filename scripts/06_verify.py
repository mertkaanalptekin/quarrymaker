"""Stage 06 - verify.

Automated checks (each must pass or is reported clearly):
  1. mesh bounding box matches the extent within half a cell
  2. mesh min/max Z match the DEM min/max within 0.01 m
  3. Blender scene georef properties equal CRS and origin in meta.json
  4. Gaea PNG decodes back to the DEM (resampled to the Gaea grid) within the
     16-bit quantisation error; RMSE reported in metres
  5. the OBJ re-imports Z-up with the correct extents (parsed directly from
     the file, and re-imported in Blender with up axis Z)
Visual checks: top + oblique renders, side-by-side with the Google Earth
reference image from data/reference/. Writes terrain/00_base/vNNN/report.md.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import exr_io  # noqa: E402

HERE = Path(__file__).resolve().parent
REF_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_obj_bbox(path: Path):
    import numpy as np
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                v = np.array(line.split()[1:4], dtype=float)
                np.minimum(mins, v, out=mins)
                np.maximum(maxs, v, out=maxs)
                n += 1
    return [*mins.tolist(), *maxs.tolist()], n


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail):
        self.items.append({"name": name, "ok": bool(ok), "detail": detail})

    @property
    def all_ok(self):
        return all(c["ok"] for c in self.items)


def main(argv=None) -> int:
    import numpy as np
    import rasterio
    from PIL import Image

    args, cfg, paths, log = common.stage_context(__doc__, "06", argv)
    if not args.force and common.stage_up_to_date(paths, cfg, "06"):
        log.info("Stage 06 up to date - skipping.")
        return 0
    meta = common.read_json(paths.base / "meta.json")
    blend = paths.base / "quarry_base.blend"
    obj = paths.rhino / "quarry_base.obj"
    gaea_meta_p = paths.gaea / "gaea_meta.json"
    for p, st in ((blend, "04"), (gaea_meta_p, "05")):
        if not p.exists():
            raise common.PipelineError(f"{p} is missing; run stage {st} first.")
    bcfg = cfg.get("blender") or {}
    blender, _ = common.find_blender(cfg)
    if not blender:
        raise common.PipelineError("Blender path is not set (config blender.path or BLENDER_PATH).")

    render_top = paths.checks / "06_render_top.png"
    render_obl = paths.checks / "06_render_oblique.png"
    job = {"blend": str(blend), "obj": str(obj) if obj.exists() else None,
           "render_top": str(render_top), "render_oblique": str(render_obl),
           "render_px": int(bcfg.get("check_render_px", 1600)),
           "samples": int(bcfg.get("check_render_samples", 32)),
           "result": str(paths.checks / "blender_06_result.json")}
    job_path = paths.checks / "blender_job_06.json"
    common.write_json(job_path, job)
    b04 = _load("build_blender", "04_build_blender.py")
    rc = b04.run_blender(blender, HERE / "blender_helpers" / "verify_in_blender.py", job_path, log)
    if rc != 0 or not Path(job["result"]).exists():
        raise common.PipelineError(f"Blender verification run failed (exit {rc}); see {paths.log_file}")
    br = common.read_json(job["result"])
    if br.get("error"):
        raise common.PipelineError("Blender verification failed: " + br["error"])

    C = Checks()
    e = meta["extent"]
    r = meta["grid"]["cell_size_m"]
    ox, oy = meta["origin_x"], meta["origin_y"]
    tol_xy = r / 2 + 1e-3

    # 1. bbox vs extent ---------------------------------------------------
    mb = br["mesh_bbox"]
    exp = [e["xmin"] - ox, e["ymin"] - oy, e["xmax"] - ox, e["ymax"] - oy]
    got = [mb[0], mb[1], mb[3], mb[4]]
    diffs = [abs(a - b) for a, b in zip(got, exp)]
    C.add("Mesh bbox matches extent (within half a cell)", max(diffs) <= tol_xy,
          f"local mesh X {got[0]:.3f}..{got[2]:.3f}, Y {got[1]:.3f}..{got[3]:.3f}; extent X "
          f"{exp[0]:.3f}..{exp[2]:.3f}, Y {exp[1]:.3f}..{exp[3]:.3f}; max diff {max(diffs):.3f} m "
          f"(limit {tol_xy:.3f} m; vertices sit at cell centres so {r / 2:g} m is expected)")

    # 2. Z range ----------------------------------------------------------
    zmin, zmax = meta["elevation"]["min"], meta["elevation"]["max"]
    dz = max(abs(mb[2] - zmin), abs(mb[5] - zmax))
    note = ""
    if br.get("dem_stride", 1) > 1:
        note = " (mesh is decimated; extremes may be skipped)"
    C.add("Mesh min/max Z match DEM (within 0.01 m)", dz <= 0.01,
          f"mesh Z {mb[2]:.4f}..{mb[5]:.4f}, DEM {zmin:.4f}..{zmax:.4f}, max diff {dz:.4f} m{note}")

    # 3. georef ------------------------------------------------------------
    g = br["georef"]
    ok = (g.get("SRID") == meta["crs"]
          and g.get("crs x") is not None and abs(g["crs x"] - ox) < 1e-6
          and g.get("crs y") is not None and abs(g["crs y"] - oy) < 1e-6
          and g.get("longitude") is not None and abs(g["longitude"] - meta["origin_lon"]) < 1e-9
          and g.get("latitude") is not None and abs(g["latitude"] - meta["origin_lat"]) < 1e-9)
    C.add("Blender georef (BlenderGIS keys) equals meta.json", ok,
          f"SRID={g.get('SRID')!r}, crs x={g.get('crs x')}, crs y={g.get('crs y')}, "
          f"lon={g.get('longitude')}, lat={g.get('latitude')}; meta: {meta['crs']}, "
          f"{ox}, {oy}, {meta['origin_lon']}, {meta['origin_lat']}")
    u = br["units"]
    C.add("Scene units metric, scale 1.0",
          u["system"] == "METRIC" and abs(u["scale_length"] - 1.0) < 1e-9, str(u))

    # 4. Gaea round trip -----------------------------------------------------
    gm = common.read_json(gaea_meta_p)
    gaea_mod = _load("export_gaea", "05_export_gaea.py")
    with rasterio.open(paths.processed / "dem_utm.tif") as ds:
        dem = ds.read(1).astype("float64")
    N, p, sq = gm["size_px"], gm["pixel_size_m"], gm["square_bounds_crs"]
    xs = sq["xmin"] + (np.arange(N) + 0.5) * p
    ys = sq["ymax"] - (np.arange(N) + 0.5) * p
    X, Y = np.meshgrid(xs, ys)
    ref = gaea_mod.sample_clamped(dem, e["xmin"], e["ymax"], r, r, X, Y)
    v = np.asarray(Image.open(paths.gaea / gm["files"]["heightmap_png"])).astype("float64")
    dec = gm["zmin"] + v / 65535.0 * (gm["zmax"] - gm["zmin"])
    err = dec - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    maxe = float(np.abs(err).max())
    step = gm["quantization_step_m"]
    C.add("Gaea PNG decodes to DEM within 16-bit quantisation",
          v.shape == (N, N) and maxe <= step / 2 + 1e-6,
          f"RMSE {rmse * 1000:.3f} mm (expected ~{gm['expected_rmse_m'] * 1000:.3f} mm), max "
          f"{maxe * 1000:.3f} mm (limit {step / 2 * 1000:.3f} mm = half step), {N}x{N} px")
    exr = exr_io.read_exr(paths.gaea / gm["files"]["heightmap_exr"])["R"]
    exr_err = float(np.abs(exr - ref).max())
    C.add("Gaea EXR holds real metres", exr_err < 1e-3 and abs(float(exr.min()) - zmin) < 0.05 + r,
          f"max abs(EXR - DEM) {exr_err:.2e} m, EXR range {exr.min():.3f}..{exr.max():.3f} m")
    # DEM cell-centre sanity: decoded heightmap at the pixel nearest each sampled DEM cell
    rr = np.arange(0, dem.shape[0], max(1, dem.shape[0] // 50))
    cc = np.arange(0, dem.shape[1], max(1, dem.shape[1] // 50))
    CX = e["xmin"] + (cc + 0.5) * r
    CY = e["ymax"] - (rr + 0.5) * r
    gx, gy = np.meshgrid(CX, CY)
    back = gaea_mod.sample_clamped(dec, sq["xmin"], sq["ymax"], p, p, gx, gy)
    rmse_cells = float(np.sqrt(np.mean((back - dem[np.ix_(rr, cc)]) ** 2)))

    # 5. OBJ ---------------------------------------------------------------
    if obj.exists():
        ob, nv = parse_obj_bbox(obj)
        d_obj = max(abs(a - b) for a, b in zip(ob, mb))
        zup = abs(ob[2] - zmin) <= 0.01 and abs(ob[5] - zmax) <= 0.01
        C.add("OBJ is Z-up with correct extents (file coordinates)", d_obj <= 1e-3 and zup,
              f"OBJ X {ob[0]:.3f}..{ob[3]:.3f}, Y {ob[1]:.3f}..{ob[4]:.3f}, Z {ob[2]:.3f}..{ob[5]:.3f}"
              f" ({nv} vertices); max diff to Blender mesh {d_obj:.4f} m")
        ri = br.get("obj_reimport")
        if ri:
            d_ri = max(abs(a - b) for a, b in zip(ri["bbox"], mb))
            C.add("OBJ re-imports in Blender (up Z) with matching bbox", d_ri <= 1e-3,
                  f"re-imported bbox {[round(x, 3) for x in ri['bbox']]}, max diff {d_ri:.4f} m")
        else:
            C.add("OBJ re-imports in Blender (up Z) with matching bbox", False, "re-import produced no mesh")
    else:
        fmts = (cfg.get("rhino") or {}).get("formats") or []
        C.add("OBJ is Z-up with correct extents", "obj" not in fmts,
              "OBJ not exported" + ("" if "obj" in fmts else " (not in rhino.formats)"))

    # Visual: side-by-side with the Google Earth reference ------------------
    refs = sorted(p for p in paths.reference.glob("*") if p.suffix.lower() in REF_EXT)
    side = paths.checks / "06_top_vs_reference.png"
    top = Image.open(render_top).convert("RGB")
    H = 1000
    top_s = top.resize((round(top.width * H / top.height), H))
    if refs:
        ref_img = Image.open(refs[0]).convert("RGB")
        ref_s = ref_img.resize((round(ref_img.width * H / ref_img.height), H))
        sheet = Image.new("RGB", (top_s.width + ref_s.width + 20, H), "white")
        sheet.paste(top_s, (0, 0))
        sheet.paste(ref_s, (top_s.width + 20, 0))
        ref_note = f"compared with {common.rel(paths, refs[0])} (not georeferenced; visual only)"
    else:
        sheet = top_s
        ref_note = "no reference image found in data/reference/ - side-by-side shows the render only"
        log.warning(ref_note)
    sheet.save(side)

    # Report -----------------------------------------------------------------
    report = paths.base / "report.md"
    if report.exists():
        report = paths.base / f"report_rerun_{time.strftime('%Y%m%d_%H%M%S')}.md"
        log.info("report.md exists (write-once); writing %s", report.name)
    report.write_text(make_report(paths, cfg, meta, gm, br, C, rmse, rmse_cells, ref_note,
                                  [render_top, render_obl, side]))

    summary = [f"{'PASS' if c['ok'] else 'FAIL'}: {c['name']} - {c['detail']}" for c in C.items]
    summary.append(f"Gaea RMSE {rmse * 1000:.3f} mm on the Gaea grid; "
                   f"{rmse_cells:.3f} m vs DEM cells after resampling round trip")
    summary.append(f"Report: {common.rel(paths, report)}")
    if not C.all_ok:
        for line in summary:
            log.error(line)
        raise common.PipelineError(
            "Verification failed: " + "; ".join(c["name"] for c in C.items if not c["ok"])
            + f". Details in {common.rel(paths, report)}")
    common.finish_stage(paths, cfg, "06", summary, [report],
                        [render_top, render_obl, side], log, extra={"checks": C.items})
    return 0


def make_report(paths, cfg, meta, gm, br, C, rmse, rmse_cells, ref_note, images) -> str:
    e = meta["extent"]
    aoi = meta["aoi_wgs84"]
    lines = [
        f"# Base terrain report - run {paths.run_id}",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}, config hash `{meta['config_hash']}`, "
        f"Blender {br.get('blender_version')}.",
        "",
        "## Inputs",
        f"- AOI (WGS84): W {aoi['west']}, S {aoi['south']}, E {aoi['east']}, N {aoi['north']}; "
        f"buffer {aoi['buffer_m']:g} m",
        f"- DEM: {meta['dem']['source']} via OpenTopography globaldem (`{meta['dem']['raw_file']}`), "
        f"native {meta['dem']['native_grid']}",
        f"- Capture date: {meta['dem']['capture_date_note']}",
        f"- Vertical datum: {meta['dem']['vertical_datum']}",
        f"- Imagery: {meta['imagery']['mode']} - {meta['imagery'].get('attribution')}",
        "",
        "## Settings",
        f"- CRS {meta['crs']}, metres, Z up",
        f"- Extent: X {e['xmin']:.1f}..{e['xmax']:.1f}, Y {e['ymin']:.1f}..{e['ymax']:.1f} "
        f"({e['width_m']:.0f} x {e['height_m']:.0f} m)",
        f"- Local origin: origin_x = {meta['origin_x']:.1f}, origin_y = {meta['origin_y']:.1f} "
        f"(lon {meta['origin_lon']:.7f}, lat {meta['origin_lat']:.7f})",
        f"- DEM grid {meta['grid']['cols']} x {meta['grid']['rows']} at {meta['grid']['cell_size_m']:g} m; "
        f"resampling {meta['resampling']['dem_method']} ({meta['resampling']['dem_from']} -> "
        f"{meta['resampling']['dem_to']}){'; UPSAMPLED' if meta['resampling']['dem_upsampled'] else ''}",
        f"- Nodata cells filled: {meta['dem']['nodata_cells_filled']}",
        f"- Elevation {meta['elevation']['min']:.2f} .. {meta['elevation']['max']:.2f} m",
        f"- Ortho {meta['imagery']['pixels'][0]} x {meta['imagery']['pixels'][1]} px at "
        f"{meta['imagery']['pixel_size_m']:.3f} m",
        f"- Mesh: {br['mesh_verts']} vertices, {br['mesh_quads']}/{br['mesh_faces']} quads, "
        f"DEM stride {br.get('dem_stride', 1)}",
        f"- Gaea: {gm['size_px']} px, {gm['pixel_size_m']:.4f} m/px, width {gm['terrain_width_m']:.2f} m, "
        f"height {gm['height_range_m']:.2f} m",
        "",
        "## Checks",
        "| | check | detail |",
        "|---|---|---|",
    ]
    for c in C.items:
        lines.append(f"| {'PASS' if c['ok'] else '**FAIL**'} | {c['name']} | {c['detail']} |")
    lines += [
        "",
        f"Gaea 16-bit round-trip RMSE: **{rmse * 1000:.3f} mm** on the Gaea grid. Sampling the "
        f"Gaea grid back at DEM cell centres gives {rmse_cells:.3f} m RMSE; this is the bilinear "
        "resampling error, not quantisation.",
        "",
        "## Visual checks",
        *[f"- `{common.rel(paths, p)}`" for p in images],
        f"- Reference: {ref_note}",
        "",
        "## Known limitations",
        f"- At {meta['grid']['cell_size_m']:g} m resolution, quarry benches (about 10-15 m high) are "
        "**not captured**; slopes between benches appear as smooth ramps.",
        f"- The DEM's capture date is earlier than the imagery's ({meta['dem']['capture_date_note']}) "
        "Excavation after the DEM capture shows in the texture but not in the geometry.",
        "- COP30 is a surface model (DSM): buildings, stockpiles and machinery may be included.",
        "- Reprojection to the projected CRS resamples the DEM "
        f"({meta['resampling']['dem_method']}); no smoothing or edits were applied.",
        "- The Gaea heightmap is padded with edge values to make it square; padded strips are not real terrain.",
        "",
        "## Output files",
    ]
    for d in (paths.base, paths.rhino, paths.gaea, paths.processed):
        for f in sorted(d.rglob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                lines.append(f"- `{common.rel(paths, f)}` ({f.stat().st_size / 1e6:.2f} MB)")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    common.run_main(main, "06")
