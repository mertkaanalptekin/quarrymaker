"""Runs inside Blender for stage 07. Builds quarry_terraced.blend:

  10_terraces/TERRAIN_terraced     interpreted fine mesh over the pit (+margin),
                                   point attributes confidence / terrace_class / dz_interp
  10_terraces/PIT_outline          pit boundary curve(s)
  00_base_context/TERRAIN_base_context   measured 30 m terrain with a hole under the patch
  00_base_context/ORIGIN_marker
  10_terraces_rig                  cameras + sun framed on the pit

Same BlenderGIS georef (SRID, crs x/y, lon/lat) and origin as the base.
Never opens or saves quarry_base.blend.
"""
import json
import logging
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bl_common as bl  # noqa: E402


def job_path():
    rest = sys.argv[sys.argv.index("--") + 1:]
    return rest[rest.index("--job") + 1]


def build(job, log):
    meta = json.loads(Path(job["meta"]).read_text(encoding="utf-8"))
    prm = json.loads(Path(job["params"]).read_text(encoding="utf-8"))
    root = Path(job["checks_dir"]).parent.parent
    blend = Path(job["blend"])
    ext = meta["extent"]
    ox, oy = float(meta["origin_x"]), float(meta["origin_y"])
    origin = (ox, oy)
    r = float(meta["grid"]["cell_size_m"])

    if blend.exists():
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        log.info("%s exists (write-once): opened, not rebuilt.", blend)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        build_scene(job, meta, prm, ext, origin, r, log)
        bl.save_blend(blend)
        log.info("Saved %s", blend)

    terr = bpy.data.objects["TERRAIN_terraced"]
    exports = []
    rhino = Path(job["rhino_dir"])
    rhino.mkdir(parents=True, exist_ok=True)
    tex = rhino / "quarry_terraced_texture.jpg"
    if not tex.exists():
        import shutil
        shutil.copy2(job["ortho_jpg"], tex)
    exports.append(tex)
    if "obj" in job["formats"]:
        p = rhino / "quarry_terraced.obj"
        if not p.exists():
            bl.export_obj(terr, p, tex.name)
        exports += [p, p.with_suffix(".mtl")]
    if "fbx" in job["formats"]:
        p = rhino / "quarry_terraced.fbx"
        if not p.exists():
            bl.export_fbx(terr, p)
        exports.append(p)
    notes = rhino / "README_terraced.md"
    if not notes.exists():
        notes.write_text(
            f"# Interpreted terraces - run {meta['run_id']}\n\n"
            "INTERPRETED geometry (benches inferred from imagery + DEM), not a survey.\n"
            "Same coordinates as the base export: metres, Z up, local XY = real - origin.\n"
            f"- origin_x = {ox:.1f}, origin_y = {oy:.1f} ({meta['crs']})\n"
            f"- bench height {prm['bench_height_m']:.2f} m, face angle {prm['face_angle_deg']:g} deg\n"
            f"- bench floor elevations: {', '.join(f'{v:.1f}' for v in prm['bench_floor_levels_m'])} m\n"
            "Rhino: set units to Meters; turn off 'Map OBJ Y to Rhino Z'.\n", encoding="utf-8")
    exports.append(notes)

    checks = Path(job["checks_dir"])
    obl = checks / "07_render_oblique.png"
    top = checks / "07_render_top.png"
    bl.render("CAM_terraces_oblique", obl, 900, 16)
    cam = bpy.data.objects["CAM_terraces_top"]
    bl.render("CAM_terraces_top", top, 900, 16, list(cam.get("render_aspect", [1, 1])))

    scn = bpy.context.scene
    st = bl.mesh_stats(terr)
    return {
        "blender_version": bpy.app.version_string,
        "patch_verts": st["verts"], "patch_faces": st["faces"], "patch_bbox": st["bbox"],
        "attributes": [a.name for a in terr.data.attributes],
        "georef": {k: scn.get(k) for k in ("SRID", "crs x", "crs y", "latitude", "longitude")},
        "exports": [str(Path(p).relative_to(root)) for p in exports],
        "check_images": [str(obl.relative_to(root)), str(top.relative_to(root))],
    }


def build_scene(job, meta, prm, ext, origin, r, log):
    pa = prm["patch"]
    res = prm["grid"]["resolution_m"]
    s = pa["stride"]
    data = np.load(job["patch"])
    z = data["z"].astype("float64")
    h, w = z.shape
    xs = pa["xmin"] + (np.arange(w) * s + 0.5) * res
    ys = pa["ymax"] - (np.arange(h) * s + 0.5) * res

    tex = bl.copy_texture(job["ortho_tif"], Path(job["textures_dir"]))
    mat = bl.ortho_material("MAT_terrain_ortho", tex)
    mat_conf = bl.attribute_material("MAT_terrace_confidence", "confidence",
                                     [(0.0, (0.85, 0.15, 0.1, 1)), (0.5, (0.95, 0.8, 0.2, 1)),
                                      (1.0, (0.1, 0.7, 0.25, 1))])
    # classes 0 outside, 1 floor, 2 bench, 3 face, 4 water, 5 pit without clear benches
    mat_cls = bl.attribute_material("MAT_terrace_class", "terrace_class_f",
                                    [(0.0, (0.5, 0.5, 0.5, 1)), (0.2, (0.3, 0.6, 1.0, 1)),
                                     (0.4, (1.0, 0.85, 0.25, 1)), (0.6, (0.9, 0.25, 0.15, 1)),
                                     (0.8, (0.15, 0.35, 1.0, 1)), (1.0, (0.75, 0.75, 0.8, 1))])
    mat_conf.use_fake_user = True
    mat_meth = bl.attribute_material("MAT_bench_method", "bench_method_f",
                                     [(0.0, (0.6, 0.6, 0.6, 1)), (0.5, (0.15, 0.75, 0.35, 1)),
                                      (1.0, (1.0, 0.5, 0.1, 1))])
    mat_meth.use_fake_user = True
    mat_cls.use_fake_user = True

    # --- interpreted patch ---
    me = bl.grid_mesh("TERRAIN_terraced", z, xs, ys, ext, origin)
    bl.add_point_attribute(me, "confidence", data["confidence"])
    bl.add_point_attribute(me, "terrace_class", data["classes"], "INT")
    bl.add_point_attribute(me, "terrace_class_f", data["classes"].astype("float32") / 5.0)
    bl.add_point_attribute(me, "dz_interp", data["dz"])
    if "method" in data.files:
        # 0 none, 1 benches confirmed by elevation data, 2 traced from the photo (illustrative)
        bl.add_point_attribute(me, "bench_method", data["method"], "INT")
        bl.add_point_attribute(me, "bench_method_f", data["method"].astype("float32") / 2.0)
    me.materials.append(mat)
    coll = bpy.data.collections.new("10_terraces")
    bpy.context.scene.collection.children.link(coll)
    terr = bpy.data.objects.new("TERRAIN_terraced", me)
    coll.objects.link(terr)
    terr["interpreted"] = True
    terr["note"] = ("INTERPRETED benches inferred from imagery + DEM (not measured). Attribute "
                    "bench_method: 1 = confirmed by elevation data, 2 = traced from the photo "
                    "(illustrative heights). Real coordinates = local XY + (origin_x, origin_y).")
    for k in ("bench_height_m", "face_angle_deg", "coherence_R", "low_confidence",
              "coherence_significance", "method"):
        if prm.get(k) is not None:
            terr[k] = prm[k]
    terr["traced_areas"] = int((prm.get("traced") or {}).get("count", 0))
    terr["bench_floor_levels_m"] = [float(v) for v in prm["bench_floor_levels_m"]]
    terr["run_id"] = prm["run_id"]

    # --- measured context with a hole under the patch ---
    dem = np.load(job["dem_npy"]).astype("float64")
    H, W = dem.shape
    cx_abs = ext["xmin"] + (np.arange(W) + 0.5) * r
    cy_abs = ext["ymax"] - (np.arange(H) + 0.5) * r
    inside_x = (cx_abs > pa["xmin"] + r) & (cx_abs < pa["xmax"] - r)
    inside_y = (cy_abs < pa["ymax"] - r) & (cy_abs > pa["ymin"] + r)
    vin = inside_y[:, None] & inside_x[None, :]
    fin = vin[:-1, :-1] & vin[1:, :-1] & vin[1:, 1:] & vin[:-1, 1:]
    cme = bl.grid_mesh("TERRAIN_base_context", dem, cx_abs, cy_abs, ext, origin, face_mask=~fin)
    cme.materials.append(mat)
    ccoll = bpy.data.collections.new("00_base_context")
    bpy.context.scene.collection.children.link(ccoll)
    ctx = bpy.data.objects.new("TERRAIN_base_context", cme)
    ctx["note"] = "Measured 30 m DEM (copy of the base), hole under the interpreted patch."
    ccoll.objects.link(ctx)
    bl.add_origin_marker(ccoll, meta, max(50.0, r * 3))

    # --- pit outline curves ---
    gj = json.loads(Path(job["outline"]).read_text(encoding="utf-8"))
    zfull = z
    for i, f in enumerate(gj["features"]):
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for j, poly in enumerate(polys):
            ring = np.asarray(poly[0], dtype="float64")
            cols = np.clip(((ring[:, 0] - pa["xmin"]) / (res * s) - 0.5).round().astype(int), 0, w - 1)
            rows = np.clip(((pa["ymax"] - ring[:, 1]) / (res * s) - 0.5).round().astype(int), 0, h - 1)
            zz = zfull[rows, cols] + 1.0
            cu = bpy.data.curves.new(f"PIT_outline_{i}_{j}", "CURVE")
            cu.dimensions = "3D"
            sp = cu.splines.new("POLY")
            sp.points.add(len(ring) - 1)
            pts = np.c_[ring[:, 0] - origin[0], ring[:, 1] - origin[1], zz, np.ones(len(ring))]
            sp.points.foreach_set("co", pts.reshape(-1).astype("float32"))
            sp.use_cyclic_u = True
            cu.bevel_depth = 0.6
            ob = bpy.data.objects.new(f"PIT_outline_{i}_{j}", cu)
            coll.objects.link(ob)

    bl.setup_scene_units_and_georef(meta)
    rig = bpy.data.collections.new("10_terraces_rig")
    bpy.context.scene.collection.children.link(rig)
    wm, hm = pa["xmax"] - pa["xmin"], pa["ymax"] - pa["ymin"]
    cx, cy = (pa["xmin"] + pa["xmax"]) / 2 - origin[0], (pa["ymin"] + pa["ymax"]) / 2 - origin[1]
    _, obl, _, dist = bl.add_rig(rig, cx, cy, wm, hm, float(z.min()), float(z.max()),
                                 top_name="CAM_terraces_top", obl_name="CAM_terraces_oblique")
    bpy.context.scene.camera = obl
    bl.setup_world_and_render(dist * 4)
    log.info("Built TERRAIN_terraced %dx%d and context mesh", h, w)


def main():
    job = json.loads(Path(job_path()).read_text(encoding="utf-8"))
    log = logging.getLogger("quarry.blender07")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(job["log_file"], encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [07 blender] %(levelname)s %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))
    log.addHandler(fh)
    try:
        res = build(job, log)
    except Exception as e:
        log.error("Terrace build failed: %s\n%s", e, traceback.format_exc())
        res = {"error": f"{type(e).__name__}: {e}"}
    Path(job["result"]).write_text(json.dumps(res, indent=2), encoding="utf-8")


main()
