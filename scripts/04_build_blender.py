"""Stage 04 - build the Blender base terrain (headless).

Two modes, same file:
  * launcher (plain Python):  python scripts/04_build_blender.py [--config ..]
      writes a job file and runs
      blender --background --factory-startup --python 04_build_blender.py -- --job <job.json>
  * builder (inside Blender): reads dem_utm.npy + meta.json (Blender's Python
      has numpy but not rasterio/yaml), builds the mesh with foreach_set, one
      vertex per DEM cell, quads, UVs from the extent, ortho material, metric
      units, BlenderGIS georef properties, cameras + sun, saves the .blend and
      exports OBJ/FBX for Rhino.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (stdlib-only at import time; safe inside Blender)

BLEND_NAME = "quarry_base.blend"
OBJ_NAME = "quarry_base.obj"
FBX_NAME = "quarry_base.fbx"
TEX_NAME = "quarry_base_texture.jpg"


# ======================================================================
# Launcher (plain Python)
# ======================================================================

def launcher(argv=None) -> int:
    args, cfg, paths, log = common.stage_context(__doc__, "04", argv)
    if not args.force and common.stage_up_to_date(paths, cfg, "04"):
        log.info("Stage 04 up to date - skipping.")
        return 0
    meta_path = paths.base / "meta.json"
    npy = paths.processed / "dem_utm.npy"
    if not meta_path.exists() or not npy.exists():
        raise common.PipelineError("Stage 03 outputs are missing; run stage 03 first.")
    meta = common.read_json(meta_path)
    proc_meta = common.read_json(paths.processed / "meta.json")
    if proc_meta.get("config_hash") != meta.get("config_hash"):
        raise common.PipelineError("data/processed belongs to a different config; re-run stage 03.")

    blender, where = common.find_blender(cfg)
    if not blender:
        raise common.PipelineError("Blender path is not set (config blender.path or BLENDER_PATH).")
    bcfg = cfg.get("blender") or {}
    blend = paths.base / BLEND_NAME
    job = {
        "run_id": paths.run_id,
        "log_file": str(paths.log_file),
        "meta": str(meta_path),
        "dem_npy": str(npy),
        "ortho_tif": str(paths.processed / "ortho_utm.tif"),
        "ortho_jpg": str(paths.processed / "ortho_utm.jpg"),
        "blend": str(blend),
        "textures_dir": str(paths.base / "textures"),
        "rhino_dir": str(paths.rhino),
        "checks_dir": str(paths.checks),
        "formats": [f.lower() for f in ((cfg.get("rhino") or {}).get("formats") or ["obj", "fbx"])],
        "decimate_ratio": float(bcfg.get("mesh_decimate_ratio", 1.0)),
        "mode": "export_only" if blend.exists() else "build",
        "result": str(paths.checks / "blender_04_result.json"),
    }
    if job["mode"] == "export_only":
        log.info("%s already exists (write-once): it will be opened, not rebuilt; "
                 "only missing exports are written.", blend)
    job_path = paths.checks / "blender_job_04.json"
    common.write_json(job_path, job)
    rc = run_blender(blender, Path(__file__).resolve(), job_path, log)
    if rc != 0 or not Path(job["result"]).exists():
        raise common.PipelineError(f"Blender build failed (exit code {rc}); see {paths.log_file}")
    res = common.read_json(job["result"])
    if res.get("error"):
        raise common.PipelineError("Blender build failed: " + res["error"])

    outs = [blend] + [paths.root / p for p in res["exports"]]
    summary = [
        f"Mesh TERRAIN_base: {res['verts']} vertices, {res['faces']} quads "
        f"({res['grid'][1]} x {res['grid'][0]}, stride {res['stride']})",
        f"Local bbox X {res['bbox'][0]:.2f}..{res['bbox'][3]:.2f}  Y {res['bbox'][1]:.2f}.."
        f"{res['bbox'][4]:.2f}  Z {res['bbox'][2]:.2f}..{res['bbox'][5]:.2f} m",
        f"BlenderGIS georef: SRID={res['georef']['SRID']}, crs x={res['georef']['crs x']}, "
        f"crs y={res['georef']['crs y']}",
        f"Saved {common.rel(paths, blend)}; exports: {', '.join(res['exports'])}",
    ]
    common.finish_stage(paths, cfg, "04", summary, outs,
                        [paths.root / p for p in res["check_images"]], log,
                        extra={"blender": res.get("blender_version")})
    return 0


def run_blender(blender: str, script: Path, job_path: Path, log) -> int:
    cmd = [blender, "--background", "--factory-startup", "--python", str(script),
           "--", "--job", str(job_path)]
    log.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line = line.rstrip()
        if line and not line.startswith("Fra:"):
            log.info("[blender] %s", line)
    return proc.wait()


# ======================================================================
# Builder (inside Blender)
# ======================================================================

def blender_main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    log = logging.getLogger("quarry.blender")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(job["log_file"], encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [04 blender] %(levelname)s %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))
    log.addHandler(fh)
    try:
        res = build(job, log)
    except Exception as e:  # report to the launcher
        import traceback
        log.error("Build failed: %s\n%s", e, traceback.format_exc())
        res = {"error": f"{type(e).__name__}: {e}"}
        Path(job["result"]).write_text(json.dumps(res, indent=2))
        sys.exit(1)
    Path(job["result"]).write_text(json.dumps(res, indent=2))


def build(job: dict, log) -> dict:
    import bpy

    meta = json.loads(Path(job["meta"]).read_text())
    root = Path(job["checks_dir"]).parent.parent
    blend = Path(job["blend"])
    rhino = Path(job["rhino_dir"])
    rhino.mkdir(parents=True, exist_ok=True)

    if job["mode"] == "build":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.preferences.filepaths.save_version = 0
        obj, stats = build_scene(job, meta, log)
        Path(job["textures_dir"]).mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        bpy.ops.file.make_paths_relative()
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        log.info("Saved %s", blend)
    else:
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        obj = bpy.data.objects["TERRAIN_base"]
        stats = mesh_stats(obj)
        stats["stride"] = obj.get("dem_stride", 1)
        stats["grid"] = list(obj.get("grid_rows_cols", [0, 0]))

    exports = export_rhino(job, meta, obj, rhino, log)
    preview = Path(job["checks_dir"]) / "04_blender_preview.png"
    render_preview(preview, log)

    scn = bpy.context.scene
    return {
        "blender_version": bpy.app.version_string,
        **stats,
        "georef": {k: scn.get(k) for k in ("SRID", "crs x", "crs y", "latitude", "longitude")},
        "exports": [str(p.relative_to(root)) for p in exports],
        "check_images": [str(preview.relative_to(root))],
    }


def build_scene(job, meta, log):
    import bpy
    import numpy as np
    from mathutils import Vector

    ext = meta["extent"]
    r = float(meta["grid"]["cell_size_m"])
    ox, oy = float(meta["origin_x"]), float(meta["origin_y"])
    dem = np.load(job["dem_npy"]).astype("float64")
    H, W = dem.shape
    if (H, W) != (meta["grid"]["rows"], meta["grid"]["cols"]):
        raise RuntimeError(f"dem_utm.npy shape {dem.shape} does not match meta.json grid")

    ratio = float(job.get("decimate_ratio", 1.0))
    stride = 1 if ratio >= 1.0 else max(1, int(round(1.0 / np.sqrt(ratio))))
    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    if stride > 1:
        log.warning("mesh_decimate_ratio=%.3f -> taking every %d-th DEM cell (no interpolation)",
                    ratio, stride)
    z = dem[np.ix_(rows, cols)]
    h, w = z.shape

    # One vertex per DEM cell centre; XY = real coordinates minus the origin.
    xs_abs = ext["xmin"] + (cols + 0.5) * r
    ys_abs = ext["ymax"] - (rows + 0.5) * r
    X, Y = np.meshgrid(xs_abs - ox, ys_abs - oy)
    co = np.stack([X, Y, z], axis=-1).reshape(-1).astype("float32")

    idx = np.arange(h * w, dtype="int32").reshape(h, w)
    # NW -> SW -> SE -> NE : counter-clockwise from above, normals point +Z
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1)
    nf = (h - 1) * (w - 1)

    me = bpy.data.meshes.new("TERRAIN_base")
    me.vertices.add(h * w)
    me.vertices.foreach_set("co", co)
    me.loops.add(nf * 4)
    me.loops.foreach_set("vertex_index", quads)
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.arange(0, nf * 4, 4, dtype="int32"))
    try:  # Blender < 4.0 needs loop_total; newer versions derive it
        me.polygons.foreach_set("loop_total", np.full(nf, 4, dtype="int32"))
    except (AttributeError, RuntimeError, TypeError):
        pass

    # UVs from the extent bounds (the ortho covers exactly the extent).
    U = (np.meshgrid(xs_abs, ys_abs)[0] - ext["xmin"]) / (ext["xmax"] - ext["xmin"])
    V = (np.meshgrid(xs_abs, ys_abs)[1] - ext["ymin"]) / (ext["ymax"] - ext["ymin"])
    uv_v = np.stack([U, V], axis=-1).reshape(-1, 2)
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", uv_v[quads].reshape(-1).astype("float32"))

    me.update(calc_edges=True)
    if me.validate(verbose=False):
        log.warning("mesh.validate() corrected something - inspect the mesh")
    try:
        me.shade_smooth()
    except AttributeError:
        me.polygons.foreach_set("use_smooth", np.ones(nf, dtype=bool))

    coll = bpy.data.collections.new("00_base")
    bpy.context.scene.collection.children.link(coll)
    obj = bpy.data.objects.new("TERRAIN_base", me)
    coll.objects.link(obj)
    obj["run_id"] = job["run_id"]
    obj["crs"] = meta["crs"]
    obj["origin_x"] = ox
    obj["origin_y"] = oy
    obj["dem_source"] = meta["dem"]["source"]
    obj["dem_stride"] = stride
    obj["grid_rows_cols"] = [h, w]
    obj["note"] = "Measured DEM, unedited. Real coordinates = local XY + (origin_x, origin_y)."

    marker = bpy.data.objects.new("ORIGIN_marker", None)
    marker.empty_display_type = "ARROWS"
    marker.empty_display_size = max(50.0, r * 3)
    marker["origin_x"] = ox
    marker["origin_y"] = oy
    coll.objects.link(marker)

    # Material: ortho as Base Color, high roughness.
    tex_dir = Path(job["textures_dir"])
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex = tex_dir / "ortho_utm.tif"
    if not tex.exists():
        shutil.copy2(job["ortho_tif"], tex)
    mat = bpy.data.materials.new("MAT_terrain_ortho")
    try:
        mat.use_nodes = True
    except AttributeError:
        pass
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    img_node = nt.nodes.new("ShaderNodeTexImage")
    img_node.location = (-400, 200)
    img_node.image = bpy.data.images.load(str(tex))
    img_node.image.name = "ortho_utm"
    img_node.extension = "EXTEND"
    img_node.interpolation = "Cubic"
    nt.links.new(img_node.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    for name in ("Specular IOR Level", "Specular"):
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = 0.1
            break
    me.materials.append(mat)

    # Scene units & BlenderGIS georeferencing.
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"
    scn.unit_settings.scale_length = 1.0
    scn.unit_settings.length_unit = "METERS"
    set_blendergis_georef(scn, meta)

    # Camera rig + sun.
    zmin, zmax = float(z.min()), float(z.max())
    wm, hm = ext["xmax"] - ext["xmin"], ext["ymax"] - ext["ymin"]
    cx, cy = (ext["xmin"] + ext["xmax"]) / 2 - ox, (ext["ymin"] + ext["ymax"]) / 2 - oy
    rig = bpy.data.collections.new("00_base_rig")
    scn.collection.children.link(rig)

    top_cd = bpy.data.cameras.new("CAM_top_ortho")
    top_cd.type = "ORTHO"
    top_cd.ortho_scale = max(wm, hm)
    top_cd.clip_start = 1.0
    top_cd.clip_end = (zmax - zmin) + 2000.0
    top = bpy.data.objects.new("CAM_top_ortho", top_cd)
    top.location = (cx, cy, zmax + 1000.0)
    top["render_aspect"] = [wm, hm]
    rig.objects.link(top)

    import math
    obl_cd = bpy.data.cameras.new("CAM_oblique")
    obl_cd.lens = 35.0
    diag = math.hypot(wm, hm)
    hfov = 2 * math.atan(obl_cd.sensor_width / (2 * obl_cd.lens))
    dist = 1.05 * (diag / 2) / math.tan(hfov / 2)
    az, el = math.radians(202.5), math.radians(35.0)  # from SSW, 35 deg above horizon
    target = Vector((cx, cy, (zmin + zmax) / 2))
    pos = target + dist * Vector((math.cos(el) * math.sin(az), math.cos(el) * math.cos(az),
                                  math.sin(el)))
    obl = bpy.data.objects.new("CAM_oblique", obl_cd)
    obl.location = pos
    obl.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
    obl_cd.clip_start = 1.0
    obl_cd.clip_end = dist * 4
    rig.objects.link(obl)

    sun_d = bpy.data.lights.new("SUN_main", "SUN")
    sun_d.energy = 4.0
    sun_d.angle = math.radians(1.0)
    sun = bpy.data.objects.new("SUN_main", sun_d)
    s_az, s_el = math.radians(315.0), math.radians(40.0)  # cartographic NW light
    to_sun = Vector((math.cos(s_el) * math.sin(s_az), math.cos(s_el) * math.cos(s_az),
                     math.sin(s_el)))
    sun.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()
    sun.location = (cx, cy, zmax + 500)
    rig.objects.link(sun)
    scn.camera = obl

    world = bpy.data.worlds.new("WORLD_sky")
    scn.world = world
    try:
        world.use_nodes = True
    except AttributeError:
        pass
    bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg:
        bg.inputs["Color"].default_value = (0.55, 0.62, 0.72, 1.0)
        bg.inputs["Strength"].default_value = 0.6

    scn.render.engine = "CYCLES"
    scn.cycles.samples = 64
    scn.render.resolution_x = 1920
    scn.render.resolution_y = 1080
    scn.render.resolution_percentage = 100

    # Make the terrain visible when the file is opened (default clip_end is 1 km).
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_start = 1.0
                    space.clip_end = max(100000.0, dist * 4)

    stats = mesh_stats(obj)
    stats["stride"] = stride
    stats["grid"] = [h, w]
    log.info("Built TERRAIN_base: %d verts, %d quads, bbox %s", stats["verts"], stats["faces"],
             stats["bbox"])
    return obj, stats


def set_blendergis_georef(scn, meta):
    """Write the georef exactly as BlenderGIS's GeoScene does (geoscene.py, class SK):
    SRID (CRS string 'EPSG:xxxx'), 'crs x'/'crs y' (scene origin in CRS units),
    'longitude'/'latitude' (same origin in WGS84). Scale is left unset (BlenderGIS
    default 1 = true scale)."""
    K = common.BGIS_KEYS
    scn[K["crs"]] = str(meta["crs"])
    scn[K["crsx"]] = float(meta["origin_x"])
    scn[K["crsy"]] = float(meta["origin_y"])
    scn[K["lon"]] = float(meta["origin_lon"])
    scn[K["lat"]] = float(meta["origin_lat"])


def mesh_stats(obj) -> dict:
    import numpy as np
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype="float64")
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return {"verts": len(me.vertices), "faces": len(me.polygons),
            "bbox": [*co.min(axis=0).tolist(), *co.max(axis=0).tolist()]}


def export_rhino(job, meta, obj, rhino: Path, log) -> list[Path]:
    import bpy

    out = []
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    tex = rhino / TEX_NAME
    if not tex.exists():
        shutil.copy2(job["ortho_jpg"], tex)
    out.append(tex)

    if "obj" in job["formats"]:
        p = rhino / OBJ_NAME
        if not p.exists():
            bpy.ops.wm.obj_export(filepath=str(p), export_selected_objects=True,
                                  forward_axis="Y", up_axis="Z", global_scale=1.0,
                                  export_uv=True, export_normals=True, export_materials=True,
                                  path_mode="STRIP", apply_modifiers=True)
            mtl = p.with_suffix(".mtl")
            if mtl.exists():
                lines = [ln for ln in mtl.read_text().splitlines() if not ln.startswith("map_Kd")]
                fixed, done = [], False
                for ln in lines:
                    fixed.append(ln)
                    if ln.startswith("newmtl") and not done:
                        fixed.append(f"map_Kd {TEX_NAME}")
                        done = True
                mtl.write_text("\n".join(fixed) + "\n")
            log.info("Exported %s (forward Y, up Z)", p)
        out += [p, p.with_suffix(".mtl")]

    if "fbx" in job["formats"]:
        p = rhino / FBX_NAME
        if not p.exists():
            bpy.ops.export_scene.fbx(filepath=str(p), use_selection=True, object_types={"MESH"},
                                     axis_forward="Y", axis_up="Z", global_scale=1.0,
                                     apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE",
                                     path_mode="COPY", embed_textures=True, mesh_smooth_type="FACE")
            log.info("Exported %s", p)
        out.append(p)

    notes = rhino / "rhino_import_notes.md"
    if not notes.exists():
        notes.write_text(rhino_notes(meta, obj))
    out.append(notes)
    return out


def rhino_notes(meta, obj) -> str:
    e = meta["extent"]
    ox, oy = meta["origin_x"], meta["origin_y"]
    return f"""# Rhino import notes - run {meta['run_id']}

**Set the Rhino document units to Meters before importing.** All files are in metres,
Z up, at true scale (1 unit = 1 m).

## Coordinates
- CRS: {meta['crs']} (projected, metres).
- The mesh is stored in **local coordinates**: real (projected) coordinates minus the origin.
  - origin_x = **{ox:.1f}**
  - origin_y = **{oy:.1f}**
  - Z is the real elevation in metres ({meta['dem']['vertical_datum']}); no Z offset.
- To convert local to real: `X_real = x + {ox:.1f}`, `Y_real = y + {oy:.1f}`, `Z_real = z`.
- To place the mesh at real coordinates in Rhino: `Move` from `0,0,0` to `{ox:.1f},{oy:.1f},0`
  (large coordinates can cause display jitter in Rhino; keeping local coordinates is recommended).
- Local extent of the data: X {e['xmin'] - ox:.1f} .. {e['xmax'] - ox:.1f}, Y {e['ymin'] - oy:.1f} .. {e['ymax'] - oy:.1f}
  (mesh vertices sit at DEM cell centres, so the mesh is half a cell smaller on each side).

## Files
- `quarry_base.obj` + `quarry_base.mtl` + `{TEX_NAME}`: quad mesh with UVs and the ortho texture.
  Written with forward axis Y and up axis Z. In Rhino's OBJ import options, **turn off
  "Map OBJ Y to Rhino Z"** (coordinates are already Z-up).
- `quarry_base.fbx`: same mesh, texture embedded. Check the unit prompt on import (meters).

## Data
- DEM: {meta['dem']['source']}, {meta['grid']['cell_size_m']} m cells ({meta['grid']['cols']} x {meta['grid']['rows']}).
- Elevation range: {meta['elevation']['min']:.2f} .. {meta['elevation']['max']:.2f} m.
- Imagery: {meta['imagery'].get('attribution')}
- This is measured terrain (no benches, smoothing or erosion added).
"""


def render_preview(path: Path, log):
    import bpy
    scn = bpy.context.scene
    scn.camera = bpy.data.objects["CAM_oblique"]
    scn.render.engine = "CYCLES"
    scn.cycles.samples = 16
    scn.cycles.device = "CPU"
    scn.render.resolution_x, scn.render.resolution_y = 800, 450
    scn.render.image_settings.file_format = "PNG"
    scn.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    log.info("Preview render -> %s", path)


# ======================================================================

def _job_arg() -> str | None:
    if "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1:]
        if "--job" in rest:
            return rest[rest.index("--job") + 1]
    return None


if __name__ == "__main__":
    job_arg = _job_arg()
    if job_arg:
        blender_main(job_arg)
    else:
        common.run_main(launcher, "04")
