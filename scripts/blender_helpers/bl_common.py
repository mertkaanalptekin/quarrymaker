"""Blender-side helpers shared by stage 04 (base terrain) and stage 07
(interpreted terraces). Runs inside Blender's Python: bpy + numpy only."""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402  (stdlib-only at import time)


def grid_mesh(name, z, xs_abs, ys_abs, ext, origin, face_mask=None):
    """Quad mesh with one vertex per grid sample.

    z: (h, w) elevations, row 0 = north. xs_abs/ys_abs: absolute CRS
    coordinates of the columns/rows. ext: dict xmin..ymax used for UVs (the
    ortho covers exactly this extent). origin: (ox, oy) subtracted from XY.
    face_mask: optional (h-1, w-1) bool, False = drop that quad.
    """
    h, w = z.shape
    X, Y = np.meshgrid(np.asarray(xs_abs) - origin[0], np.asarray(ys_abs) - origin[1])
    co = np.stack([X, Y, z], axis=-1).reshape(-1).astype("float32")
    idx = np.arange(h * w, dtype="int32").reshape(h, w)
    # NW -> SW -> SE -> NE : counter-clockwise from above, normals point +Z
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1)
    if face_mask is not None:
        quads = quads[face_mask]
    quads = quads.reshape(-1)
    nf = len(quads) // 4

    me = bpy.data.meshes.new(name)
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

    XA, YA = np.meshgrid(xs_abs, ys_abs)
    U = (XA - ext["xmin"]) / (ext["xmax"] - ext["xmin"])
    V = (YA - ext["ymin"]) / (ext["ymax"] - ext["ymin"])
    uv_v = np.stack([U, V], axis=-1).reshape(-1, 2)
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", uv_v[quads].reshape(-1).astype("float32"))

    me.update(calc_edges=True)
    me.validate(verbose=False)
    try:
        me.shade_smooth()
    except AttributeError:
        me.polygons.foreach_set("use_smooth", np.ones(nf, dtype=bool))
    return me


def add_point_attribute(me, name, values, kind="FLOAT"):
    a = me.attributes.new(name=name, type=kind, domain="POINT")
    a.data.foreach_set("value", np.asarray(values).reshape(-1).astype(
        "int32" if kind == "INT" else "float32"))
    return a


def ortho_material(name, tex_path: Path, roughness=0.9):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except AttributeError:
        pass
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    img_node = nt.nodes.new("ShaderNodeTexImage")
    img_node.location = (-400, 200)
    img = bpy.data.images.load(str(tex_path), check_existing=True)
    img.name = "ortho_utm"
    img_node.image = img
    img_node.extension = "EXTEND"
    img_node.interpolation = "Cubic"
    nt.links.new(img_node.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    for nm in ("Specular IOR Level", "Specular"):
        if nm in bsdf.inputs:
            bsdf.inputs[nm].default_value = 0.1
            break
    return mat


def attribute_material(name, attribute, stops):
    """Viewport/render material that colours a POINT attribute through a ramp.
    stops: list of (position 0..1, (r, g, b, a))."""
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except AttributeError:
        pass
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = attribute
    attr.location = (-700, 200)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-450, 200)
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = stops[0]
    els[1].position, els[1].color = stops[-1]
    for pos, col in stops[1:-1]:
        e = els.new(pos)
        e.color = col
    nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def copy_texture(src, dst_dir: Path, name="ortho_utm.tif") -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def setup_scene_units_and_georef(meta):
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"
    scn.unit_settings.scale_length = 1.0
    scn.unit_settings.length_unit = "METERS"
    set_blendergis_georef(scn, meta)


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


def add_origin_marker(coll, meta, size):
    marker = bpy.data.objects.new("ORIGIN_marker", None)
    marker.empty_display_type = "ARROWS"
    marker.empty_display_size = size
    marker["origin_x"] = float(meta["origin_x"])
    marker["origin_y"] = float(meta["origin_y"])
    coll.objects.link(marker)
    return marker


def add_rig(coll, cx, cy, wm, hm, zmin, zmax, top_name="CAM_top_ortho", obl_name="CAM_oblique"):
    """Top orthographic camera, oblique perspective camera from SSW, NW sun.
    cx, cy, wm, hm in local coordinates/metres."""
    top_cd = bpy.data.cameras.new(top_name)
    top_cd.type = "ORTHO"
    top_cd.ortho_scale = max(wm, hm)
    top_cd.clip_start = 1.0
    top_cd.clip_end = (zmax - zmin) + 2000.0
    top = bpy.data.objects.new(top_name, top_cd)
    top.location = (cx, cy, zmax + 1000.0)
    top["render_aspect"] = [wm, hm]
    coll.objects.link(top)

    obl_cd = bpy.data.cameras.new(obl_name)
    obl_cd.lens = 35.0
    diag = math.hypot(wm, hm)
    hfov = 2 * math.atan(obl_cd.sensor_width / (2 * obl_cd.lens))
    dist = 1.05 * (diag / 2) / math.tan(hfov / 2)
    az, el = math.radians(202.5), math.radians(35.0)
    target = Vector((cx, cy, (zmin + zmax) / 2))
    pos = target + dist * Vector((math.cos(el) * math.sin(az), math.cos(el) * math.cos(az),
                                  math.sin(el)))
    obl = bpy.data.objects.new(obl_name, obl_cd)
    obl.location = pos
    obl.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
    obl_cd.clip_start = 1.0
    obl_cd.clip_end = dist * 4
    coll.objects.link(obl)

    sun_d = bpy.data.lights.new("SUN_main", "SUN")
    sun_d.energy = 4.0
    sun_d.angle = math.radians(1.0)
    sun = bpy.data.objects.new("SUN_main", sun_d)
    s_az, s_el = math.radians(315.0), math.radians(40.0)
    to_sun = Vector((math.cos(s_el) * math.sin(s_az), math.cos(s_el) * math.cos(s_az),
                     math.sin(s_el)))
    sun.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()
    sun.location = (cx, cy, zmax + 500)
    coll.objects.link(sun)
    return top, obl, sun, dist


def setup_world_and_render(view_clip):
    scn = bpy.context.scene
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
                    space.clip_end = max(100000.0, view_clip)


def mesh_stats(obj) -> dict:
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype="float64")
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return {"verts": len(me.vertices), "faces": len(me.polygons),
            "bbox": [*co.min(axis=0).tolist(), *co.max(axis=0).tolist()]}


def select_only(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def export_obj(obj, path: Path, texture_name: str):
    """OBJ, forward Y / up Z (opens Z-up in Rhino), MTL pointing at texture_name."""
    select_only(obj)
    bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True,
                          forward_axis="Y", up_axis="Z", global_scale=1.0,
                          export_uv=True, export_normals=True, export_materials=True,
                          path_mode="STRIP", apply_modifiers=True)
    mtl = path.with_suffix(".mtl")
    if mtl.exists():
        lines = [ln for ln in mtl.read_text(encoding="utf-8").splitlines() if not ln.startswith("map_Kd")]
        fixed, done = [], False
        for ln in lines:
            fixed.append(ln)
            if ln.startswith("newmtl") and not done:
                fixed.append(f"map_Kd {texture_name}")
                done = True
        mtl.write_text("\n".join(fixed) + "\n", encoding="utf-8")


def export_fbx(obj, path: Path):
    select_only(obj)
    bpy.ops.export_scene.fbx(filepath=str(path), use_selection=True, object_types={"MESH"},
                             axis_forward="Y", axis_up="Z", global_scale=1.0,
                             apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE",
                             path_mode="COPY", embed_textures=True, mesh_smooth_type="FACE")


def render(cam_name, out, size, samples, aspect=None):
    scn = bpy.context.scene
    scn.camera = bpy.data.objects[cam_name]
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = samples
    if aspect:
        w, h = aspect
        if w >= h:
            scn.render.resolution_x, scn.render.resolution_y = size, max(1, round(size * h / w))
        else:
            scn.render.resolution_x, scn.render.resolution_y = max(1, round(size * w / h)), size
    else:
        scn.render.resolution_x, scn.render.resolution_y = size, round(size * 9 / 16)
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)


def save_blend(path: Path):
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
