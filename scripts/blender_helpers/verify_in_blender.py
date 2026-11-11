"""Runs inside Blender for stage 06. Opens quarry_base.blend (read-only use:
the file is never saved), measures the terrain mesh, reads the BlenderGIS
georef properties, renders the top and oblique cameras, and re-imports the
Rhino OBJ (forward Y, up Z) to measure it. Writes a JSON result file."""
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def job_path():
    rest = sys.argv[sys.argv.index("--") + 1:]
    return rest[rest.index("--job") + 1]


def bbox_of(obj):
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    m = np.array(obj.matrix_world)
    co = co @ m[:3, :3].T + m[:3, 3]
    return [*co.min(axis=0).tolist(), *co.max(axis=0).tolist()]


def render(cam_name, out, long_px, samples, aspect=None):
    scn = bpy.context.scene
    scn.camera = bpy.data.objects[cam_name]
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = samples
    if aspect:
        w, h = aspect
        if w >= h:
            scn.render.resolution_x, scn.render.resolution_y = long_px, max(1, round(long_px * h / w))
        else:
            scn.render.resolution_x, scn.render.resolution_y = max(1, round(long_px * w / h)), long_px
    else:
        scn.render.resolution_x, scn.render.resolution_y = long_px, round(long_px * 9 / 16)
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)


def main():
    job = json.loads(Path(job_path()).read_text())
    res = {}
    try:
        bpy.ops.wm.open_mainfile(filepath=job["blend"])
        scn = bpy.context.scene
        terrain = bpy.data.objects["TERRAIN_base"]
        res["blender_version"] = bpy.app.version_string
        res["mesh_bbox"] = bbox_of(terrain)
        res["mesh_verts"] = len(terrain.data.vertices)
        res["mesh_quads"] = sum(1 for p in terrain.data.polygons if p.loop_total == 4)
        res["mesh_faces"] = len(terrain.data.polygons)
        res["dem_stride"] = int(terrain.get("dem_stride", 1))
        res["georef"] = {k: scn.get(k) for k in ("SRID", "crs x", "crs y", "latitude", "longitude")}
        res["units"] = {"system": scn.unit_settings.system,
                        "scale_length": scn.unit_settings.scale_length,
                        "length_unit": scn.unit_settings.length_unit}
        res["collections"] = {c.name: [o.name for o in c.objects] for c in bpy.data.collections}
        img = next((i for i in bpy.data.images if i.name == "ortho_utm"), None)
        res["texture"] = {"filepath": img.filepath if img else None,
                          "size": list(img.size) if img else None}
        top = bpy.data.objects["CAM_top_ortho"]
        aspect = list(top.get("render_aspect", [16, 9]))
        render("CAM_top_ortho", job["render_top"], job["render_px"], job["samples"], aspect)
        render("CAM_oblique", job["render_oblique"], job["render_px"], job["samples"])

        res["obj_reimport"] = None
        if job.get("obj") and Path(job["obj"]).exists():
            before = set(bpy.data.objects)
            bpy.ops.wm.obj_import(filepath=job["obj"], forward_axis="Y", up_axis="Z")
            new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
            if new:
                res["obj_reimport"] = {"objects": [o.name for o in new], "bbox": bbox_of(new[0])}
    except Exception as e:
        import traceback
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
    Path(job["result"]).write_text(json.dumps(res, indent=2))


main()
