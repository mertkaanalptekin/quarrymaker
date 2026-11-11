"""Runs inside Blender (headless) for stage 0. Prints one JSON line prefixed
with PROBE_JSON: describing the Blender build and the BlenderGIS add-on."""
import json
import sys

import bpy  # noqa: E402  (import first: puts addon_utils on sys.path)
import addon_utils  # noqa: E402

info = {
    "blender_version": bpy.app.version_string,
    "python_version": sys.version.split()[0],
    "numpy": None,
    "blendergis": None,
    "scripts_dirs": [],
}
try:
    import numpy
    info["numpy"] = numpy.__version__
except Exception as e:  # pragma: no cover
    info["numpy_error"] = str(e)

try:
    info["scripts_dirs"] = [str(p) for p in bpy.utils.script_paths()]
except Exception:
    pass

for mod in addon_utils.modules(refresh=True):
    bl_info = getattr(mod, "bl_info", {}) or {}
    name = bl_info.get("name", "")
    if name.lower() == "blendergis" or mod.__name__.lower().endswith("blendergis"):
        is_enabled, is_loaded = addon_utils.check(mod.__name__)
        info["blendergis"] = {
            "module": mod.__name__,
            "folder": str(getattr(mod, "__file__", "")).rsplit("/", 1)[0].rsplit("\\", 1)[0],
            "version": ".".join(str(v) for v in bl_info.get("version", ())),
            "enabled": bool(is_enabled),
        }
        break

print("PROBE_JSON:" + json.dumps(info))
sys.stdout.flush()
