"""Stage 0 - environment check.

Verifies Python packages, the Blender executable, the BlenderGIS add-on, the
OpenTopography API key and the config fields the pipeline must not guess.
Prints a clear list of anything missing. Installs nothing.

Exit code 0 = ready, 1 = something required is missing.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REQUIRED_PKGS = {  # import name -> pip name
    "rasterio": "rasterio",
    "numpy": "numpy",
    "pyproj": "pyproj",
    "shapely": "shapely",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "requests": "requests",
}
GDAL_TOOLS = ["gdalinfo", "gdalwarp", "gdal_translate"]


def probe_blender(blender: str) -> dict:
    probe = Path(__file__).resolve().parent / "blender_helpers" / "probe_env.py"
    cmd = [blender, "--background", "--python", str(probe)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        return {"error": f"Blender executable not found at {blender}"}
    except subprocess.TimeoutExpired:
        return {"error": "Blender did not finish the probe within 180 s"}
    for line in out.stdout.splitlines():
        if line.startswith("PROBE_JSON:"):
            return json.loads(line[len("PROBE_JSON:"):])
    return {"error": "Blender ran but returned no probe output.\n"
                     + (out.stderr or out.stdout)[-1500:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    args = ap.parse_args(argv)

    missing: list[str] = []   # blocking
    warnings: list[str] = []  # non-blocking
    report: dict = {}

    print("== Stage 0: environment check ==")

    # Python ------------------------------------------------------------
    pyver = sys.version_info
    report["python"] = sys.version.split()[0]
    print(f"Python {report['python']} ({sys.executable})")
    if pyver < (3, 10):
        missing.append(f"Python 3.10+ is required (found {report['python']}).")

    report["packages"] = {}
    for mod, pipname in REQUIRED_PKGS.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            report["packages"][pipname] = ver
            print(f"  ok  {pipname} {ver}")
        except Exception:
            report["packages"][pipname] = None
            missing.append(f"Python package '{pipname}' is not installed "
                           f"(install with: pip install {pipname}).")
            print(f"  --  {pipname} MISSING")

    gdal = {t: shutil.which(t) for t in GDAL_TOOLS}
    report["gdal_cli"] = gdal
    if not any(gdal.values()):
        print("  (optional) GDAL CLI tools not found - not needed.")

    # Config ------------------------------------------------------------
    cfg = None
    try:
        cfg = common.load_config(args.config)
        print(f"Config: {cfg['_config_path']}")
    except common.PipelineError as e:
        missing.append(str(e))
    except ImportError:
        missing.append("Cannot read config.yaml without pyyaml.")

    if cfg is not None:
        try:
            aoi = common.require_aoi(cfg)
            print(f"  AOI  W {aoi['west']}  S {aoi['south']}  E {aoi['east']}  N {aoi['north']}"
                  f"  (+{aoi['buffer_m']:.0f} m buffer)")
        except common.PipelineError as e:
            missing.append(str(e))
        try:
            print(f"  CRS  {common.crs_string(cfg)}")
        except common.PipelineError as e:
            missing.append(str(e))
        mode = (cfg.get("imagery") or {}).get("mode", "esri_tiles")
        if mode == "user_geotiff":
            ug = (cfg.get("imagery") or {}).get("user_geotiff")
            if not ug:
                missing.append("imagery.mode is user_geotiff but imagery.user_geotiff is empty.")
            elif not (Path(cfg["_root"]) / ug).exists() and not Path(ug).exists():
                missing.append(f"imagery.user_geotiff not found: {ug}")
        elif mode not in ("esri_tiles", "none"):
            missing.append(f"imagery.mode must be esri_tiles | user_geotiff | none (got {mode!r}).")

    # Blender -----------------------------------------------------------
    blender, where = common.find_blender(cfg or {})
    if not blender:
        missing.append("Blender path is not set. Set blender.path in config.yaml or the "
                       "BLENDER_PATH environment variable.")
        print("Blender: NOT CONFIGURED")
    else:
        print(f"Blender: {blender}  (from {where})")
        info = probe_blender(blender)
        report["blender"] = {"path": blender, "found_via": where, **info}
        if "error" in info:
            missing.append(f"Blender check failed: {info['error']}")
        else:
            print(f"  version {info['blender_version']}, Python {info['python_version']}, "
                  f"numpy {info.get('numpy')}")
            if not info.get("numpy"):
                missing.append("Blender's bundled Python has no numpy (needed by stage 04).")
            bg = info.get("blendergis")
            if not bg:
                warnings.append(
                    "BlenderGIS add-on not found in this Blender. The terrain is still built "
                    "with BlenderGIS-compatible georef properties, but you need the add-on to "
                    "use basemaps/imports on it. Install from https://github.com/domlysz/BlenderGIS")
                print("  BlenderGIS: NOT INSTALLED")
            else:
                print(f"  BlenderGIS {bg['version']} at {bg['folder']} "
                      f"({'enabled' if bg['enabled'] else 'installed but not enabled'})")
                if not bg["enabled"]:
                    warnings.append("BlenderGIS is installed but not enabled "
                                    "(Edit > Preferences > Add-ons).")

    # API key -----------------------------------------------------------
    has_key = bool(os.environ.get("OPENTOPO_API_KEY"))
    report["opentopo_api_key_set"] = has_key
    print(f"OPENTOPO_API_KEY: {'set' if has_key else 'NOT SET'}")
    if not has_key:
        cached = _dem_cached(cfg)
        if cached:
            warnings.append(f"OPENTOPO_API_KEY is not set, but the DEM is cached ({cached}).")
        else:
            missing.append("OPENTOPO_API_KEY is not set. Get a free key at "
                           "https://portal.opentopography.org (My Account > Request API key) "
                           "and export it in your shell; never put it in a file.")

    # Report ------------------------------------------------------------
    report["missing"] = missing
    report["warnings"] = warnings
    if cfg is not None:
        common.write_json(Path(cfg["_root"]) / "logs" / "env_report.json", report)

    print()
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  ! {w}")
    if missing:
        print("MISSING - the pipeline cannot run until these are fixed:")
        for i, m in enumerate(missing, 1):
            print(f"  {i}. {m}")
        return 1
    print("Environment OK.")
    return 0


def _dem_cached(cfg) -> str | None:
    if not cfg:
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fetch_dem", Path(__file__).resolve().parent / "01_fetch_dem.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        p = mod.dem_cache_path(cfg)
        return str(p) if p.exists() else None
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
