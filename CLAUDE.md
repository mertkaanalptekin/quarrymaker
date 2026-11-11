# Quarry Research Project — Base Terrain Pipeline

## Purpose
Build an automated, reproducible pipeline that creates the **base terrain** of a quarry site for an architectural research project.

Inputs: an area of interest, a 30 m DEM from OpenTopography, and satellite imagery.

Outputs:
- a georeferenced, true-scale Blender terrain that is compatible with the BlenderGIS add-on;
- a heightmap package for Gaea;
- mesh exports for Rhino.

Later stages read from this base: Gaea terraforming, precipitation simulations, architectural modeling and renders. Base terrain is the single source of truth for scale, coordinates and elevation.

Scope of this task: **base terrain only.** Do not build Gaea graphs, simulations or architecture yet. Only produce the files those stages will consume.

---

## Ground rules
1. **Scripts, not clicks.** All work runs through Python scripts and headless Blender (`blender --background --python ...`). Use Blender or Rhino MCP only to inspect results or make small interactive checks, never to build.
2. **One coordinate system.**
   - Use the projected CRS from `config.yaml` (default EPSG:32635, UTM 35N). Units are meters and Z is up.
   - Every output records the same local origin (`origin_x`, `origin_y`) in `meta.json`.
3. **Never overwrite base outputs.**
   - `terrain/00_base/` is write-once per run ID.
   - A new run writes to a new versioned folder, e.g. `terrain/00_base/v002/`.
4. **Keep measured data separate from interpreted data.**
   - The base terrain is the DEM exactly as measured.
   - Do not add benches, smoothing or erosion here.
   - Resampling is allowed only as configured, and must be logged.
5. **Ask, don't guess.**
   - If the bounding box, CRS, Blender path or API key is missing, stop and ask the user.
   - Never invent coordinates.
6. **No secrets in files.** Read the OpenTopography key from the environment variable `OPENTOPO_API_KEY`.
7. **Checkpoint after each stage.** Each stage ends by writing its check image(s) and a short summary. After each stage, stop and show the user the check images before moving on, unless they run the pipeline with `--auto`.

---

## Environment check (stage 0)
Write `scripts/check_env.py`. It verifies:
- Python 3.10+ with `rasterio`, `numpy`, `pyproj`, `shapely`, `Pillow`, `pyyaml` and `requests`. GDAL CLI tools are optional.
- The Blender executable, from `config.yaml` or `BLENDER_PATH`. Report its version.
- That the BlenderGIS add-on is installed in that Blender. Find the add-on folder and report its version.
- That `OPENTOPO_API_KEY` is set.

Report anything missing as a clear list. If an install needs approval, ask the user before running it.

**BlenderGIS compatibility.** Before writing any Blender script, read BlenderGIS's geoscene source code. Confirm how it stores georeferencing on the scene: the CRS and the scene origin, normally the scene custom properties `crs`, `crsx` and `crsy`. The terrain built in stage 4 must set the same properties, so the user can keep using BlenderGIS tools (basemaps, extra imports) on this file and everything lands in the right place.

---

## Project structure
```
quarry_project/
  CLAUDE.md
  config.yaml
  run_pipeline.py
  scripts/
    check_env.py
    01_fetch_dem.py
    02_fetch_imagery.py
    03_preprocess.py
    04_build_blender.py      # runs inside Blender
    05_export_gaea.py
    06_verify.py
  data/
    raw/            # untouched downloads, cached
    processed/      # reprojected / clipped rasters
    reference/      # user's Google Earth image, notes
  terrain/
    00_base/vNNN/   # .blend, exports, meta.json, report
  exchange/
    gaea/vNNN/      # heightmap package for Gaea
    rhino/vNNN/     # meshes + texture for Rhino
  checks/vNNN/      # check images
  logs/
```

## config.yaml (create with placeholders and ask the user to fill in)
```yaml
run_id: auto            # auto-increments v001, v002...
aoi:
  # WGS84 lon/lat of the area of interest; the user supplies these
  west: null
  south: null
  east: null
  north: null
  buffer_m: 500         # extra margin so edges don't clip the site
crs_epsg: 32635
dem:
  source: COP30         # OpenTopography demtype; COP30 preferred over SRTMGL1 (SRTM is from 2000, before most of the excavation)
  target_resolution_m: 30   # keep native; set lower (e.g., 10) only for smoother render meshes, logged as resampled
  resampling: bilinear
imagery:
  mode: esri_tiles      # esri_tiles | user_geotiff | none
  zoom: 17
  user_geotiff: null
blender:
  path: null
  mesh_decimate_ratio: 1.0   # 1.0 = one vertex per DEM cell
gaea:
  size_px: 2048         # square heightmap
  bit_depth: 16
rhino:
  formats: [obj, fbx]
```

---

## Stages

### 01 — Fetch DEM (`01_fetch_dem.py`)
- Request the AOI plus the buffer from the OpenTopography Global DEM API: the `globaldem` endpoint, with `demtype` taken from the config and `outputFormat=GTiff`.
- Save to `data/raw/dem_<source>_<bbox-hash>.tif`.
- **Cache:** skip the download if the file already exists.
- Log the source, request time and bbox. Fail clearly on API errors (invalid key, bbox too large).
- Check image: a quick hillshade PNG of the raw DEM.

### 02 — Fetch imagery (`02_fetch_imagery.py`)
- `esri_tiles` mode:
  - Download World Imagery XYZ tiles that cover the buffered AOI at the configured zoom.
  - Mosaic them and write a georeferenced GeoTIFF (Web Mercator). Reprojecting happens in stage 03.
  - Throttle requests, cache tiles, and put an attribution line in `meta.json`. Check the service's usage terms and tell the user what they are.
- `user_geotiff` mode: copy the user's georeferenced image instead.
- The user's Google Earth image in `data/reference/` has no georeferencing. Use it only for visual comparison in stage 06.
- Check image: a downscaled preview of the mosaic.

### 03 — Preprocess (`03_preprocess.py`)
- Reproject the DEM and the imagery to `crs_epsg`, then clip both to the same extent with the same pixel alignment. The imagery keeps its own finer resolution, snapped to the DEM's grid bounds.
- Fill DEM nodata by interpolation, and report how many cells were filled.
- Compute the **local origin**: the lower-left corner of the extent, rounded to the nearest 100 m.
- Write `data/processed/dem_utm.tif`, `data/processed/ortho_utm.tif` and `meta.json`. `meta.json` holds:
  - CRS, extent and origin;
  - DEM source, resolution and capture-date note;
  - minimum and maximum elevation;
  - resampling settings;
  - imagery source and attribution;
  - the run ID and a hash of the config.
- Check image: the hillshade blended over the ortho at 50% opacity, to confirm alignment by eye. Also produce an elevation histogram.

### 04 — Build Blender terrain (`04_build_blender.py`, headless)
- Read `dem_utm.tif` with numpy inside Blender's Python. If rasterio is not available in Blender's Python, have stage 03 also write a `.npy` file plus `meta.json` for this step to read.
- Build the mesh directly from the array with `foreach_set`. Do not use a displace modifier or subdivision.
  - One vertex per DEM cell, positioned at real XY minus the origin, with Z at the real elevation in meters.
  - Faces are quads.
- Use UV coordinates mapped from the extent bounds. Create a material that uses `ortho_utm.tif` as Base Color, with roughness set high.
- Set scene units to metric with scale 1.0. Set the BlenderGIS georef scene properties (CRS and origin) found in stage 0.
- Name objects clearly: `TERRAIN_base`, plus an empty called `ORIGIN_marker` at the origin. Put them in a collection called `00_base`.
- Add a camera rig: one top orthographic camera and one oblique perspective camera. Add a sun lamp.
- Save to `terrain/00_base/vNNN/quarry_base.blend`.
- Exports to `exchange/rhino/vNNN/`:
  - OBJ with forward axis Y and up axis Z, so it opens Z-up in Rhino. Write the texture next to it.
  - FBX.
  - `rhino_import_notes.md`, which records the origin offset and states that Rhino units must be meters.

### 05 — Gaea package (`05_export_gaea.py`)
- From `dem_utm.tif`, make a **square** heightmap of `size_px`. Pad it with edge values rather than stretching it, and record the padding in pixels.
- Normalize elevations to the full 16-bit range: `(z - zmin) / (zmax - zmin) * 65535`. Write a 16-bit PNG and an EXR (32-bit float, real meters).
- Write the matching ortho as a color map, using the same crop and padding.
- Write `gaea_meta.json`. It holds:
  - zmin, zmax, pixel size in meters, and real width and height in meters;
  - the padding and the origin;
  - the exact formula to convert a Gaea output back to real elevation.
- Write `README_gaea.md`. It explains how to set the terrain size and height in Gaea so the scale is right (width in meters, height = zmax − zmin).
- Write `scripts/reimport_from_gaea.py` but leave it as a stub that only defines the function signature. The Gaea stage will fill it in later.

### 06 — Verify (`06_verify.py`)
These checks are automated. Each one must pass, or the problem must be reported clearly:
- The mesh bounding box matches the extent within half a cell.
- The mesh minimum and maximum Z match the DEM minimum and maximum within 0.01 m.
- The Blender scene's georef properties equal the CRS and origin in `meta.json`.
- The Gaea PNG decodes back to the original DEM within the expected 16-bit quantization error. Report the RMSE in meters.
- The OBJ re-imports with Z up and the correct extents.

Visual checks:
- Headless renders from the top and oblique cameras, written to `checks/vNNN/`.
- A side-by-side image of the top render next to the user's Google Earth reference image.

Write `terrain/00_base/vNNN/report.md` containing the inputs, settings, the results of every check, known limitations, and the list of output files.

Known limitations to state in the report: at 30 m resolution, quarry benches (about 10–15 m high) are not captured, and the DEM's capture date is earlier than the imagery's.

---

## Orchestrator (`run_pipeline.py`)
- `python run_pipeline.py` runs stages 0 to 6 in order.
- `--from 03` and `--only 05` run part of the pipeline. `--auto` skips the pauses between stages.
- Each stage is idempotent: it skips work when its outputs exist and the config hash hasn't changed.
- Log everything to `logs/run_vNNN.log`.
- Stop at the first failed stage and print a short explanation of the error.

## Done when
- Running `python run_pipeline.py --auto` from a clean checkout, with a filled `config.yaml`, produces every output above with all checks passing.
- Opening `quarry_base.blend` shows the textured, true-scale terrain at the correct BlenderGIS georeference.
- The Rhino OBJ opens Z-up, in meters, with the documented origin offset.
- The Gaea package and its README are enough to set up a Gaea project at the correct scale without any guesswork.

## Out of scope for this task (later stages will use these outputs)
- Bench and haul-road reconstruction, and pit masks.
- Gaea graphs.
- Hydrology and precipitation simulation.
- Architectural modeling.
- Final render setup.
