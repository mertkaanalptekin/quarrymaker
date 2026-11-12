# Quarry Research Project — Base Terrain Pipeline

## Purpose
Build an automated, reproducible pipeline that creates the **base terrain** of a quarry site for an architectural research project.

Inputs: an area of interest, a 30 m DEM from OpenTopography, and satellite imagery.

Outputs:
- a georeferenced, true-scale Blender terrain that is compatible with the BlenderGIS add-on;
- a heightmap package for Gaea;
- mesh exports for Rhino;
- an **interpreted** terrace model (stage 07): quarry benches calculated from the satellite image with the help of the DEM, as a Blender file in the same georeference.

Later stages read from this base: Gaea terraforming, precipitation simulations, architectural modeling and renders. Base terrain is the single source of truth for scale, coordinates and elevation.

Scope: the measured base terrain (stages 00–06) plus the interpreted terrace model (stage 07). Do not build Gaea graphs, simulations or architecture yet. Only produce the files those stages will consume.

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
   - Do not add benches, smoothing or erosion to `terrain/00_base/`.
   - Resampling is allowed only as configured, and must be logged.
   - Interpreted geometry (benches from stage 07) lives only in `terrain/10_terraces/vNNN/` and `exchange/rhino/vNNN/terraced/`, is labelled as interpreted, and never changes the base. Stage 07 checks the base files' hashes before and after.
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

**BlenderGIS compatibility.** Before writing any Blender script, read BlenderGIS's geoscene source code. Confirm how it stores georeferencing on the scene: the CRS and the scene origin. The terrain built in stage 4 must set the same properties, so the user can keep using BlenderGIS tools (basemaps, extra imports) on this file and everything lands in the right place.
- Confirmed in BlenderGIS 2.2.14 `geoscene.py` (class `SK`): scene custom properties `SRID` (e.g. `EPSG:32635`), `crs x`, `crs y` (origin in CRS units), `longitude`, `latitude` (origin in WGS84). Not `crs`/`crsx`/`crsy`. Both stage 04 and stage 07 .blend files set these.

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
    07_terraces.py           # interpreted benches (+ terraces.py library)
  data/
    raw/            # untouched downloads, cached
    processed/      # reprojected / clipped rasters
    reference/      # user's Google Earth image, notes
  terrain/
    00_base/vNNN/   # .blend, exports, meta.json, report  (measured, write-once)
    10_terraces/vNNN/  # interpreted benches: rasters, .blend, report
  exchange/
    gaea/vNNN/      # heightmap package for Gaea
    rhino/vNNN/     # meshes + texture for Rhino (terraced/ = interpreted)
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
terraces:
  enabled: true
  resolution_m: 2.0
  bench_height_m: auto      # or the known bench height in metres
  face_angle_deg: 70
  pit_mask: null            # optional GeoJSON pit outline
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

### 07 — Interpret terraces (`07_terraces.py`)
Calculates the quarry's benches from the satellite image with the help of the DEM. All output is **interpreted** and kept apart from the base.
- **Quarry area** (`quarry_area`), which separates the quarry from farmland, villages, greenhouses and factories:
  1. bare-rock candidates from a quarry index, using a two-level Otsu threshold, kept only where the DEM shows relief;
  2. an opening that cuts thin bright links such as roads;
  3. only bodies of at least `quarry_min_area_ha` with real relief are kept;
  4. closing and hole filling turn the result into one continuous area.

  The user can supply `terraces.pit_mask` (GeoJSON) instead.
- **Edges:** image edges that run along the DEM contours (bench crests and toes).
- **Bare ground only:** benches are looked for on bare rock or soil. The test uses colour saturation, which is lighting-independent, split inside the outline only when vegetation is clearly present.
- **Bench height:** benches are mined at fixed elevations, so edge strength projected onto elevation repeats every H. H maximises that coherence above the baseline of the elevation distribution. Edges on planar wall parts weigh more, because a 30 m DEM squeezes elevations at the rim and floor. A known H can be set.
- **Region by region:** a mining complex holds several pits and dumps with their own levels, plus areas dug after the DEM.
  - The pattern is tested in overlapping regions of `region_size_m` (400 m).
  - A region counts when its score beats two kinds of null maps by `min_significance` (×2):
    - circularly shifted edge weights;
    - spatially shifted edge maps, which keep regular line patterns such as field or road grids.
  - Significant regions join into **bench zones** (usually one pit or dump each) when their heights and levels agree.
  - Neighbouring zones merge when the union keeps one coherent level system.
  - Each zone gets one H, refined to 0.02 m, and one level system.
  - A zone is re-tested as a whole and must pass a plausibility check: implied bench floor at most `max_bench_width_m`, and H not at the edge of the search range.
- **Ask, don't guess:** if no zone survives, the stage stops. It still writes `07_bench_evidence.png` and the other diagnostic images, and records the stage as failed, so the app shows the images.
- **Levels and surface:** in each zone, bench floors sit at fixed elevations `G + (n + 0.5)·H`. The DEM surface is corrected locally, by at most ±H/2, to meet them. Benches are built only where a zone has evidence (class 5 = pit without clear benches, which keeps the measured surface).
- **Benches cut along the photo's bench lines** (`method: auto`, where no zone is confirmed; illustrative):
  - **Lines:** bench lines are traced at the photo's full resolution with Canny edges (gradient, 1-px thinning, hysteresis). Only long, extended lines (≥ `line_min_length_m`) on bare ground are kept, so rubble and blocks give no lines.
  - **Cut:** a sheet follows the DEM at large scale (`bench_smoothing_m`) and may break only along the lines (a screened Poisson problem with cuts, solved by conjugate gradients). The applied change is `dz = sheet(lines) − sheet(no lines)`:
    - the ground between lines becomes flat benches;
    - each line becomes a face whose height is the DEM's own drop there;
    - hills and low points stay as measured.
  - **Clean-up:** close edge pairs merge into one face, small enclosed islands are neutralised, and the change is capped at `max_cut_m`.
  - **Labelling:** `bench_method` = 2 marks these cells, and they are drawn in `07_bench_lines.png`.
  - **Known limits:**
    - steps are only as tall as the old DEM allows;
    - road and yard edges can make small steps;
    - in a small synthetic pit (pit B) the cut benches were less accurate than the plain DEM (51% vs 67% within 3 m), even though they look terraced.
- **Geometry (confirmed zones):** a staircase with the configured face angle, where faces take a fraction `tan(slope)/tan(face angle)` of each period. Flat pit floors keep the measured shape. Water is flattened.
- **Outputs:**
  - `terraces_dem.tif`, `classes.tif`, `confidence.tif`, `dz.tif` and `surface_correction.tif`;
  - `pit_outline.geojson` and `terraces_params.json` (bench height, floor elevations, statistics);
  - `quarry_terraced.blend`: same BlenderGIS georef and origin. Interpreted mesh with `confidence`, `terrace_class` and `dz_interp` attributes, plus the measured context mesh and the pit outline curve;
  - OBJ/FBX in `exchange/rhino/vNNN/terraced/`;
  - `report_terraces.md` and check images.
- **Checks:**
  - the base is untouched (hashes);
  - interpreted = measured outside the pit;
  - no systematic shift (mean dz within H/4);
  - the .blend georef equals `meta.json`;
  - the mesh Z range and position match the raster.
- **Test:** `tests/synthetic_quarry.py make|evaluate` builds quarries with known benches and measures the recovery. Results (share of the bench area within 3 m of the truth, plain DEM → stage 07):
  - `--scenario simple`, one pit: 65% → 89%, with H 11.84 m for a true 12 m.
  - `--scenario complex`:
    - pit A (12 m benches): 67% → 84%;
    - pit B (10 m benches on other levels, small, with walls only about three DEM cells wide): 65% → 75%;
    - pit C (dug after the DEM): correctly left as measured.
  - Negative control: imagery without benches (a grid pattern) → the stage stops and asks.
  - With a known H, bench floors are within the DEM's vertical bias (~1 m).

- **Shadow ends:** a shadow falls on the side away from the sun (`sun_azimuth_deg`, 135 = southeast). At the bench edge, the photo turns bright to dark moving away from the sun; at the far end of the shadow it turns dark to bright. Edges whose brightness gradient points away from the sun are therefore dropped, leaving one line per bench.

### 08 — Plan linework (`08_linework.py`, `linework.py`)
DXF R12 (Rhino/CAD) plus GeoJSON in `exchange/rhino/vNNN/plan/`, in local metres like the OBJ.
- **QUARRY_OUTLINE:** from step 07's quarry area.
- **BENCH_LINES:** Canny lines at full resolution, shadow ends removed, 1-px thinned, then chained into polylines with m-connectivity (so diagonal and zigzag lines are not split), simplified to 0.5 m, 3D at the terraced surface.
- **ROADS:** a white top-hat (bright, narrower than about 26 m, unsaturated), thinned to centrelines.
- **AGRICULTURE:** flat, vegetated or saturated ground outside the quarry.
- **BLOCK_YARDS:** the density of small bright blobs (top-hat > 0.08, density > 0.30) on gentle slopes inside the quarry.

Coordinates are written to centimetres. On the real site the DXF is 4 MB, with 2,701 bench polylines (92.5 km), and an independent DXF reader opens it with no audit errors.

### 09 — Quarry expansion (`09_expansion.py`)
Uses Esri World Imagery Wayback. The config lives at `s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json`, and tiles come from `wayback.maptiles.arcgis.com` (both hosts are allowed in the "quarrymaker" cloud environment).
1. Hash 5 sample tiles of the site in every release and group releases with identical tiles: one group is one distinct photo, dated by its first release.
2. Download each photo at zoom 16 (up to `max_versions`, spread over time).
3. Run `quarry_area` on each.
4. Output, in `terrain/20_expansion/vNNN/`: area per date (JSON and CSV), the first-appearance growth map, GeoJSON outlines, `exchange/rhino/vNNN/plan/quarry_expansion.dxf` (one layer per date), and check images `09_*`.

---

## Orchestrator (`run_pipeline.py`)
- `python run_pipeline.py` runs stages 0 to 9 in order.
- `--from 03` and `--only 05` run part of the pipeline. `--auto` skips the pauses between stages.
- Each stage is idempotent: it skips work when its outputs exist and the config hash hasn't changed.
- Log everything to `logs/run_vNNN.log`.
- Stop at the first failed stage and print a short explanation of the error.

## User app (`app/`, launchers)
The user is an architect, not a programmer, so every feature must be reachable from the app.
- `Start Quarry Maker.command` (Mac/Linux) and `Start Quarry Maker.bat` (Windows) create a private `.venv`, install `requirements.txt` on first start, and run `app/quarry_app.py`.
- `app/quarry_app.py` is a local web app: standard library only, bound to 127.0.0.1, with a per-launch token. It lets the user:
  - draw the area and the pit outline on a map (Leaflet + Esri imagery);
  - pick Blender (auto-detected) and set the bench height;
  - enter the OpenTopography key, which stays in memory only and is passed to the pipeline as `OPENTOPO_API_KEY`, never written;
  - run one stage at a time (checkpoints with captioned check images) or all stages;
  - open the results in Blender or the file browser.
- `app/config_io.py` edits only the managed values in `config.yaml`, keeping all comments. Saving an area also sets `crs_epsg` to the area's UTM zone.
- `GUIDE.md` is the plain-language user guide. Keep it, the app's step descriptions and the error hints in sync with the pipeline.

## Done when
- Running `python run_pipeline.py --auto` from a clean checkout, with a filled `config.yaml`, produces every output above with all checks passing.
- Opening `quarry_base.blend` shows the textured, true-scale terrain at the correct BlenderGIS georeference.
- The Rhino OBJ opens Z-up, in meters, with the documented origin offset.
- The Gaea package and its README are enough to set up a Gaea project at the correct scale without any guesswork.

## Status and next steps (keep this updated)
- **Done:** stages 00–08, the app and launchers, and the UTF-8 fix for Windows. Tested on the user's real site: a marble quarry district at W29.31 S37.84 E29.39 N37.89 (UTM 35N).
- **Real-site findings:**
  - The 30 m DEM (2011–2015) cannot confirm bench levels there.
  - The bench lines traced in the photo plus the cut sheet give the terraced look the user approved.
  - The user reports that the sun is in the southeast.
- **Stage 09 (expansion over time) is built**, using Esri Wayback (the user chose this).
- **Possible:** bench heights from shadow widths (this needs the sun elevation); strict "carve below DEM" mode; an "exclude area" drawing tool in the app (the factories are inside the auto quarry area).
- **Synthetic tests:** `tests/synthetic_quarry.py make|evaluate`. Simple: 89%. Complex pit B with cut benches: 51% against 67% for the plain DEM (documented limit).

## Out of scope for this task (later stages will use these outputs)
- Haul-road and ramp reconstruction (benches and pit masks are now stage 07).
- Gaea graphs.
- Hydrology and precipitation simulation.
- Architectural modeling.
- Final render setup.
