# Quarry base-terrain pipeline

> **Not a programmer? Start here:** read **[GUIDE.md](GUIDE.md)**, then double-click
> **`Start Quarry Maker.command`** (Mac) or **`Start Quarry Maker.bat`** (Windows). A simple app
> opens in your browser: draw your quarry on a map, press "Run next step", and open the results.
> No terminal and no code needed.

Scripted pipeline that turns an area of interest, a 30 m OpenTopography DEM and
satellite imagery into:

- `terrain/00_base/vNNN/quarry_base.blend`: true-scale, textured terrain with BlenderGIS georeferencing
- `exchange/rhino/vNNN/`: OBJ (Z-up, metres) + texture, FBX, `rhino_import_notes.md`
- `exchange/gaea/vNNN/`: square 16-bit PNG + float EXR heightmap, colour map, `gaea_meta.json`, `README_gaea.md`
- `terrain/00_base/vNNN/report.md`: inputs, settings, check results and limitations
- **`terrain/10_terraces/vNNN/quarry_terraced.blend`**: quarry benches **calculated from the
  satellite image + DEM** (stage 07, interpreted, never mixed into the measured base)

The full spec is in [CLAUDE.md](CLAUDE.md).

## Setup

```bash
pip install -r requirements.txt
```

```bash
export OPENTOPO_API_KEY=...      # never put it in a file
```

Then fill in `config.yaml`:
- `aoi.west/south/east/north`: WGS84 lon/lat of the site.
- `blender.path`, or set the `BLENDER_PATH` environment variable instead.

Check the environment (Python packages, Blender version, BlenderGIS add-on, API key):

```bash
python scripts/check_env.py
```

## Run

```bash
python run_pipeline.py            # stages 00-07, pauses after each stage for review
```

```bash
python run_pipeline.py --auto     # no pauses
```

```bash
python run_pipeline.py --from 03  # or: --only 05,  --new-run,  --force
```

- **Run IDs.** With `run_id: auto`, an unchanged config resumes the latest run, and a changed config starts `vNNN+1`.
- **Write-once.** `terrain/00_base/vNNN` is never overwritten. A stage refuses to write into a run that was made with a different config hash.
- **Skipping.** A stage skips its work when its outputs exist for the same config hash (`checks/vNNN/stage_NN.json`).
- **Caching.** Downloads are cached in `data/raw/`: the DEM by bbox hash, the imagery tile by tile.
- **Logging.** Everything goes to `logs/run_vNNN.log`, and check images go to `checks/vNNN/`.

## Terraces from imagery (stage 07)

A 30 m DEM shows a quarry as a smooth bowl, but the satellite image shows the benches. Stage 07
combines the two:

1. **Pit mask.** Bare rock in the image, kept where the DEM has relief. You can override it with
   `terraces.pit_mask`, a GeoJSON polygon in lon/lat or the project CRS.
2. **Bench edges.** Image edges that run *along* the DEM contours.
3. **Bench height.** Benches are mined at fixed elevations, so edge strength projected onto
   elevation repeats every *H*. The stage scans *H* and checks the result against spatially
   shifted edge maps. If no real pattern is found, it stops and asks you for a known bench height.
4. **Levels.** Bench floor elevations `G + (n + 0.5)·H` come from the phase of that pattern. The
   DEM surface is corrected locally (by at most ±H/2) to meet them.
5. **Geometry.** Flat floors and faces at `face_angle_deg`, built on the measured surface. Flat pit
   floors keep the measured shape.

In Blender, `quarry_terraced.blend` has the same BlenderGIS georeference as the base. The
`TERRAIN_terraced` mesh carries these point attributes:
- `confidence`
- `terrace_class` (0 outside, 1 floor, 2 bench, 3 face, 4 water)
- `dz_interp`

Switch its material to `MAT_terrace_confidence` or `MAT_terrace_class` to inspect them. Bench
floor elevations are listed in `terraces_params.json` and on the object.

**How accurate.** On a synthetic quarry with known benches
(`python tests/synthetic_quarry.py make|evaluate`):

| bench area within 3 m of the truth | plain 30 m DEM | stage 07 |
|---|---|---|
| one pit (12 m benches) | 65% | 89% (height found: 11.84 m) |
| complex: pit A (12 m) | 67% | 84% |
| complex: pit B (10 m, other levels, small) | 65% | 75% |
| complex: pit dug after the DEM | – | left as measured (correct) |
| image without benches | – | stops and asks (no invented benches) |

The method works region by region, so a mining complex with several pits and dumps (each with its
own bench levels) is handled. It builds benches only where a region shows clear evidence.
`07_bench_evidence.png` shows where.

Accuracy is limited by these:
- **DEM date.** The DEM predates the imagery: newly dug areas have no depth information. See
  `confidence.tif`.
- **Plan detail.** Bench positions in plan follow the 30 m DEM contours. Very tight curves are
  simplified.
- **Unmodelled features.** Face angle and haul ramps are not measured.
- **Small pits.** Pits whose walls are only a few 30 m DEM cells wide are less reliable.

If you know the bench height, set `terraces.bench_height_m`.

## BlenderGIS georeferencing

Read from BlenderGIS `geoscene.py` (class `SK`, add-on version 2.2.14). The georef is
stored as scene custom properties with these exact keys:

| key | value |
|---|---|
| `SRID` | CRS string, e.g. `EPSG:32635` |
| `crs x`, `crs y` | scene origin in CRS units (= `origin_x`, `origin_y` from `meta.json`) |
| `longitude`, `latitude` | the same origin in WGS84 |

These are *not* `crs`/`crsx`/`crsy`. Stage 04 writes these keys, and stage 06
checks them against `meta.json`. `scale` is left unset; BlenderGIS reads that as
1, which is true scale.

## Conventions

- **Local origin.** The lower-left corner of the extent, rounded to the nearest 100 m. Blender and Rhino use local coordinates: `local = real - origin`. Z is the real elevation.
- **Vertices.** One vertex per DEM cell centre, so the mesh is half a cell smaller than the extent on each side.
- **Extent.** The buffered AOI, snapped outward to multiples of the DEM resolution. Every raster, the mesh and the Gaea package share this extent.

## Testing without network

`tests/make_synthetic_fixture.py` builds a throwaway project with a **synthetic**
DEM, synthetic tiles and a synthetic reference image (terraces off). It pre-seeds the
caches, so stages 01 and 02 make no requests. `tests/synthetic_quarry.py` builds a quarry with
known benches, for testing stage 07:

```bash
python tests/make_synthetic_fixture.py /tmp/quarry_test --blender /path/to/blender
```

```bash
python run_pipeline.py --config /tmp/quarry_test/config.yaml --auto
```
