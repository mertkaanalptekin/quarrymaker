# Quarry base-terrain pipeline

Scripted pipeline that turns an area of interest, a 30 m OpenTopography DEM and
satellite imagery into:

- `terrain/00_base/vNNN/quarry_base.blend`: true-scale, textured terrain with BlenderGIS georeferencing
- `exchange/rhino/vNNN/`: OBJ (Z-up, metres) + texture, FBX, `rhino_import_notes.md`
- `exchange/gaea/vNNN/`: square 16-bit PNG + float EXR heightmap, colour map, `gaea_meta.json`, `README_gaea.md`
- `terrain/00_base/vNNN/report.md`: inputs, settings, check results and limitations

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
python run_pipeline.py            # stages 00-06, pauses after each stage for review
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
DEM, synthetic tiles and a synthetic reference image. It pre-seeds the caches,
so stages 01 and 02 make no requests:

```bash
python tests/make_synthetic_fixture.py /tmp/quarry_test --blender /path/to/blender
```

```bash
python run_pipeline.py --config /tmp/quarry_test/config.yaml --auto
```
