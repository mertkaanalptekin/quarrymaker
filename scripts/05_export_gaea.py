"""Stage 05 - Gaea heightmap package.

From dem_utm.tif, builds a SQUARE grid of size_px x size_px that covers the
extent at a single pixel size (the longer side of the extent fills the
image; the shorter side is padded with edge values - nothing is stretched).
Writes to exchange/gaea/vNNN/:

  heightmap_16bit.png   (z - zmin) / (zmax - zmin) * 65535, 16-bit grayscale
  heightmap_m.exr       32-bit float, real metres (same grid)
  colormap.png          ortho on the same grid, same crop and padding
  gaea_meta.json        scale, padding, origin, conversion formulas
  README_gaea.md        how to set up the Gaea project at true scale
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import exr_io  # noqa: E402


def sample_clamped(arr, x0, y1, px_x, px_y, X, Y):
    """Bilinear sample of a north-up grid (pixel-is-area, top-left corner at
    (x0, y1), pixel size px_x/px_y) at real coordinates X, Y. Coordinates that
    fall outside are clamped to the nearest edge pixel = edge-value padding."""
    import numpy as np

    h, w = arr.shape[:2]
    c = np.clip((X - x0) / px_x - 0.5, 0, w - 1)
    r = np.clip((y1 - Y) / px_y - 0.5, 0, h - 1)
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
    c1, r1 = np.minimum(c0 + 1, w - 1), np.minimum(r0 + 1, h - 1)
    fc, fr = c - c0, r - r0
    if arr.ndim == 3:
        fc, fr = fc[..., None], fr[..., None]
    a = arr.astype("float64")
    top = a[r0, c0] * (1 - fc) + a[r0, c1] * fc
    bot = a[r1, c0] * (1 - fc) + a[r1, c1] * fc
    return top * (1 - fr) + bot * fr


def square_grid(meta, size_px):
    """Return the square grid definition (real-world bounds, pixel size)."""
    e = meta["extent"]
    wm, hm = e["xmax"] - e["xmin"], e["ymax"] - e["ymin"]
    p = max(wm, hm) / size_px
    side = p * size_px
    cx, cy = (e["xmin"] + e["xmax"]) / 2, (e["ymin"] + e["ymax"]) / 2
    sq = {"xmin": cx - side / 2, "xmax": cx + side / 2, "ymin": cy - side / 2, "ymax": cy + side / 2}
    pad = {  # in output pixels (can be fractional when the extent is not a whole number of pixels)
        "left": (e["xmin"] - sq["xmin"]) / p, "right": (sq["xmax"] - e["xmax"]) / p,
        "top": (sq["ymax"] - e["ymax"]) / p, "bottom": (e["ymin"] - sq["ymin"]) / p,
    }
    return sq, p, side, pad


def main(argv=None) -> int:
    import numpy as np
    import rasterio
    from PIL import Image, ImageDraw

    args, cfg, paths, log = common.stage_context(__doc__, "05", argv)
    if not args.force and common.stage_up_to_date(paths, cfg, "05"):
        log.info("Stage 05 up to date - skipping.")
        return 0
    gcfg = cfg.get("gaea") or {}
    N = int(gcfg.get("size_px", 2048))
    if int(gcfg.get("bit_depth", 16)) != 16:
        raise common.PipelineError("Only gaea.bit_depth: 16 is supported.")

    meta = common.read_json(paths.base / "meta.json")
    with rasterio.open(paths.processed / "dem_utm.tif") as ds:
        dem = ds.read(1).astype("float64")
        r = ds.res[0]
    zmin, zmax = float(dem.min()), float(dem.max())
    if abs(zmin - meta["elevation"]["min"]) > 1e-3 or abs(zmax - meta["elevation"]["max"]) > 1e-3:
        raise common.PipelineError("dem_utm.tif does not match meta.json - re-run stage 03.")
    if zmax - zmin <= 0:
        raise common.PipelineError("DEM is flat (zmax == zmin); cannot normalise.")
    e = meta["extent"]

    sq, p, side, pad = square_grid(meta, N)
    log.info("Gaea grid: %d px square, %.4f m/px, covers %.1f m; padding px L%.1f R%.1f T%.1f B%.1f",
             N, p, side, pad["left"], pad["right"], pad["top"], pad["bottom"])
    if p > r:
        log.warning("Gaea pixel (%.2f m) is coarser than the DEM cell (%.2f m): the heightmap is "
                    "downsampled; raise gaea.size_px to keep every DEM cell.", p, r)
    else:
        log.info("Heightmap resampled (bilinear) from %.2f m DEM cells to %.4f m pixels.", r, p)

    xs = sq["xmin"] + (np.arange(N) + 0.5) * p
    ys = sq["ymax"] - (np.arange(N) + 0.5) * p
    X, Y = np.meshgrid(xs, ys)
    hz = sample_clamped(dem, e["xmin"], e["ymax"], r, r, X, Y)
    q = np.round((hz - zmin) / (zmax - zmin) * 65535.0)
    q16 = np.clip(q, 0, 65535).astype("uint16")

    out = paths.gaea
    out.mkdir(parents=True, exist_ok=True)
    png = out / "heightmap_16bit.png"
    exr = out / "heightmap_m.exr"
    color = out / "colormap.png"
    Image.fromarray(q16).save(png)
    exr_io.write_exr(exr, hz.astype("float32"))

    with rasterio.open(paths.processed / "ortho_utm.tif") as ds:
        rgb = ds.read().transpose(1, 2, 0)
    img = Image.fromarray(rgb)
    content_w, content_h = (e["xmax"] - e["xmin"]) / p, (e["ymax"] - e["ymin"]) / p
    if img.width > content_w * 1.5:  # pre-reduce so bilinear sampling does not alias
        img = img.resize((max(1, round(content_w)), max(1, round(content_h))), Image.LANCZOS)
    arr = np.asarray(img)
    px_x = (e["xmax"] - e["xmin"]) / arr.shape[1]
    px_y = (e["ymax"] - e["ymin"]) / arr.shape[0]
    col = sample_clamped(arr, e["xmin"], e["ymax"], px_x, px_y, X, Y)
    Image.fromarray(np.clip(np.round(col), 0, 255).astype("uint8")).save(color)

    step = (zmax - zmin) / 65535.0
    gmeta = {
        "run_id": paths.run_id,
        "config_hash": common.config_hash(cfg),
        "crs": meta["crs"],
        "size_px": N,
        "bit_depth": 16,
        "zmin": zmin,
        "zmax": zmax,
        "height_range_m": zmax - zmin,
        "pixel_size_m": p,
        "terrain_width_m": side,
        "terrain_height_m": side,
        "data_extent_width_m": e["xmax"] - e["xmin"],
        "data_extent_height_m": e["ymax"] - e["ymin"],
        "dem_cell_size_m": r,
        "square_bounds_crs": sq,
        "square_bounds_local": {k: v - (meta["origin_x"] if k.startswith("x") else meta["origin_y"])
                                for k, v in sq.items()},
        "padding_px": pad,
        "padding_px_rounded": {k: int(round(v)) for k, v in pad.items()},
        "padding_mode": "edge values (nearest edge cell), no stretching",
        "content_px": {"x0": pad["left"], "y0": pad["top"],
                       "x1": N - pad["right"], "y1": N - pad["bottom"]},
        "origin_x": meta["origin_x"],
        "origin_y": meta["origin_y"],
        "resampling": "bilinear from dem_utm.tif",
        "quantization_step_m": step,
        "expected_rmse_m": step / np.sqrt(12),
        "formulas": {
            "png16_to_metres": "z = zmin + (v / 65535) * (zmax - zmin)",
            "normalised_0_1_to_metres": "z = zmin + n * (zmax - zmin)",
            "gaea_height_m_to_metres": "z = zmin + h   (when Gaea Height = zmax - zmin)",
            "metres_to_png16": "v = round((z - zmin) / (zmax - zmin) * 65535)",
            "pixel_to_crs": "x = square_xmin + (col + 0.5) * pixel_size_m ; "
                            "y = square_ymax - (row + 0.5) * pixel_size_m   (row 0 = north)",
            "pixel_to_crs_other_resolution": "for an output of M px: pixel = terrain_width_m / M",
            "crs_to_local_blender_rhino": "x_local = x - origin_x ; y_local = y - origin_y",
        },
        "files": {"heightmap_png": png.name, "heightmap_exr": exr.name, "colormap": color.name},
        "imagery_attribution": meta["imagery"].get("attribution"),
    }
    gmeta_path = out / "gaea_meta.json"
    common.write_json(gmeta_path, gmeta)
    readme = out / "README_gaea.md"
    readme.write_text(gaea_readme(gmeta, meta))

    # Check image: heightmap + colour map with the data area outlined.
    prev = paths.checks / "05_gaea_package.png"
    a = Image.fromarray((q16 >> 8).astype("uint8"), "L").convert("RGB").resize((768, 768))
    b = Image.open(color).convert("RGB").resize((768, 768))
    s = 768 / N
    box = [pad["left"] * s, pad["top"] * s, (N - pad["right"]) * s, (N - pad["bottom"]) * s]
    for im in (a, b):
        ImageDraw.Draw(im).rectangle(box, outline=(255, 40, 40), width=2)
    sheet = Image.new("RGB", (1536 + 10, 768), "white")
    sheet.paste(a, (0, 0))
    sheet.paste(b, (778, 0))
    sheet.save(prev)

    summary = [
        f"Square heightmap {N} px, {p:.4f} m/px -> Gaea terrain width {side:.2f} m",
        f"Gaea height = zmax - zmin = {zmax - zmin:.2f} m (zmin {zmin:.2f}, zmax {zmax:.2f})",
        "Padding px (L/R/T/B): " + ", ".join(f"{pad[k]:.1f}" for k in ("left", "right", "top", "bottom")),
        f"16-bit step {step * 1000:.2f} mm; files in {common.rel(paths, out)}",
    ]
    common.finish_stage(paths, cfg, "05", summary, [png, exr, color, gmeta_path, readme],
                        [prev], log)
    return 0


def gaea_readme(g, meta) -> str:
    pad = g["padding_px"]
    return f"""# Gaea package - run {g['run_id']}

Base terrain for Gaea, from {meta['dem']['source']} ({g['dem_cell_size_m']:g} m cells), {g['crs']}.
Everything here is measured terrain: no benches, smoothing or erosion.

## Files
| file | content |
|---|---|
| `heightmap_16bit.png` | {g['size_px']} x {g['size_px']} px, 16-bit grey, 0 = zmin, 65535 = zmax |
| `heightmap_m.exr` | same grid, 32-bit float, **real elevation in metres** (R=G=B) |
| `colormap.png` | ortho on the same grid, same crop and padding |
| `gaea_meta.json` | all numbers below, plus the conversion formulas |

## Set the scale in Gaea (the two numbers that matter)
In the project's terrain definition (the **Scale** and **Height** settings):

- **Scale / terrain width = {g['terrain_width_m']:.2f} m**  (square: {g['size_px']} px x {g['pixel_size_m']:.4f} m/px)
- **Height = {g['height_range_m']:.2f} m**  (= zmax - zmin = {g['zmax']:.2f} - {g['zmin']:.2f})
- Resolution: {g['size_px']} (match the heightmap so 1 Gaea pixel = 1 heightmap pixel).

Import `heightmap_16bit.png` with a File/import node and **disable any auto-level or
normalise option**, so 0 stays at the base and 65535 reaches the full Height.
Gaea elevation 0 corresponds to real elevation **{g['zmin']:.2f} m**.

Quick check: the highest point should read {g['height_range_m']:.2f} m in Gaea, and the
terrain should be {g['terrain_width_m']:.0f} m across. If either is off, the scale is wrong.

## Padding (the image is square; the data is not)
The data extent is {g['data_extent_width_m']:.0f} x {g['data_extent_height_m']:.0f} m. The
shorter side was padded with **edge values** (not stretched). Padding in pixels:
left {pad['left']:.1f}, right {pad['right']:.1f}, top {pad['top']:.1f}, bottom {pad['bottom']:.1f}.
The real data sits in pixels x {g['content_px']['x0']:.1f}..{g['content_px']['x1']:.1f},
y {g['content_px']['y0']:.1f}..{g['content_px']['y1']:.1f} (row 0 = north).
Erosion that flows in from the padded strips is not real; mask or crop it later.

## Converting Gaea output back to real elevation
- 16-bit export, v in 0..65535: `z = {g['zmin']:.4f} + v / 65535 * {g['height_range_m']:.4f}`
- Normalised 0..1 export n: `z = {g['zmin']:.4f} + n * {g['height_range_m']:.4f}`
- Float export in Gaea metres h (Height = {g['height_range_m']:.2f}): `z = {g['zmin']:.4f} + h`
- **Export without auto-level/normalisation.** If Gaea re-normalises the output to 0..1
  using its own min/max, the absolute scale is lost; note the output's min/max then.
- Pixel (col, row) of an M x M output to projected coordinates ({g['crs']}):
  `x = {g['square_bounds_crs']['xmin']:.3f} + (col + 0.5) * {g['terrain_width_m']:.3f} / M`,
  `y = {g['square_bounds_crs']['ymax']:.3f} - (row + 0.5) * {g['terrain_width_m']:.3f} / M`.
- Local Blender/Rhino coordinates: subtract origin ({g['origin_x']:.1f}, {g['origin_y']:.1f}).

16-bit quantisation step: {g['quantization_step_m'] * 1000:.2f} mm (expected RMSE
{g['expected_rmse_m'] * 1000:.2f} mm). `scripts/reimport_from_gaea.py` will bring results back.
"""


if __name__ == "__main__":
    common.run_main(main, "05")
