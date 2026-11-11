"""Bring a Gaea result back into the project's coordinate system.

STUB - to be implemented in the Gaea stage. Only the signature is defined.
The conversion rules it must follow are in exchange/gaea/vNNN/gaea_meta.json
("formulas") and README_gaea.md.
"""
from __future__ import annotations

from pathlib import Path


def reimport_from_gaea(
    gaea_output: Path,
    gaea_meta_json: Path,
    out_dir: Path,
    *,
    height_m: float | None = None,
    crop_padding: bool = True,
    write_blender: bool = False,
) -> Path:
    """Convert a Gaea heightmap export back to a georeferenced elevation raster.

    Args:
        gaea_output: heightmap exported from Gaea (16-bit PNG/TIFF or float EXR).
        gaea_meta_json: gaea_meta.json of the base run the Gaea project started from.
        out_dir: new versioned folder to write into (never terrain/00_base).
        height_m: the Height used in Gaea, if different from zmax - zmin.
        crop_padding: drop the edge-value padding and return only the data extent.
        write_blender: also build a Blender mesh in the same local coordinates.

    Returns:
        Path to the written GeoTIFF (real metres, project CRS).
    """
    raise NotImplementedError("reimport_from_gaea is implemented in the Gaea stage.")
