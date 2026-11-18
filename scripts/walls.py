"""Quarry walls seen in the satellite photo: dark wall bands and sun geometry.

A sawn quarry wall that faces away from the sun is in shade, and it throws
a shadow onto the floor below. From above, both show as one dark band along
the wall, darker than the bright floors on either side. Two facts follow:
  * the band marks a wall exactly (its plan position), and
  * the floor on the band's side AWAY from the sun is the lower one.
Walls facing the sun are lit and show no dark band; they are found by the
ordinary photo edges instead.

numpy + rasterio only.
"""
from __future__ import annotations

import datetime
import math

import numpy as np

import linework as lw
import terraces as T

# local solar time of the image (sun-synchronous satellites in Esri World
# Imagery: WorldView, GeoEye, QuickBird, Pleiades cross at ~10:30)
SATELLITE_LOCAL_TIME_H = 10.5


def sun_position(date: datetime.date, lat: float, local_solar_hour: float = SATELLITE_LOCAL_TIME_H):
    """Approximate solar azimuth (deg from north, clockwise) and elevation (deg)."""
    doy = date.timetuple().tm_yday
    g = 2 * math.pi / 365 * (doy - 1 + (local_solar_hour - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.radians(15 * (local_solar_hour - 12))
    phi = math.radians(lat)
    el = math.asin(math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha))
    az = math.atan2(-math.sin(ha) * math.cos(decl),
                    math.cos(phi) * math.sin(decl) - math.sin(phi) * math.cos(decl) * math.cos(ha))
    return math.degrees(az) % 360, math.degrees(el)


def imagery_metadata(lon: float, lat: float, wayback_config_url: str, timeout: float = 30.0):
    """Capture date and sensor of the current Esri World Imagery photo at a point,
    from the newest Wayback release's metadata service. Returns a dict or None."""
    import re

    import requests
    try:
        cfgw = requests.get(wayback_config_url, timeout=timeout).json()
        rel = sorted(cfgw.values(), key=lambda v: re.search(r"(\d{4}-\d{2}-\d{2})", v["itemTitle"]).group(1))[-1]
        for layer in range(3, 9):          # 15 cm ... 4.8 m resolution layers
            r = requests.get(f"{rel['metadataLayerUrl']}/{layer}/query", timeout=timeout, params=dict(
                geometry=f"{lon},{lat}", geometryType="esriGeometryPoint", inSR=4326,
                spatialRel="esriSpatialRelIntersects", outFields="*", returnGeometry="false", f="json")).json()
            feats = r.get("features") or []
            if feats:
                a = feats[0]["attributes"]
                d = str(a.get("SRC_DATE"))
                return {"date": f"{d[:4]}-{d[4:6]}-{d[6:8]}", "sensor": a.get("SRC_DESC"),
                        "resolution_m": a.get("SRC_RES"), "accuracy_m": a.get("SRC_ACC"),
                        "provider": a.get("NICE_DESC"), "release": rel["itemTitle"]}
    except Exception:
        return None
    return None


def wall_bands(rgb, res: float, rel_thr: float = 0.85, floor_m: float = 4.0):
    """Dark bands along walls: pixels clearly darker than the bright floor next
    to them (local upper envelope), not vegetation. rgb uint8 (H, W, 3)."""
    x = rgb.astype("float32") / 255.0
    lum = x.mean(-1)
    env = lw.max_filter(T.gauss(lum, 1.0).astype("float32"), max(1, int(round(floor_m / res))))
    rel = T.gauss(lum, 0.6) / np.maximum(T.gauss(env, 2.0 / res), 1e-3)
    exg = 2 * x[..., 1] - x[..., 0] - x[..., 2]
    band = (rel < rel_thr) & (exg < 0.05) & (lum > 0.12) & (x.max(-1) > 0)
    return T.box_blur(band.astype(float), 1) > 0.5


def bands_from_file(ortho_path, dst_transform, dst_shape, crs, within, block_m=600.0, log=None):
    """Wall bands detected at the photo's full resolution, block by block, and
    transferred to the analysis grid (a cell is band when any photo pixel is)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds

    Hd, Wd = dst_shape
    res_d = dst_transform.a
    bpx = max(64, int(round(block_m / res_d)))
    out = np.zeros(dst_shape, bool)
    margin = 20.0
    with rasterio.open(ortho_path) as src:
        res_s = src.res[0]
        blocks = [(r0, c0) for r0 in range(0, Hd, bpx) for c0 in range(0, Wd, bpx)
                  if within[r0:r0 + bpx, c0:c0 + bpx].any()]
        for n, (r0, c0) in enumerate(blocks):
            r1, c1 = min(Hd, r0 + bpx), min(Wd, c0 + bpx)
            win = from_bounds(dst_transform.c + c0 * res_d - margin, dst_transform.f - r1 * res_d - margin,
                              dst_transform.c + c1 * res_d + margin, dst_transform.f - r0 * res_d + margin,
                              src.transform).round_offsets().round_lengths()
            rgb = src.read([1, 2, 3], window=win, boundless=True, fill_value=0).transpose(1, 2, 0)
            band = wall_bands(rgb, res_s)
            sub_tf = rasterio.transform.from_origin(dst_transform.c + c0 * res_d, dst_transform.f - r0 * res_d,
                                                    res_d, res_d)
            tmp = np.zeros((r1 - r0, c1 - c0), "uint8")
            reproject(band.astype("uint8"), tmp, src_transform=src.window_transform(win), src_crs=crs,
                      dst_transform=sub_tf, dst_crs=crs, resampling=Resampling.max)
            out[r0:r1, c0:c1] = tmp > 0
            if log and (n + 1) % 20 == 0:
                log.info("  wall bands: %d/%d blocks", n + 1, len(blocks))
    return out & within


def wall_lines_from_bands(band, res: float, min_len_m: float = 12.0):
    """Centre lines of the wall bands, keeping only long ones (walls, not block piles)."""
    from rasterio.transform import Affine
    sk = lw.thin(band)
    lab, _ = T.label_components(sk, Affine.identity())
    cnt = np.bincount(lab.ravel())
    keep = cnt * res >= min_len_m
    keep[0] = False
    return keep[lab]
