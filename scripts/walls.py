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


def model_shadows(z, res: float, sun_az: float, sun_el: float, max_m: float = 40.0):
    """Where a terrain model would look dark from above with this sun: cast
    shadows (ray-marched toward the sun) and faces turned away from the sun."""
    H, W = z.shape
    sc, sr = math.sin(math.radians(sun_az)), -math.cos(math.radians(sun_az))   # toward the sun (col, row)
    t = math.tan(math.radians(sun_el))
    shade = np.zeros(z.shape, bool)
    rows, cols = np.mgrid[0:H, 0:W]
    for k in np.arange(1, max_m / res + 1):
        r = np.clip(np.round(rows + sr * k).astype(int), 0, H - 1)
        c = np.clip(np.round(cols + sc * k).astype(int), 0, W - 1)
        shade |= z[r, c] > z + k * res * t + 0.3
    gy, gx = np.gradient(z, res)                       # row (south) and column (east) slopes
    nxyz = np.stack([-gx, gy, np.ones_like(z)], -1)    # east, north, up
    nxyz /= np.linalg.norm(nxyz, axis=-1, keepdims=True)
    s = np.array([math.cos(math.radians(sun_el)) * math.sin(math.radians(sun_az)),
                  math.cos(math.radians(sun_el)) * math.cos(math.radians(sun_az)), math.sin(math.radians(sun_el))])
    self_shade = (nxyz @ s) < 0.35
    return shade | self_shade


def shadow_agreement(model_dark, photo_band, within, tol_px: int = 1):
    """How well the model's dark areas match the photo's wall bands (inside
    `within`). precision: model-dark pixels near a photo band; recall: photo
    band pixels near model-dark ones. Returns dict with precision, recall, f1."""
    def dil(m):
        return T.box_blur(m.astype(float), tol_px) > 0 if tol_px > 0 else m
    md = model_dark & within
    pb = photo_band & within
    if md.sum() == 0 or pb.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = float((md & dil(pb)).sum()) / float(md.sum())
    r = float((pb & dil(md)).sum()) / float(pb.sum())
    f = 2 * p * r / max(p + r, 1e-9)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


WAYBACK_CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
ESRI_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def detail_ortho(root, utm_bounds, epsg: int, zoom: int, out_path, tile_url=None, res_m=0.5,
                 max_tiles=6000, throttle=0.1, log=None):
    """Download Esri World Imagery tiles at `zoom` over utm_bounds (xmin, ymin,
    xmax, ymax) and write them as a GeoTIFF in the project CRS at res_m.
    Tiles are cached in data/raw/tiles/esri_world_imagery/<zoom>/ (shared with
    stage 02). Returns out_path; raises on failure (callers fall back)."""
    import importlib.util
    import time
    from pathlib import Path

    import rasterio
    import requests
    from PIL import Image
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("fetch_imagery", here / "02_fetch_imagery.py")
    f2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f2)
    xmin, ymin, xmax, ymax = utm_bounds
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    w, n = to_ll.transform(xmin, ymax)
    e, s = to_ll.transform(xmax, ymin)
    w2, s2 = to_ll.transform(xmin, ymin)
    e2, n2 = to_ll.transform(xmax, ymax)
    w, e, s, n = min(w, w2), max(e, e2), min(s, s2), max(n, n2)
    tx0, ty0 = f2.lonlat_to_tile(w, n, zoom)
    tx1, ty1 = f2.lonlat_to_tile(e, s, zoom)
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    if nx * ny > max_tiles:
        raise RuntimeError(f"detail photo needs {nx * ny} tiles (cap {max_tiles})")
    url_t = tile_url or ESRI_TILE_URL
    cache = Path(root) / "data" / "raw" / "tiles" / "esri_world_imagery" / str(zoom)
    sess = requests.Session()
    sess.headers["User-Agent"] = f2.USER_AGENT
    mos = np.zeros((ny * 256, nx * 256, 3), "uint8")
    got = 0
    for j, y in enumerate(range(ty0, ty1 + 1)):
        for i, x in enumerate(range(tx0, tx1 + 1)):
            tp = cache / str(x) / f"{y}.img"
            if not tp.exists():
                for attempt in range(4):
                    try:
                        r = sess.get(url_t.format(z=zoom, x=x, y=y), timeout=30)
                        if r.status_code == 200 and r.content:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"tile download failed: z{zoom} x{x} y{y}")
                tp.parent.mkdir(parents=True, exist_ok=True)
                tp.write_bytes(r.content)
                got += 1
                time.sleep(throttle)
            with Image.open(tp) as im:
                mos[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = np.asarray(im.convert("RGB"))
        if log and (j + 1) % 10 == 0:
            log.info("  detail photo: row %d/%d (%d new tiles)", j + 1, ny, got)
    mx0, _, _, my1 = f2.tile_bounds_merc(tx0, ty0, zoom)
    px = 2 * f2.WEBM / 2 ** zoom / 256
    W = int(np.ceil((xmax - xmin) / res_m))
    H = int(np.ceil((ymax - ymin) / res_m))
    dst = np.zeros((3, H, W), "uint8")
    tf = from_origin(xmin, ymax, res_m, res_m)
    reproject(mos.transpose(2, 0, 1), dst, src_transform=from_origin(mx0, my1, px, px), src_crs="EPSG:3857",
              dst_transform=tf, dst_crs=f"EPSG:{epsg}", resampling=Resampling.bilinear)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8",
                       crs=f"EPSG:{epsg}", transform=tf, compress="deflate", tiled=True,
                       photometric="RGB") as ds:
        ds.write(dst)
    tmp.replace(out_path)
    if log:
        log.info("Detail photo: zoom %d, %d tiles (%d downloaded), %.2f m pixels", zoom, nx * ny, got, res_m)
    return out_path


def read_user_lines(path, origin, log=None):
    """Bench edges drawn by the user (Rhino .3dm, .dxf or GeoJSON) as a list of
    (N, 2) arrays in project CRS metres. Rhino/DXF coordinates are the local
    model coordinates of the Quarry Maker exports (real = local + origin);
    Rhino unit systems are converted to metres. All curves on visible layers
    are used, or only those on a layer whose name contains 'bench' if any."""
    from pathlib import Path
    p = Path(path)
    ox, oy = origin
    out = []
    if p.suffix.lower() == ".3dm":
        import rhino3dm as r
        m = r.File3dm.Read(str(p))
        if m is None:
            raise RuntimeError(f"cannot read {p}")
        scale = {"Millimeters": 0.001, "Centimeters": 0.01, "Meters": 1.0, "Kilometers": 1000.0,
                 "Inches": 0.0254, "Feet": 0.3048}.get(str(m.Settings.ModelUnitSystem).split(".")[-1], 1.0)
        layers = [L.FullPath for L in m.Layers]
        bench_layers = {i for i, n in enumerate(layers) if "bench" in n.lower()}
        for o in m.Objects:
            g = o.Geometry
            li = o.Attributes.LayerIndex
            if not isinstance(g, r.Curve) or (bench_layers and li not in bench_layers):
                continue
            if isinstance(g, r.PolylineCurve):
                pts = np.array([(g.Point(i).X, g.Point(i).Y) for i in range(g.PointCount)])
            else:
                n = max(8, int(g.GetLength() * scale / 0.5)) if hasattr(g, "GetLength") else 200
                ts = np.linspace(g.Domain.T0, g.Domain.T1, n)
                pts = np.array([(g.PointAt(t).X, g.PointAt(t).Y) for t in ts])
            out.append(pts * scale + np.array([ox, oy]))
    elif p.suffix.lower() == ".dxf":
        from linework import read_dxf_polylines
        out = [a + np.array([ox, oy]) for a in read_dxf_polylines(p)]
    else:
        import json
        gj = json.loads(p.read_text(encoding="utf-8"))
        for f in gj.get("features", []):
            gm = f["geometry"]
            parts = gm["coordinates"] if gm["type"] == "MultiLineString" else [gm["coordinates"]]
            out += [np.asarray(c, dtype="float64")[:, :2] for c in parts]
    out = [a for a in out if len(a) >= 2]
    if log:
        tot = sum(float(np.hypot(*np.diff(a, axis=0).T).sum()) for a in out)
        log.info("Your bench lines: %d curves, %.1f km (%s)", len(out), tot / 1000.0, p.name)
    return out


def rasterize_lines(lines, transform, shape):
    """1-cell lines on the analysis grid, plus for every line cell the index of its polyline."""
    from rasterio import features
    from shapely.geometry import LineString
    shapes = [(LineString(a), i + 1) for i, a in enumerate(lines)]
    lab = features.rasterize(shapes, out_shape=shape, transform=transform, dtype="int32", all_touched=True)
    return lab > 0, lab


def line_low_sides(lines, transform, shape, band, z_smooth, res, side_m=6.0, log=None):
    """For each polyline, which side is lower, decided for the whole line:
    the dark wall band (shade + shadow) lies on the lower side when present,
    otherwise the smoothed DEM decides. Returns a per-cell (low_c, low_r) unit
    vector field on line cells (column = east, row = south) and stats, incl.
    low_on_left: per line, True when the lower floor is on the left of the
    drawing direction as seen on the map (x east, y north)."""
    Hh, Ww = shape
    lowc = np.zeros(shape)
    lowr = np.zeros(shape)
    inv = ~transform
    n_band = n_dem = 0
    lefts = []
    for a in lines:
        cc, rr = inv * (a[:, 0], a[:, 1])
        cc, rr = np.asarray(cc), np.asarray(rr)
        # resample along the line every ~res
        seg = np.hypot(np.diff(cc), np.diff(rr))
        s = np.r_[0, np.cumsum(seg)]
        if s[-1] < 1:
            lefts.append(False)
            continue
        t = np.arange(0, s[-1], 0.5)
        c = np.interp(t, s, cc)
        r = np.interp(t, s, rr)
        dc = np.gradient(c)
        dr = np.gradient(r)
        nrm = np.hypot(dc, dr) + 1e-9
        # left normal (in column/row space)
        nc, nr = -dr / nrm, dc / nrm
        k = max(1, int(round(side_m / res)))
        band_l = band_r = 0.0
        for j in range(1, k + 1):
            cl = np.clip(np.round(c + nc * j).astype(int), 0, Ww - 1)
            rl = np.clip(np.round(r + nr * j).astype(int), 0, Hh - 1)
            cr_ = np.clip(np.round(c - nc * j).astype(int), 0, Ww - 1)
            rr_ = np.clip(np.round(r - nr * j).astype(int), 0, Hh - 1)
            band_l += band[rl, cl].sum()
            band_r += band[rr_, cr_].sum()
        tot = band_l + band_r
        if tot > 0.15 * len(t) and abs(band_l - band_r) > 0.4 * tot:
            left_low = band_l > band_r
            n_band += 1
        else:
            j = max(2, int(round(15.0 / res)))
            cl = np.clip(np.round(c + nc * j).astype(int), 0, Ww - 1)
            rl = np.clip(np.round(r + nr * j).astype(int), 0, Hh - 1)
            cr_ = np.clip(np.round(c - nc * j).astype(int), 0, Ww - 1)
            rr_ = np.clip(np.round(r - nr * j).astype(int), 0, Hh - 1)
            left_low = float((z_smooth[rl, cl] - z_smooth[rr_, cr_]).sum()) < 0
            n_dem += 1
        lefts.append(not bool(left_low))       # grid rows run south: grid-left = map-right
        sg = 1.0 if left_low else -1.0
        ci = np.clip(np.round(c).astype(int), 0, Ww - 1)
        ri = np.clip(np.round(r).astype(int), 0, Hh - 1)
        lowc[ri, ci] = sg * nc
        lowr[ri, ci] = sg * nr
    if log:
        log.info("Step direction of your lines: %d from wall shadows in the photo, %d from the terrain",
                 n_band, n_dem)
    return lowc, lowr, {"from_shadow_bands": n_band, "from_terrain": n_dem, "low_on_left": lefts}
