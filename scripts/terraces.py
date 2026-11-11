"""Quarry terrace (bench) interpretation - numpy/rasterio only.

Idea
----
A 30 m DEM shows the pit as a smooth bowl; the satellite image shows the
benches as lines (sunlit floors, shadowed faces) that run along the contours.
Open-pit benches are mined at fixed elevation levels, so if image edge
strength is projected onto elevation it repeats with the bench height H.

  1. pit mask       bright, unsaturated, non-vegetated imagery + DEM relief
  2. edge score     image edges running ALONG the DEM contours
  3. bench height   periodicity of edge strength over elevation (scan H)
  4. bench levels   phase of that periodicity, refined locally (window)
  5. staircase      volume-preserving terrace surface built on the smooth
                    DEM surface using the local overall slope + face angle

Everything here is INTERPRETED data. The measured DEM is never modified.
"""
from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------
# Filters (numpy only)
# ----------------------------------------------------------------------


def box_blur(a, r: int):
    """Mean filter with a (2r+1)^2 window, edge-replicated. O(N)."""
    a = np.asarray(a, dtype="float64")
    if r <= 0:
        return a.copy()
    p = np.pad(a, r, mode="edge")
    c = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    c[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def gauss(a, sigma_px: float):
    """Gaussian approximation by three box blurs."""
    if sigma_px < 0.5:
        return a.astype("float64", copy=True)
    r = max(1, int(round(math.sqrt(12 * sigma_px ** 2 / 3 + 1) / 2 - 0.5)))
    out = a.astype("float64")
    for _ in range(3):
        out = box_blur(out, r)
    return out


def gauss_complex(a, sigma_px):
    return gauss(a.real, sigma_px) + 1j * gauss(a.imag, sigma_px)


def otsu(values, bins: int = 256) -> float:
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=bins)
    p = hist.astype("float64") / max(hist.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    m = np.cumsum(p * centers)
    mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m) ** 2 / (w0 * (1 - w0))
    return float(centers[np.nanargmax(sb)])


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------
# 1. Pit mask
# ----------------------------------------------------------------------


def quarry_index(rgb):
    """rgb: (H, W, 3) uint8 or float 0..1. High = bare rock/excavation."""
    x = rgb.astype("float64")
    if x.max() > 1.5:
        x = x / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    lum = x.mean(-1)
    sat = (mx - mn) / (mx + 1e-6)
    exg = (2 * g - r - b) / (r + g + b + 1e-6)  # excess green (vegetation)
    return lum - 0.8 * sat - 2.0 * np.clip(exg, 0, None), lum, sat, exg


def water_index(rgb):
    x = rgb.astype("float64") / (255.0 if rgb.max() > 1.5 else 1.0)
    lum = x.mean(-1)
    return (lum < 0.22) & (x[..., 2] >= x[..., 0] - 0.01)


def label_components(mask, transform):
    """Label 8-connected components using rasterio (no scipy). Returns
    (labels int32, list of shapely polygons in CRS units, one per label)."""
    from rasterio import features
    from shapely.geometry import shape

    polys = [shape(g) for g, v in features.shapes(mask.astype("uint8"), mask=mask.astype(bool),
                                                  connectivity=8, transform=transform) if v == 1]
    if not polys:
        return np.zeros(mask.shape, "int32"), []
    labels = features.rasterize(((p, i + 1) for i, p in enumerate(polys)), out_shape=mask.shape,
                                transform=transform, dtype="int32")
    return labels, polys


def detect_pit(rgb, slope_deg, z0, transform, res, *, threshold=None, min_area_m2=20000,
               max_hole_m2=50000, min_relief_m=10.0, min_mean_slope_deg=4.0, close_m=6.0,
               smooth_m=12.0, relief_window_m=150.0, min_local_relief_m=3.0,
               keep_polygon=None):
    """Return (mask bool, polygons, info). rgb must be on the same grid."""
    from rasterio import features
    from shapely.geometry import Polygon

    qi, lum, sat, exg = quarry_index(rgb)
    valid = rgb.max(-1) > 0
    qi = gauss(np.where(valid, qi, 0), smooth_m / res)
    if threshold is None:
        # two-level Otsu: first split vegetation/dark from the rest, then
        # split bright bare rock (excavation, dumps) from fields and soil
        t1 = otsu(qi[valid])
        thr = otsu(qi[valid & (qi > t1)])
    else:
        thr = threshold
    cand = (qi > thr) & valid
    # DEM gate: excavations and dumps have local relief; bright flat fields,
    # yards and roads do not. (Small flat holes such as a pit floor are
    # restored by the closing and hole filling below.)
    if min_local_relief_m > 0:
        sp = relief_window_m / res / 2
        m1 = gauss(z0, sp)
        local_std = np.sqrt(np.clip(gauss(z0 * z0, sp) - m1 * m1, 0, None))
        cand &= local_std > min_local_relief_m
    # closing (dilate then erode) to bridge roads/shadows inside the pit
    rc = max(1, int(round(close_m / res)))
    cand = box_blur(box_blur(cand.astype(float), rc) > 0.01, rc) > 0.99
    labels, polys = label_components(cand, transform)
    keep = []
    info = {"threshold": thr, "candidates": len(polys), "rejected": []}
    px_area = res * res
    for i, p in enumerate(polys):
        # fill small holes
        holes = [h for h in p.interiors if Polygon(h).area > max_hole_m2]
        p = Polygon(p.exterior, holes)
        sel = labels == i + 1
        area = p.area
        if area < min_area_m2:
            continue
        zz = z0[sel]
        relief = float(np.percentile(zz, 98) - np.percentile(zz, 2)) if zz.size else 0
        ms = float(slope_deg[sel].mean()) if zz.size else 0
        why = None
        if relief < min_relief_m and ms < min_mean_slope_deg:
            why = f"flat (relief {relief:.1f} m, mean slope {ms:.1f} deg)"
        if keep_polygon is not None and not p.intersects(keep_polygon):
            why = "outside AOI"
        if why:
            info["rejected"].append({"area_m2": round(area), "reason": why})
            continue
        keep.append(p)
    mask = np.zeros(cand.shape, bool)
    if keep:
        mask = features.rasterize(((p, 1) for p in keep), out_shape=cand.shape,
                                  transform=transform, dtype="uint8").astype(bool)
    info["kept"] = len(keep)
    info["area_m2"] = float(mask.sum() * px_area)
    return mask, keep, info


# ----------------------------------------------------------------------
# 2. Edge score
# ----------------------------------------------------------------------


def contour_edge_score(lum, z_smooth, res, *, fine_m=1.5, coarse_m=25.0, norm_m=40.0):
    """Strength of image edges whose gradient is parallel to the DEM gradient,
    i.e. lines that run along the contours (bench crests and toes)."""
    bp = gauss(lum, fine_m / res / 2) - gauss(lum, coarse_m / res)
    gr, gc = np.gradient(bp)
    zr, zc = np.gradient(z_smooth)
    zn = np.hypot(zr, zc) + 1e-9
    along = np.abs(gr * zr / zn + gc * zc / zn)
    across = np.abs(gr * (-zc) / zn + gc * zr / zn)
    contrast = gauss(np.hypot(gr, gc), norm_m / res) + 1e-6
    # favour edges that are clearly more along-slope than across-slope
    score = np.clip(along - 0.5 * across, 0, None) / contrast
    return score


# ----------------------------------------------------------------------
# 3-4. Bench height and levels
# ----------------------------------------------------------------------


def coherence_spectrum(z, w, h_values):
    """R(H) = |sum w exp(2 pi i z / H)| / sum w, via a fine elevation histogram."""
    lo, hi = float(z.min()), float(z.max())
    nb = max(64, int((hi - lo) / 0.05))
    hist, edges = np.histogram(z, bins=nb, range=(lo, hi + 1e-6), weights=w)
    zc = 0.5 * (edges[:-1] + edges[1:])
    tot = hist.sum() + 1e-12
    R = np.empty(len(h_values))
    C = np.empty(len(h_values), complex)
    for i, H in enumerate(h_values):
        c = (hist * np.exp(2j * np.pi * zc / H)).sum()
        C[i] = c / tot
        R[i] = abs(c) / tot
    return R, C


def estimate_bench_height(z, w, base_w, h_min, h_max, step=0.05):
    """Return dict with H, phase level, spectra. base_w = uniform weights in
    the same zone (baseline from the elevation distribution itself)."""
    hs = np.arange(h_min, h_max + 1e-9, step)
    Re, Ce = coherence_spectrum(z, w, hs)
    Rb, _ = coherence_spectrum(z, base_w, hs)
    score = Re - Rb
    i = int(np.argmax(score))
    H = float(hs[i])
    level = float((np.angle(Ce[i]) / (2 * np.pi)) * H) % H
    return {"H": H, "R": float(Re[i]), "R_baseline": float(Rb[i]), "score": float(score[i]),
            "edge_level_m": level, "h_values": hs, "R_edges": Re, "R_base": Rb}


def null_scores(z, w, zone, h_min, h_max, n=8, seed=0, step=0.05):
    """Best spectrum score after shifting the edge map against the DEM.

    Real benches put edges at specific elevations; shifting the edge map
    destroys that, while generic line patterns (roads, field edges, texture)
    keep producing similar scores. Returns the list of null maxima."""
    rng = np.random.default_rng(seed)
    rows, cols = np.where(zone)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    hh, ww = r1 - r0, c1 - c0
    sub_w = w[r0:r1, c0:c1]
    sub_z = z[r0:r1, c0:c1]
    sub_zone = zone[r0:r1, c0:c1]
    out = []
    for _ in range(n):
        dy = int(rng.uniform(0.25, 0.75) * hh) * rng.choice([-1, 1])
        dx = int(rng.uniform(0.25, 0.75) * ww) * rng.choice([-1, 1])
        ws = np.roll(np.roll(sub_w, dy, 0), dx, 1) * sub_zone
        if ws.sum() <= 0:
            continue
        e = estimate_bench_height(sub_z[sub_zone], ws[sub_zone], np.ones(int(sub_zone.sum())),
                                  h_min, h_max, step)
        out.append(e["score"])
    return out


def local_levels(z, w, H, sigma_px, global_level, prior=0.05):
    """Field of local 'face-centre' level (mod H) and its confidence 0..1."""
    e = w * np.exp(2j * np.pi * z / H)
    cw = gauss_complex(e, sigma_px)
    ww = gauss(w, sigma_px) + 1e-12
    conf = np.clip(np.abs(cw) / ww, 0, 1)
    glob = np.exp(2j * np.pi * global_level / H)
    c = cw + prior * ww * glob
    level = (np.angle(c) / (2 * np.pi)) * H
    return level, conf


# ----------------------------------------------------------------------
# 5. Staircase
# ----------------------------------------------------------------------


def staircase(z0, level, H, phi):
    """Volume-preserving bench profile on the smooth surface z0.

    u = (z0 - level)/H ; faces are centred on integer u and occupy a fraction
    phi of each period (phi = tan(overall slope) / tan(face angle)); benches
    are flat between faces. mean(z - z0) = 0 over a period.
    Returns (z, is_face bool, bench index int).
    """
    u = (z0 - level) / H
    n = np.floor(u + 0.5)
    r = u - n  # in [-0.5, 0.5)
    half = phi / 2
    g = np.where(r < -half, n - 0.5, np.where(r >= half, n + 0.5, n - 0.5 + (r + half) / phi))
    face = np.abs(r) < half
    return level + H * g, face, n.astype("int32")


def overall_slope_deg(z_smooth, res):
    gr, gc = np.gradient(z_smooth, res)
    return np.degrees(np.arctan(np.hypot(gr, gc)))
