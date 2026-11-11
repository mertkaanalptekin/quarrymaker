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


def bare_ground(rgb, res, within, smooth_m=12.0, min_gap=0.04):
    """Bare rock / soil inside `within` (benches are cut into bare ground, so
    vegetated slopes inside a large hand-drawn outline are left out).

    Uses colour saturation, which, unlike brightness, barely changes between
    sunlit and shadowed rock. The saturation inside the outline is split in two
    (Otsu); the split is used only when the two groups clearly differ (bare
    rock and vegetation both present). Otherwise everything is kept."""
    _, _, sat, _ = quarry_index(rgb)
    has = rgb.max(-1) > 0
    ss = gauss(np.where(has, sat, 0), smooth_m / res)
    inside = within & has
    if inside.sum() < 100:
        return inside
    v = ss[inside]
    t = otsu(v)
    lo, hi = v[v < t], v[v >= t]
    if lo.size < 0.05 * v.size or hi.size < 0.05 * v.size or float(hi.mean() - lo.mean()) < min_gap:
        return inside
    return inside & (ss < t)


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
    """Field of local 'face-centre' level (mod H) and its confidence 0..1.
    H and global_level may be scalars or per-pixel arrays."""
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


def planar_weights(w, z_smooth, zone, res):
    """Down-weight edges where the smoothed surface is strongly curved (pit rim
    and floor). There a 30 m DEM squeezes elevations, which biases the bench
    height low; on the straight middle of a wall it does not."""
    gr, gc = np.gradient(z_smooth, res)
    grr, _ = np.gradient(gr, res)
    _, gcc = np.gradient(gc, res)
    lap = np.abs(grr + gcc)
    lap0 = float(np.median(lap[zone])) if zone.any() else 1.0
    return w / (1.0 + (lap / max(lap0, 1e-9)) ** 2)


def overall_slope_deg(z_smooth, res):
    gr, gc = np.gradient(z_smooth, res)
    return np.degrees(np.arctan(np.hypot(gr, gc)))


# ----------------------------------------------------------------------
# Regional analysis (large or multi-pit complexes)
# ----------------------------------------------------------------------
# A mining complex holds several pits and dumps, each with its own bench
# levels (and sometimes its own bench height), plus areas excavated after
# the DEM was captured. One global periodicity mixes them until it cancels
# out, so the pattern is tested region by region instead. Benches are built
# only where a region shows significant evidence; elsewhere the measured
# terrain is kept.


def _neff(w):
    s = float(w.sum())
    return s * s / max(float((w * w).sum()), 1e-12)


def _null_scores(spec, Rb, w2d, zone2d, wf, rng, n_null):
    """Best spectrum scores of edge maps that cannot carry a real bench signal.

    Two kinds, and the stricter wins:
      * circular shift of the weights along the zone pixels (same evidence
        meets other elevations);
      * spatial shift of the edge map (keeps regular line patterns such as
        field or road grids, which on a uniform slope also look periodic in
        elevation). Shifts that pile the evidence onto few pixels are skipped,
        so the comparison stays fair."""
    n = len(wf)
    neff = _neff(wf)
    out = []
    for _ in range(n_null):
        Rn, _ = spec(np.roll(wf, int(rng.uniform(0.2, 0.8) * n)))
        out.append(float((Rn - Rb).max()))
    hh, ww = zone2d.shape
    got, tries = 0, 0
    while got < n_null and tries < 6 * n_null:
        tries += 1
        dy = int(rng.uniform(0.15, 0.85) * hh) * int(rng.choice([-1, 1]))
        dx = int(rng.uniform(0.15, 0.85) * ww) * int(rng.choice([-1, 1]))
        wr = np.roll(np.roll(w2d, dy, 0), dx, 1)[zone2d]
        if wr.sum() <= 0 or _neff(wr) < 0.6 * neff:
            continue
        Rn, _ = spec(wr)
        out.append(float((Rn - Rb).max()))
        got += 1
    return out


class _Spectrum:
    """Coherence spectra of one region; the elevation->phase table is built
    once and reused for the real and all shifted (null) weight maps."""

    def __init__(self, z, h_values, bin_m=0.1):
        lo = float(z.min())
        nb = max(32, int((float(z.max()) - lo) / bin_m) + 1)
        self.idx = np.clip(((z - lo) / bin_m).astype(np.int64), 0, nb - 1)
        self.nb = nb
        zc = lo + (np.arange(nb) + 0.5) * bin_m
        self.E = np.exp(2j * np.pi * zc[None, :] / np.asarray(h_values, float)[:, None])

    def __call__(self, w):
        hist = np.bincount(self.idx, weights=w, minlength=self.nb)
        c = self.E @ hist / (hist.sum() + 1e-12)
        return np.abs(c), c


def regional_scan(zl, w, zone, res, tile_m, h_min, h_max, fixed_h=None, n_null=8,
                  min_zone_frac=0.08, seed=0, step_h=0.1):
    """Test bench periodicity in overlapping square regions (50% overlap).

    Returns a list of dicts: r0, r1, c0, c1 (pixel window), H, level (face-centre
    level mod H), score (coherence above the elevation baseline), null (best
    score of shifted edge maps), sig (score / null), n (bench-zone pixels)."""
    rng = np.random.default_rng(seed)
    hs = np.array([fixed_h], float) if fixed_h else np.arange(h_min, h_max + 1e-9, step_h)
    tp = max(16, int(round(tile_m / res)))
    step = max(8, tp // 2)
    Hh, Ww = zone.shape
    tiles = []
    for r0 in range(0, max(1, Hh - step), step):
        for c0 in range(0, max(1, Ww - step), step):
            r1, c1 = min(Hh, r0 + tp), min(Ww, c0 + tp)
            zt = zone[r0:r1, c0:c1]
            n = int(zt.sum())
            if n < max(500, min_zone_frac * zt.size):
                continue
            ww = w[r0:r1, c0:c1]
            wf = ww[zt]
            if wf.sum() <= 0:
                continue
            spec = _Spectrum(zl[r0:r1, c0:c1][zt], hs)
            Re, Ce = spec(wf)
            Rb, _ = spec(np.ones_like(wf))
            score = Re - Rb
            i = int(np.argmax(score))
            nulls = _null_scores(spec, Rb, ww, zt, wf, rng, n_null)
            null = max(nulls) if nulls else 0.0
            H = float(hs[i])
            tiles.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "n": n, "H": H,
                          "curve": score, "hs": hs,
                          "level": float((np.angle(Ce[i]) / (2 * np.pi)) * H % H),
                          "score": float(score[i]), "null": null,
                          "sig": float(score[i] / max(null, 1e-6))})
    return tiles


def tile_is_significant(t, sig_min=1.5, score_min=0.05):
    return t["sig"] >= sig_min and t["score"] >= score_min


def tile_fields(tiles, shape, sig_lo=1.3, sig_hi=2.0, score_min=0.05):
    """Blend tile results into smooth per-pixel fields with tent windows.

    evidence: 0..1, how strongly the surrounding regions show benches.
    H: bench height, weighted by evidence (NaN where no region has evidence)."""
    num_e = np.zeros(shape, "float32")
    den = np.zeros(shape, "float32")
    num_h = np.zeros(shape, "float32")
    den_h = np.zeros(shape, "float32")
    for t in tiles:
        h, w = t["r1"] - t["r0"], t["c1"] - t["c0"]
        ty = 1 - np.abs(np.linspace(-1, 1, h + 2)[1:-1])
        tx = 1 - np.abs(np.linspace(-1, 1, w + 2)[1:-1])
        tri = np.outer(ty, tx).astype("float32")
        e = float(smoothstep(t["sig"], sig_lo, sig_hi)) if t["score"] >= score_min else 0.0
        sl = (slice(t["r0"], t["r1"]), slice(t["c0"], t["c1"]))
        den[sl] += tri
        if e > 0:
            num_e[sl] += tri * e
            num_h[sl] += tri * e * t["H"]
            den_h[sl] += tri * e
    evidence = np.where(den > 0, num_e / np.maximum(den, 1e-6), 0).astype("float32")
    hfield = np.where(den_h > 0, num_h / np.maximum(den_h, 1e-6), np.nan).astype("float32")
    return evidence, hfield


def tile_zones(tiles, sig_min=1.5, score_min=0.05, h_tol=0.15):
    """Group significant overlapping regions into 'bench zones' (usually one pit
    or dump each). Two regions join only if their bench heights agree within
    h_tol and their bench levels agree within a quarter bench, so neighbouring
    pits with different level systems stay apart."""
    sig = [i for i, t in enumerate(tiles) if tile_is_significant(t, sig_min, score_min)]
    parent = {i: i for i in sig}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def compatible(ta, tb):
        if not (ta["r0"] < tb["r1"] and tb["r0"] < ta["r1"] and ta["c0"] < tb["c1"] and tb["c0"] < ta["c1"]):
            return False
        hm = 0.5 * (ta["H"] + tb["H"])
        if abs(ta["H"] - tb["H"]) > h_tol * hm:
            # heights disagree: still the same pit if one tile's curve peaks near
            # the other's height (noisy region)
            ia = int(np.argmin(np.abs(ta["hs"] - tb["H"])))
            ib = int(np.argmin(np.abs(tb["hs"] - ta["H"])))
            if ta["curve"][ia] < 0.7 * ta["score"] and tb["curve"][ib] < 0.7 * tb["score"]:
                return False
        d = (ta["level"] - tb["level"] + hm / 2) % hm - hm / 2
        return abs(d) <= hm / 4

    for a in sig:
        for b in sig:
            if a < b and compatible(tiles[a], tiles[b]):
                parent[find(a)] = find(b)
    groups = {}
    for i in sig:
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: -sum(tiles[i]["n"] for i in g))


def zone_solution(zl, w, zone, tiles, group, fixed_h=None, n_null=8, seed=1):
    """Bench height and level system of one zone, from all its regions together,
    plus a zone-level significance test (circular shift of the edge weights)."""
    sel = np.zeros(zone.shape, bool)
    for i in group:
        t = tiles[i]
        sel[t["r0"]:t["r1"], t["c0"]:t["c1"]] = True
    sel &= zone
    z = zl[sel]
    wf = w[sel]
    if fixed_h:
        H = float(fixed_h)
    else:
        curve = np.sum([tiles[i]["curve"] * tiles[i]["n"] for i in group], axis=0)
        H0 = float(tiles[group[0]]["hs"][int(np.argmax(curve))])
        # refine on this zone's own pixels (0.02 m steps around the coarse peak)
        hs = np.arange(max(1.0, H0 - 0.3), H0 + 0.3 + 1e-9, 0.02)
        sp = _Spectrum(z, hs)
        H = float(hs[int(np.argmax(sp(wf)[0] - sp(np.ones_like(wf))[0]))])
    spec = _Spectrum(z, [H])
    Re, Ce = spec(wf)
    Rb, _ = spec(np.ones_like(wf))
    score = float(Re[0] - Rb[0])
    rng = np.random.default_rng(seed)
    nulls = []
    if n_null:
        rows, cols = np.where(sel)
        win = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        nulls = _null_scores(spec, Rb, w[win], sel[win], wf, rng, n_null)
    null = max(max(nulls, default=0.0), 1e-6)
    level = float((np.angle(Ce[0]) / (2 * np.pi)) * H % H)
    return {"H": H, "level": level, "score": score, "null": null, "sig": score / null, "pixels": int(sel.sum())}


def _groups_touch(tiles, ga, gb):
    for a in ga:
        ta = tiles[a]
        for b in gb:
            tb = tiles[b]
            if ta["r0"] < tb["r1"] and tb["r0"] < ta["r1"] and ta["c0"] < tb["c1"] and tb["c0"] < ta["c1"]:
                return True
    return False


def merge_zones(zl, w, zone, tiles, groups, fixed_h=None, keep=0.8):
    """Merge neighbouring zones that share one level system.

    Two touching zones are merged when the coherence of the union stays at
    least `keep` x the pixel-weighted coherence of the two apart, i.e. their
    benches really sit on the same elevations (one pit split by noise).
    Different pits (other levels or heights) lose coherence and stay apart."""
    groups = [list(g) for g in groups]
    cache = {}

    def sol(g):
        key = tuple(sorted(g))
        if key not in cache:
            cache[key] = zone_solution(zl, w, zone, tiles, list(key), fixed_h=fixed_h, n_null=0)
        return cache[key]
    while True:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if not _groups_touch(tiles, groups[i], groups[j]):
                    continue
                a, b = sol(groups[i]), sol(groups[j])
                u = sol(groups[i] + groups[j])
                ref = (a["score"] * a["pixels"] + b["score"] * b["pixels"]) / max(1, a["pixels"] + b["pixels"])
                ratio = u["score"] / max(ref, 1e-6)
                if ratio >= keep and (best is None or ratio > best[0]):
                    best = (ratio, i, j)
        if best is None:
            break
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]
    return sorted(groups, key=lambda g: -sum(tiles[k]["n"] for k in g))


def zone_fields(tiles, zones, shape):
    """Per-pixel zone index (-1 = none) and evidence (0..1): each pixel belongs to
    the zone whose significant region covers it most centrally."""
    best = np.zeros(shape, "float32")
    zidx = np.full(shape, -1, "int16")
    for zi, g in enumerate(zones):
        for i in g:
            t = tiles[i]
            h, w = t["r1"] - t["r0"], t["c1"] - t["c0"]
            ty = 1 - np.abs(np.linspace(-1, 1, h + 2)[1:-1])
            tx = 1 - np.abs(np.linspace(-1, 1, w + 2)[1:-1])
            tri = np.outer(ty, tx).astype("float32")
            sl = (slice(t["r0"], t["r1"]), slice(t["c0"], t["c1"]))
            better = tri > best[sl]
            best[sl] = np.where(better, tri, best[sl])
            zidx[sl] = np.where(better, zi, zidx[sl])
    # central half of a region (tent >= 0.25) counts fully; fade towards its edge
    evidence = np.clip(best / 0.25, 0, 1).astype("float32")
    return zidx, evidence
