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
                          "score": float(score[i]), "null": null, "nulls": nulls,
                          "null_med": float(np.median(nulls)) if nulls else 0.0,
                          "sig": float(score[i] / max(null, 1e-6))})
    return tiles


def apply_sitewide_threshold(tiles, sig_min):
    """Correct for testing many regions at once.

    With ~100+ regions, a few pass a per-region test by chance. The null
    scores of ALL regions are pooled (each relative to its own median null
    score) into one site-wide chance level: a region must beat its own best
    null by sig_min AND the largest chance ratio seen anywhere on the site.
    Each tile's 'sig' becomes this combined value (>= sig_min means significant)."""
    ratios = [n / max(t["null_med"], 1e-6) for t in tiles for n in t.get("nulls", [])]
    rstar = max(ratios) if ratios else 0.0
    for t in tiles:
        t["sig_local"] = t["sig"]
        site = t["score"] / max(rstar * t["null_med"], 1e-6) * sig_min if rstar > 0 else t["sig"]
        t["sig"] = float(min(t["sig"], site))
    return rstar


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


# ----------------------------------------------------------------------
# Bench lines traced as lines (Canny edges) and benches carved per floor
# ----------------------------------------------------------------------


def canny_lines(lum, res, sigma_m=1.0, low_q=0.70, high_q=0.90, min_len_m=18.0, within=None,
                sun_azimuth_deg=None, return_dropped=False):
    """Bench lines from the photo: Canny edges (gradient, thin to 1 px along
    the gradient, hysteresis), kept only as long connected lines.

    Bench faces and their shadows draw long, continuous light/dark edges;
    rubble, blocks and texture give short broken ones. Returns a bool mask."""
    from rasterio import features
    from rasterio.transform import Affine

    g = gauss(lum, sigma_m / res)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    nms = np.zeros_like(mag, dtype=bool)
    # neighbour offsets for 4 quantised gradient directions (0, 45, 90, 135 deg)
    for a0, (dy, dx) in ((0, (0, 1)), (45, (1, 1)), (90, (1, 0)), (135, (1, -1))):
        sel = (np.abs(ang - a0) <= 22.5) | (np.abs(ang - a0) >= 157.5) if a0 == 0 else (np.abs(ang - a0) <= 22.5)
        fwd = np.roll(np.roll(mag, -dy, 0), -dx, 1)
        bwd = np.roll(np.roll(mag, dy, 0), dx, 1)
        nms |= sel & (mag >= fwd) & (mag >= bwd)
    dropped = np.zeros_like(nms)
    if sun_azimuth_deg is not None:
        # A shadow lies on the side away from the sun. At the bench edge the photo
        # turns bright -> dark moving away from the sun (brightness gradient points
        # TOWARDS the sun); at the far end of the shadow it turns dark -> bright
        # (gradient points away). Keep bench edges, drop shadow-end edges.
        az = np.radians(float(sun_azimuth_deg))
        toward = -np.cos(az) * gy / (mag + 1e-12) + np.sin(az) * gx / (mag + 1e-12)  # rows point south
        dropped = nms & (toward < -0.3)
        nms = nms & ~dropped
    ref = mag[within] if within is not None and within.any() else mag
    lo, hi = np.quantile(ref, low_q), np.quantile(ref, high_q)
    weak = nms & (mag > lo)
    if within is not None:
        weak &= within
    strong = weak & (mag > hi)
    out = np.zeros(mag.shape, bool)
    if not weak.any():
        return out
    tf = Affine(res, 0, 0, 0, -res, 0)
    labels = features.rasterize(
        ((geom, i + 1) for i, (geom, v) in enumerate(features.shapes(weak.astype("uint8"), mask=weak,
                                                                     connectivity=8, transform=tf))
         if v == 1), out_shape=weak.shape, transform=tf, dtype="int32")
    n = int(labels.max())
    if n == 0:
        return out
    cnt = np.bincount(labels.ravel(), minlength=n + 1)
    has_strong = np.bincount(labels.ravel(), weights=strong.ravel().astype(float), minlength=n + 1) > 0
    # extent of each component (a compact blob of edges is not a line)
    rows, cols = np.nonzero(labels)
    lab = labels[rows, cols]
    rmin = np.full(n + 1, 1e9); rmax = np.full(n + 1, -1e9)
    cmin = np.full(n + 1, 1e9); cmax = np.full(n + 1, -1e9)
    np.minimum.at(rmin, lab, rows); np.maximum.at(rmax, lab, rows)
    np.minimum.at(cmin, lab, cols); np.maximum.at(cmax, lab, cols)
    ext = np.hypot(rmax - rmin, cmax - cmin) * res
    length = cnt * res
    keep = has_strong & (length >= min_len_m) & (ext >= 0.45 * length) & (ext >= min_len_m * 0.8)
    keep[0] = False
    if return_dropped:
        return keep[labels], dropped & (mag > hi)
    return keep[labels]


def _shift(a, dy, dx, fill=0.0):
    """Shift without wrap-around."""
    out = np.full_like(a, fill)
    H, W = a.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def benched_surface(z0, lines, res, ell_m=30.0, cut=0.0, iters=600, tol=1e-4):
    """Piecewise-flat sheet that follows the DEM at large scale but may break
    only along bench lines.

    Minimises  sum_cells (z - z0)^2 / ell^2  +  sum_neighbours w_pq (z_p - z_q)^2
    with w_pq = 1 inside floors and `cut` across bench-line pixels: between two
    lines the sheet becomes a flat bench, and at a line it steps by the drop
    the DEM requires. Solved by Jacobi-preconditioned conjugate gradients
    (numpy only). ell is in metres (pixels = ell / res)."""
    lam = (res / ell_m) ** 2
    open_ = (~lines).astype("float64")
    # edge weights between (r, c) and its right / lower neighbour
    wx = np.where((open_ * _shift(open_, 0, -1)) > 0, 1.0, cut)
    wy = np.where((open_ * _shift(open_, -1, 0)) > 0, 1.0, cut)
    wx[:, -1] = 0.0
    wy[-1, :] = 0.0

    def A(z):
        dxp = wx * (z - _shift(z, 0, -1))           # to the right neighbour
        dyp = wy * (z - _shift(z, -1, 0))           # to the lower neighbour
        return lam * z + dxp - _shift(dxp, 0, 1) + dyp - _shift(dyp, 1, 0)

    diag = lam + wx + _shift(wx, 0, 1) + wy + _shift(wy, 1, 0)
    b = lam * z0
    x = z0.astype("float64").copy()
    r = b - A(x)
    zv = r / diag
    p = zv.copy()
    rz = float((r * zv).sum())
    bn = float(np.sqrt((b * b).sum())) + 1e-12
    for _ in range(iters):
        Ap = A(p)
        alpha = rz / float((p * Ap).sum())
        x += alpha * p
        r -= alpha * Ap
        if float(np.sqrt((r * r).sum())) / bn < tol:
            break
        zv = r / diag
        rz_new = float((r * zv).sum())
        p = zv + (rz_new / rz) * p
        rz = rz_new
    return x


def quarry_area(rgb, z0, res, transform, keep_polygon=None, min_area_ha=5.0, open_m=12.0,
                close_m=25.0, max_hole_ha=8.0, min_relief_m=5.0, threshold=None):
    """The whole quarry area (pits, dumps, internal roads, yards), separated
    from farmland, villages, greenhouses and factories.

    1. bare-rock candidates: quarry index (bright, unsaturated, not green),
       two-level Otsu, kept where the DEM has local relief;
    2. opening (erode, dilate) cuts thin bright links such as roads, so fields
       and buildings along a road do not join the quarry;
    3. only large bodies with real relief survive (greenhouses and roofs are flat);
    4. closing and hole filling make one continuous area (internal roads,
       vegetation islands and yards belong to the quarry).
    Returns (mask, polygons, info)."""
    from rasterio import features
    from shapely.geometry import Polygon

    zs = gauss(z0, 10 / res)
    slope = overall_slope_deg(zs, res)
    cand, _, info = detect_pit(rgb, slope, z0, transform, res, min_area_m2=0.0, min_relief_m=0.0,
                               min_mean_slope_deg=0.0, keep_polygon=None, threshold=threshold)
    thr_used = info["threshold"]
    ro = max(1, int(round(open_m / res)))
    opened = box_blur(box_blur(cand.astype(float), ro) > 0.999, ro) > 0.001
    labels, polys = label_components(opened, transform)
    m1 = gauss(z0, 75 / res)
    relief = np.sqrt(np.clip(gauss(z0 * z0, 75 / res) - m1 * m1, 0, None))
    keep = []
    for i, p in enumerate(polys):
        if p.area < min_area_ha * 1e4:
            continue
        sel = labels == i + 1
        if float(np.percentile(relief[sel], 75)) < min_relief_m:
            continue
        if keep_polygon is not None and not p.intersects(keep_polygon):
            continue
        keep.append(p)
    mask = np.zeros(cand.shape, bool)
    if keep:
        mask = features.rasterize(((p, 1) for p in keep), out_shape=cand.shape, transform=transform,
                                  dtype="uint8").astype(bool)
    rc = max(1, int(round(close_m / res)))
    mask = box_blur(box_blur(mask.astype(float), rc) > 0.001, rc) > 0.999
    # fill holes smaller than max_hole_ha
    labels, polys = label_components(mask, transform)
    filled = []
    for p in polys:
        holes = [h for h in p.interiors if Polygon(h).area > max_hole_ha * 1e4]
        filled.append(Polygon(p.exterior, holes))
    if filled:
        mask = features.rasterize(((p, 1) for p in filled), out_shape=cand.shape, transform=transform,
                                  dtype="uint8").astype(bool)
    return mask, filled, {"area_m2": float(mask.sum() * res * res), "bodies": len(filled), "threshold": thr_used,
                          "source": "quarry area: bare rock + relief, roads cut, large bodies only"}



def bench_lines_from_file(ortho_path, dst_transform, dst_shape, crs, within, block_m=600.0,
                          sigma_m=1.0, min_len_m=18.0, sun_azimuth_deg=135.0, log=None):
    """Bench lines traced at the photo's full resolution, block by block, and
    transferred to the analysis grid (a cell is a line cell when any photo
    pixel in it is). Only blocks touching `within` are processed."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds

    Hd, Wd = dst_shape
    res_d = dst_transform.a
    bpx = max(64, int(round(block_m / res_d)))
    out = np.zeros(dst_shape, bool)
    margin = 40.0
    with rasterio.open(ortho_path) as src:
        res_s = src.res[0]
        blocks = [(r0, c0) for r0 in range(0, Hd, bpx) for c0 in range(0, Wd, bpx)
                  if within[r0:r0 + bpx, c0:c0 + bpx].any()]
        for n, (r0, c0) in enumerate(blocks):
            r1, c1 = min(Hd, r0 + bpx), min(Wd, c0 + bpx)
            x0 = dst_transform.c + c0 * res_d - margin
            x1 = dst_transform.c + c1 * res_d + margin
            y1 = dst_transform.f - r0 * res_d + margin
            y0 = dst_transform.f - r1 * res_d - margin
            win = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
            rgb = src.read([1, 2, 3], window=win, boundless=True, fill_value=0).astype("float32")
            lum = rgb.mean(0) / 255.0
            lines = canny_lines(lum, res_s, sigma_m=sigma_m, min_len_m=min_len_m, within=rgb.max(0) > 0,
                                sun_azimuth_deg=sun_azimuth_deg)
            sub_tf = rasterio.transform.from_origin(dst_transform.c + c0 * res_d,
                                                    dst_transform.f - r0 * res_d, res_d, res_d)
            tmp = np.zeros((r1 - r0, c1 - c0), "uint8")
            reproject(lines.astype("uint8"), tmp, src_transform=src.window_transform(win), src_crs=crs,
                      dst_transform=sub_tf, dst_crs=crs, resampling=Resampling.max)
            out[r0:r1, c0:c1] = tmp > 0
            if log and (n + 1) % 20 == 0:
                log.info("  bench lines: %d/%d blocks", n + 1, len(blocks))
    return out & within


def cut_benches(z0, lines, within, res, ell_m=30.0, island_max_m2=400.0, dz_max=12.0,
                merge_m=4.0, transform=None):
    """Benches cut into the measured surface along the bench lines.

    dz = sheet(z0, lines) - sheet(z0, no lines): the change that the lines
    alone cause (see benched_surface). Hills and low points stay as measured;
    between lines the ground becomes flat benches and each line becomes a face
    whose height is the elevation drop the DEM requires there.
    Small fully enclosed islands (piles, bushes) are neutralised and the change
    is limited to +-dz_max. Returns (dz, info)."""
    from rasterio import features
    from rasterio.transform import Affine

    rows = np.where(within.any(1))[0]
    cols = np.where(within.any(0))[0]
    if rows.size == 0:
        return np.zeros(z0.shape, "float32"), {}
    pad = int(round(3 * ell_m / res))
    r0, r1 = max(0, rows[0] - pad), min(z0.shape[0], rows[-1] + 1 + pad)
    c0, c1 = max(0, cols[0] - pad), min(z0.shape[1], cols[-1] + 1 + pad)
    sl = (slice(r0, r1), slice(c0, c1))
    zc, lc, wc = z0[sl].astype("float64"), lines[sl], within[sl]
    # a face often shows as two close edges (crest and toe, or the two sides of its
    # shadow): widen the lines so such a pair becomes one face, not a thin floor
    if merge_m > 0:
        lc = box_blur(lc.astype(float), max(1, int(round(merge_m / 2 / res)))) > 0.01
    dz = benched_surface(zc, lc, res, ell_m) - benched_surface(zc, np.zeros_like(lc), res, ell_m)
    # neutralise small enclosed islands (closed little outlines become pillars or holes)
    tf = transform or Affine(res, 0, 0, 0, -res, 0)
    free = wc & ~lc
    shapes = [(g, i + 1) for i, (g, v) in enumerate(features.shapes(free.astype("uint8"), mask=free,
                                                                       connectivity=4, transform=tf)) if v == 1]
    n_isl = 0
    if shapes:
        labels = features.rasterize(shapes, out_shape=free.shape, transform=tf, dtype="int32")
        cnt = np.bincount(labels.ravel())
        small = cnt * res * res < island_max_m2
        small[0] = False
        isl = small[labels]
        n_isl = int(small.sum())
        if isl.any():
            ok = ~isl & ~lc
            num = gauss(np.where(ok, dz, 0.0), 6 / res)
            den = gauss(ok.astype(float), 6 / res) + 1e-9
            dz = np.where(isl | (lc & (gauss(isl.astype(float), 1.5) > 0.2)), num / den, dz)
    dz = np.clip(dz, -dz_max, dz_max)
    out = np.zeros(z0.shape, "float32")
    out[sl] = dz
    # step heights at the lines: the change across each line pixel
    zt = (zc + dz)
    jumps = []
    for dy, dx in ((0, 2), (2, 0)):
        a = _shift(zt, dy, dx, np.nan)
        b = _shift(zt, -dy, -dx, np.nan)
        j = np.abs(a - b)[lc & wc]
        jumps.append(j[np.isfinite(j)])
    j = np.concatenate(jumps) if jumps else np.array([])
    j = j[j > 0.3]
    info = {"islands_neutralised": n_isl,
            "line_length_km": round(float(lc[wc].sum()) * res / 1000.0, 1),
            "step_height_median_m": round(float(np.median(j)), 2) if j.size else None,
            "step_height_p75_m": round(float(np.percentile(j, 75)), 2) if j.size else None,
            "step_height_p90_m": round(float(np.percentile(j, 90)), 2) if j.size else None}
    return out, info


def _cg(apply_A, b, x0, diag, iters=800, tol=1e-5):
    x = x0.astype("float64").copy()
    r = b - apply_A(x)
    zv = r / diag
    p = zv.copy()
    rz = float((r * zv).sum())
    bn = float(np.sqrt((b * b).sum())) + 1e-12
    for _ in range(iters):
        Ap = apply_A(p)
        alpha = rz / float((p * Ap).sum())
        x += alpha * p
        r -= alpha * Ap
        if float(np.sqrt((r * r).sum())) / bn < tol:
            break
        zv = r / diag
        rz_new = float((r * zv).sum())
        p = zv + (rz_new / rz) * p
        rz = rz_new
    return x


def bench_height_from_lines(z_s, lines, within, res, max_gap_m=60.0, min_slope_deg=6.0):
    """Bench height guessed from the photo and the DEM together: on a benched
    slope consecutive bench lines lie one bench apart, so the spacing between
    neighbouring lines measured downhill, times the tangent of the large-scale
    slope, is one bench height. Returns (H or None, number of samples)."""
    gr, gc = np.gradient(z_s, res)
    g = np.hypot(gr, gc)
    ok = lines & within & (np.degrees(np.arctan(g)) > min_slope_deg)
    rr, cc = np.nonzero(ok)
    if rr.size < 50:
        return None, int(rr.size)
    if rr.size > 20000:
        sel = np.random.default_rng(0).choice(rr.size, 20000, replace=False)
        rr, cc = rr[sel], cc[sel]
    ur, uc = -gr[rr, cc] / g[rr, cc], -gc[rr, cc] / g[rr, cc]       # downhill
    Hh, Ww = lines.shape
    gaps = np.full(rr.size, np.nan)
    alive = np.ones(rr.size, bool)
    start = max(2, int(np.ceil(4.0 / res)))                          # leave the own line
    for k in range(start, int(max_gap_m / res) + 1):
        r = np.round(rr + ur * k).astype(int)
        c = np.round(cc + uc * k).astype(int)
        inb = (r >= 0) & (r < Hh) & (c >= 0) & (c < Ww)
        hit = np.zeros(rr.size, bool)
        hit[inb] = lines[r[inb], c[inb]]
        new = alive & hit
        gaps[new] = k * res
        alive &= ~hit & inb
    h = gaps * g[rr, cc]
    h = h[np.isfinite(h)]
    if h.size < 50:
        return None, int(h.size)
    return float(np.median(h)), int(h.size)


def photo_benches(z0, lines, within, res, bench_h=9.0, face_angle_deg=85.0, field_m=20.0,
                  reach_m=40.0, max_spacing_m=60.0, snap_m=10.0, dir_m=60.0, base_m=120.0,
                  feather_m=20.0, carve=0.5, blocky=True, block_m=150.0, flat_floors=True, flat_m=25.0,
                  cell_m=8.0):
    """Benches laid out by the photo's bench lines (not by the DEM's contours).

    1. At each line pixel: the bench direction (structure tensor of the
       lines), the downhill side (DEM smoothed at dir_m) and the spacing to
       the next line downhill (the bench width seen in the photo).
    2. These are spread over the quarry (normalised convolution at field_m);
       their weight fades to zero beyond reach_m from any line, so no
       benches are invented where the photo shows none.
    3. A bench number u is solved whose gradient is (downhill normal /
       spacing): its contour lines run along the photo's benches, one per
       bench. u is then snapped to whole numbers on the traced lines.
    4. Floors are flat steps of u (one bench height H each) with walls at
       face_angle_deg, laid on the DEM smoothed at base_m; the whole system is
       lowered carve*H (carved into the ground). Every wall is a full bench
       high; where the photo's benches end, the walls end.
    blocky: marble benches are cut as straight faces meeting at right angles
    (the zigzag rows in the photo). The bench directions are snapped to the
    two perpendicular directions that dominate the lines within block_m;
    the ground is divided into square cells of cell_m aligned with them and
    each cell takes one bench level, so walls run straight along cell edges
    and turn at right angles (cell_m = 0: smooth walls). Floors are then
    flattened within their own level (local average over flat_m).
    Returns (dz, info)."""
    H = float(bench_h)
    rows = np.where(within.any(1))[0]
    cols = np.where(within.any(0))[0]
    if rows.size == 0:
        return np.zeros(z0.shape, "float32"), {}
    pad = int(round(max(feather_m, reach_m) / res)) + 2
    r0, r1 = max(0, rows[0] - pad), min(z0.shape[0], rows[-1] + 1 + pad)
    c0, c1 = max(0, cols[0] - pad), min(z0.shape[1], cols[-1] + 1 + pad)
    sl = (slice(r0, r1), slice(c0, c1))
    zc, wc = z0[sl].astype("float64"), within[sl]
    from linework import thin
    sk = thin(lines[sl] & wc)
    if sk.sum() < 10:
        return np.zeros(z0.shape, "float32"), {"line_length_km": 0.0}
    Hh, Ww = zc.shape
    # 1. direction (doubled-angle tensor), downhill sign, spacing
    s = sk.astype(float)
    gy, gx = np.gradient(gauss(s, 1.0))
    tsig = 1.0 if blocky else 2.0
    jxx, jyy, jxy = gauss(gx * gx, tsig), gauss(gy * gy, tsig), gauss(gx * gy, tsig)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    if blocky:
        # dominant perpendicular pair of directions (4-fold angle average)
        wl0 = sk.astype(float)
        sb = block_m / 2 / res
        c4, s4 = gauss(wl0 * np.cos(4 * ang), sb), gauss(wl0 * np.sin(4 * ang), sb)
        coh4 = np.hypot(c4, s4) / np.maximum(gauss(wl0, sb), 1e-9)
        th_d = 0.25 * np.arctan2(s4, c4)

        def snap_ang(a):
            k = np.round((a - th_d) / (np.pi / 2))
            return np.where(coh4 > 0.25, th_d + k * np.pi / 2, a)
        ang = snap_ang(ang)
    nx, ny = np.cos(ang), np.sin(ang)
    zs = gauss(zc, dir_m / 2 / res)
    dzy, dzx = np.gradient(zs, res)
    sg = -np.sign(nx * dzx + ny * dzy)
    sg[sg == 0] = 1.0
    nx, ny = nx * sg, ny * sg                         # downhill normal
    rr, cc = np.nonzero(sk)
    gap = np.full(rr.size, np.nan)
    alive = np.ones(rr.size, bool)
    for k in range(3, int(max_spacing_m / res) + 1):
        r = np.round(rr + ny[rr, cc] * k).astype(int)
        c = np.round(cc + nx[rr, cc] * k).astype(int)
        inb = (r >= 0) & (r < Hh) & (c >= 0) & (c < Ww)
        hit = np.zeros(rr.size, bool)
        hit[inb] = sk[r[inb], c[inb]]
        new = alive & hit
        gap[new] = k * res
        alive &= ~hit & inb
    have = np.isfinite(gap)
    sp_med = float(np.median(gap[have])) if have.any() else 20.0
    # 2. spread (normalised convolution); direction via the doubled angle so +-n agree
    sig = field_m / 2 / res
    wl = np.zeros((Hh, Ww))
    wl[rr, cc] = 1.0
    c2 = gauss(wl * np.cos(2 * ang), sig)
    s2 = gauss(wl * np.sin(2 * ang), sig)
    fq = np.zeros((Hh, Ww))
    wq = np.zeros((Hh, Ww))
    fq[rr[have], cc[have]] = 1.0 / gap[have]
    wq[rr[have], cc[have]] = 1.0
    freq = gauss(fq, sig) / np.maximum(gauss(wq, sig), 1e-9)
    freq = np.where(gauss(wq, sig) > 1e-4, freq, 1.0 / sp_med)
    freq = gauss(freq, sig)
    angf = 0.5 * np.arctan2(s2, c2)
    if blocky:
        angf = snap_ang(angf)
    fx, fy = np.cos(angf), np.sin(angf)
    sgf = -np.sign(fx * dzx + fy * dzy)
    sgf[sgf == 0] = 1.0
    fx, fy = fx * sgf, fy * sgf
    # presence: 1 near lines, fading to 0 beyond reach_m
    near = gauss(wl, reach_m / 2 / res)
    pres = np.clip(near / max(float(np.percentile(near[sk], 25)), 1e-9), 0, 1) * wc
    # 3. u with grad u = -(downhill normal) * freq  (u counts benches upward)
    gxu = -fx * freq * pres
    gyu = -fy * freq * pres
    tx = 0.5 * (gxu + _shift(gxu, 0, -1)) * res
    ty = 0.5 * (gyu + _shift(gyu, -1, 0)) * res
    wx = np.ones((Hh, Ww))
    wy = np.ones((Hh, Ww))
    wx[:, -1] = 0.0
    wy[-1, :] = 0.0
    lam = (res / 300.0) ** 2                             # local consistency only (converges on any site size)

    def A(z):
        dxp = wx * (z - _shift(z, 0, -1))
        dyp = wy * (z - _shift(z, -1, 0))
        return lam * z + dxp - _shift(dxp, 0, 1) + dyp - _shift(dyp, 1, 0)

    wtx, wty = wx * tx, wy * ty
    b = -wtx + _shift(wtx, 0, 1) - wty + _shift(wty, 1, 0)
    diag = lam + wx + _shift(wx, 0, 1) + wy + _shift(wy, 1, 0)
    u = _cg(A, b, np.zeros_like(zc), diag, iters=1500, tol=1e-6)
    # snap: whole numbers on the traced lines (the wall is the line)
    tgt = np.where(sk, np.round(u) - u, 0.0)
    wd = np.where(sk, 20.0, 0.0)
    lam2 = (res / snap_m) ** 2

    def A2(x):
        dxp = x - _shift(x, 0, -1)
        dyp = x - _shift(x, -1, 0)
        dxp[:, -1] = 0.0
        dyp[-1, :] = 0.0
        return (wd + lam2) * x + dxp - _shift(dxp, 0, 1) + dyp - _shift(dyp, 1, 0)

    corr = _cg(A2, wd * tgt, np.zeros_like(zc), wd + lam2 + 4.0, iters=600, tol=1e-5)
    u = u + np.clip(corr, -0.5, 0.5) * np.clip(pres * 2, 0, 1)
    # 4. staircase of u on the smoothed DEM
    zb = gauss(zc, base_m / 2 / res)
    gmag = np.hypot(*np.gradient(u, res)) * H            # implied slope of the bench system
    phi = np.clip(gmag / np.tan(np.radians(face_angle_deg)), 0.02, 0.9)
    n = np.floor(u + 0.5)
    rfr = u - n
    half = phi / 2
    st = np.where(rfr < -half, n - 0.5, np.where(rfr >= half, n + 0.5, n - 0.5 + (rfr + half) / phi))
    zt = zb + H * (st - u) - carve * H
    if cell_m > 0 and blocky:
        # square cells aligned with the local bench directions; one level per cell
        yy, xx = np.mgrid[0:Hh, 0:Ww].astype(float) * res
        th = np.where(coh4 > 0.25, th_d, 0.0)
        th = 0.25 * np.arctan2(gauss(np.sin(4 * th), 40 / res), gauss(np.cos(4 * th), 40 / res))
        a = xx * np.cos(th) + yy * np.sin(th)
        bq = -xx * np.sin(th) + yy * np.cos(th)
        ia = np.floor(a / cell_m).astype(np.int64)
        ib = np.floor(bq / cell_m).astype(np.int64)
        tq = np.round(th / (np.pi / 36)).astype(np.int64)        # 5 deg classes keep cells consistent
        key = (tq * 100003 + ia) * 100003 + ib
        uk, inv = np.unique(key.ravel(), return_inverse=True)
        cnt = np.bincount(inv)
        uc = (np.bincount(inv, weights=u.ravel()) / cnt)[inv].reshape(u.shape)
        zbc = (np.bincount(inv, weights=(zb - H * u).ravel()) / cnt)[inv].reshape(u.shape)
        blk = pres > 0.5
        stc = np.floor(uc) + 0.5
        zt = np.where(blk, zbc + H * stc - carve * H, zt)
        rfr = np.where(blk, 1.0, rfr)                    # all cells are floor pieces
        half = np.where(blk, 0.0, half)
        n = np.where(blk, np.floor(uc) + 1, n)
    if flat_floors:
        # floors flattened within their own level (local average, no jumps across walls)
        fl = (np.abs(rfr) >= half) & (pres > 0.5) & wc
        lvl = np.where(rfr >= half, n, n - 1)              # floor between walls lvl and lvl+1
        for v in np.unique(lvl[fl]):
            m = fl & (lvl == v)
            rows_ = np.where(m.any(1))[0]
            cols_ = np.where(m.any(0))[0]
            pd_ = int(flat_m / res)
            sub = (slice(max(0, rows_[0] - pd_), rows_[-1] + 1 + pd_), slice(max(0, cols_[0] - pd_), cols_[-1] + 1 + pd_))
            ms = m[sub].astype(float)
            vals = zt[sub]
            num = gauss(vals * ms, flat_m / 2 / res)
            den = gauss(ms, flat_m / 2 / res)
            zt[sub] = np.where(ms > 0, num / np.maximum(den, 1e-9), vals)
    # where the photo shows no benches keep the measured ground (smooth hand-over)
    zt = pres * zt + (1 - pres) * (zc - carve * H * pres)
    fe = gauss(wc.astype(float), feather_m / 2 / res) * wc
    fe = np.clip((fe - 0.5) * 2.0, 0, 1)
    zt = np.minimum(zt, zc + 0.5 * H)                     # never build up more than half a bench
    dz = fe * (zt - zc)
    out = np.zeros(z0.shape, "float32")
    out[sl] = dz
    step = np.zeros(zt.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1)):
        step |= np.abs(_shift(zt, dy, dx, np.nan) - zt) > 0.5 * H
    near_wall = box_blur(step.astype(float), max(1, int(round(4.0 / res)))) > 0
    on_wall = float((near_wall & sk & (pres > 0.5)).sum()) / max(1, int((sk & (pres > 0.5)).sum()))
    info = {"bench_height_m": round(H, 2), "line_length_km": round(float(sk.sum()) * res / 1000.0, 1),
            "bench_spacing_median_m": round(sp_med, 1), "lines_on_walls_pct": round(100 * on_wall, 1), "cell_m": cell_m,
            "benched_area_ha": round(float(((pres > 0.5) & wc).sum()) * res * res / 1e4, 1),
            "mean_dz_m": round(float(dz[wc].mean()), 2), "max_cut_m": round(float(-dz.min()), 1),
            "max_fill_m": round(float(dz.max()), 1)}
    return out, info
