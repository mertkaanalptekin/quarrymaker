"""Vector terrain model built on the user's bench lines (no pixel steps).

Quarry surveyors model benches as a TIN with breaklines: every bench edge is a
pair of strings, the crest on the upper side and the toe on the lower side,
and triangles join them. Here:
  * each user polyline gives two strings, offset `gap_m` to either side, with
    the floor heights found on each side of the line (the interpreted raster,
    sampled `side_m` away, cleaned along the line with a running median);
  * the ground between the lines is sampled on a regular grid (finer inside
    the excavation, coarser outside), skipping a band around the lines so
    no raster step survives near a wall;
  * a constrained Delaunay triangulation (Shewchuk's Triangle) keeps the
    strings as edges, so walls follow the drawn lines exactly and flat floors
    stay exactly flat.
`floor_polygons` then groups the flat triangles into bench floor outlines.
"""
from __future__ import annotations

import numpy as np


def sample(z, transform, x, y):
    """Bilinear sample of grid z (cell centres) at CRS points x, y."""
    inv = ~transform
    c, r = inv * (np.asarray(x, float), np.asarray(y, float))
    c = np.asarray(c) - 0.5
    r = np.asarray(r) - 0.5
    H, W = z.shape
    c0 = np.clip(np.floor(c).astype(int), 0, W - 2)
    r0 = np.clip(np.floor(r).astype(int), 0, H - 2)
    fc = np.clip(c - c0, 0, 1)
    fr = np.clip(r - r0, 0, 1)
    return (z[r0, c0] * (1 - fc) * (1 - fr) + z[r0, c0 + 1] * fc * (1 - fr)
            + z[r0 + 1, c0] * (1 - fc) * fr + z[r0 + 1, c0 + 1] * fc * fr)


def _densify(a, step):
    seg = np.hypot(*np.diff(a, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] <= 0:
        return a[:1]
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.c_[np.interp(t, s, a[:, 0]), np.interp(t, s, a[:, 1])]


def _running_median(v, k):
    if len(v) < 3 or k < 3:
        return v
    h = k // 2
    p = np.pad(v, h, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, 2 * h + 1), axis=-1)


def tin_from_lines(zt, transform, lines, excav, fine_m=5.0, coarse_m=12.0, gap_m=0.3, side_m=3.0,
                   clear_m=2.5, string_step_m=1.0, median_k=9, wall_min_m=1.0, log=None):
    """zt: interpreted surface (whole site grid); excav: bool grid of the benched area.
    lines: list of (N, 2) CRS arrays. Returns dict(verts (N, 3) CRS, tris (M, 3),
    on_string (N,) bool)."""
    import triangle
    from rasterio import features
    from shapely.geometry import LineString

    H, W = zt.shape
    res = transform.a
    x0, y1 = transform.c, transform.f
    x1, y0 = x0 + W * res, y1 - H * res
    pts, zs, segs, on_str = [], [], [], []
    nv = 0

    def add_string(xy, z):
        nonlocal nv
        n = len(xy)
        pts.append(xy)
        zs.append(z)
        on_str.append(np.ones(n, bool))
        segs.append(np.c_[np.arange(nv, nv + n - 1), np.arange(nv + 1, nv + n)])
        nv += n

    n_walls = 0
    for a in lines:
        a = a[(a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) & (a[:, 1] < y1)]
        if len(a) < 2:
            continue
        p = _densify(a, string_step_m)
        if len(p) < 2:
            continue
        t = np.gradient(p, axis=0)
        t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12
        nrm = np.c_[-t[:, 1], t[:, 0]]                      # left normal
        zl = _running_median(sample(zt, transform, *(p + nrm * side_m).T), median_k)
        zr = _running_median(sample(zt, transform, *(p - nrm * side_m).T), median_k)
        if np.median(np.abs(zl - zr)) < wall_min_m:
            # no step was built along this line: keep it as a simple breakline
            add_string(p, sample(zt, transform, *p.T))
            continue
        add_string(p + nrm * gap_m, zl)
        add_string(p - nrm * gap_m, zr)
        n_walls += 1

    # grid points: fine in the excavation, coarse elsewhere, none close to the lines
    near = np.zeros((H, W), bool)
    if lines:
        near = features.rasterize([(LineString(a), 1) for a in lines if len(a) >= 2], out_shape=(H, W),
                                  transform=transform, all_touched=True).astype(bool)
        k = max(1, int(np.ceil(clear_m / res)))
        pad = np.pad(near, k)
        acc = np.zeros_like(near)
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                acc |= pad[k + dr:k + dr + H, k + dc:k + dc + W]
        near = acc
    for step, region in ((fine_m, excav), (coarse_m, ~excav)):
        s = max(1, int(round(step / res)))
        rr, cc = np.mgrid[s // 2:H:s, s // 2:W:s]
        rr, cc = rr.ravel(), cc.ravel()
        keep = region[rr, cc] & ~near[rr, cc]
        rr, cc = rr[keep], cc[keep]
        xy = np.c_[x0 + (cc + 0.5) * res, y1 - (rr + 0.5) * res]
        pts.append(xy)
        zs.append(zt[rr, cc])
        on_str.append(np.zeros(len(xy), bool))
        nv += len(xy)
    # site border
    bx = np.r_[np.arange(x0, x1, coarse_m), x1]
    by = np.r_[np.arange(y0, y1, coarse_m), y1]
    ring = np.r_[np.c_[bx, np.full_like(bx, y0)], np.c_[np.full_like(by, x1), by][1:],
                 np.c_[bx[::-1], np.full_like(bx, y1)][1:], np.c_[np.full_like(by, x0), by[::-1]][1:-1]]
    add_string(np.r_[ring, ring[:1]], sample(zt, transform, *np.r_[ring, ring[:1]].T))

    V = np.concatenate(pts)
    Z = np.concatenate(zs)
    S = np.concatenate(segs)
    OS = np.concatenate(on_str)
    # clean the breaklines: node them at crossings and merge overlaps (heights carried as z)
    from shapely.geometry import MultiLineString
    from shapely.ops import unary_union
    seg3 = [((V[a, 0], V[a, 1], Z[a]), (V[b, 0], V[b, 1], Z[b])) for a, b in S
            if np.hypot(*(V[a] - V[b])) > 0.02]
    noded = unary_union(MultiLineString(seg3))
    geoms = list(noded.geoms) if hasattr(noded, "geoms") else [noded]
    sv, sz, ss = [], [], []
    k = 0
    for g in geoms:
        c = np.asarray(g.coords)
        if len(c) < 2:
            continue
        sv.append(c[:, :2])
        sz.append(c[:, 2] if c.shape[1] > 2 else sample(zt, transform, *c[:, :2].T))
        ss.append(np.c_[np.arange(k, k + len(c) - 1), np.arange(k + 1, k + len(c))])
        k += len(c)
    grid = ~OS
    V = np.concatenate(sv + [V[grid]])
    Z = np.concatenate(sz + [Z[grid]])
    OS = np.r_[np.ones(k, bool), np.zeros(int(grid.sum()), bool)]
    S = np.concatenate(ss)
    # Triangle needs unique vertices (1 cm)
    key = np.round(V * 100).astype(np.int64)
    _, first, inv = np.unique(key[:, 0] * 10_000_000_000 + key[:, 1], return_index=True, return_inverse=True)
    V, Z, OS = V[first], Z[first], OS[first]
    V = np.round(V, 2)
    S = inv[S]
    S = S[S[:, 0] != S[:, 1]]
    S = np.unique(np.sort(S, axis=1), axis=0)
    out = triangle.triangulate({"vertices": V, "segments": S}, "pQ")
    TV = out["vertices"]
    nT = len(TV)
    Zall = np.empty(nT)
    Zall[:len(V)] = Z
    if nT > len(V):            # vertices Triangle inserted at crossing strings: take the raster height
        Zall[len(V):] = sample(zt, transform, *TV[len(V):].T)
    OSall = np.zeros(nT, bool)
    OSall[:len(V)] = OS
    if log:
        log.info("Vector terrain (TIN): %d vertices, %d triangles; %d of your lines are walls (crest + toe strings)",
                 nT, len(out["triangles"]), n_walls)
    return {"verts": np.c_[TV, Zall], "tris": out["triangles"].astype(np.int64), "on_string": OSall,
            "walls": n_walls}


def floor_polygons(verts, tris, inside_fn=None, flat_tol=0.05, min_area_m2=50.0, simplify_m=0.25):
    """Group flat triangles (all three corners at the same height) that share an
    edge and a height into bench floors. Returns list of (shapely Polygon, z)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    z = verts[tris, 2]
    flat = (z.max(1) - z.min(1)) < flat_tol
    idx = np.nonzero(flat)[0]
    if inside_fn is not None and idx.size:
        cen = verts[tris[idx], :2].mean(1)
        idx = idx[inside_fn(cen[:, 0], cen[:, 1])]
    if idx.size == 0:
        return []
    zq = np.round(z[idx].mean(1) / flat_tol).astype(np.int64)
    # union-find over shared edges between flat triangles of the same height
    parent = np.arange(idx.size)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = {}
    for k, t in enumerate(tris[idx]):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = (min(a, b), max(a, b))
            j = edges.get(e)
            if j is None:
                edges[e] = k
            elif zq[j] == zq[k]:
                ra, rb = find(j), find(k)
                if ra != rb:
                    parent[ra] = rb
    roots = np.array([find(k) for k in range(idx.size)])
    out = []
    for g in np.unique(roots):
        mem = idx[roots == g]
        if len(mem) < 2:
            continue
        poly = unary_union([Polygon(verts[t, :2]) for t in tris[mem]])
        polys = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
        zf = float(np.median(verts[tris[mem], 2]))
        for p in polys:
            if p.area >= min_area_m2:
                out.append((p.simplify(simplify_m, preserve_topology=True), zf))
    return out


def line_levels(lines, low_left, z0, transform, bench_h, reach_m=60.0, step_m=4.0, carve=0.5,
                anchor_w=0.02, wall_w=4.0, min_rays=3, log=None):
    """Heights of the floors on both sides of every bench line, constant along
    each line (bench edges are level).

    Unknowns: level of the high side and of the low side of every line.
    Constraints (least squares):
      * high - low = bench_h                       (one bench per wall)
      * rays cast from each line to both sides meet the next line: the floor
        between them is shared, so the two facing sides have the same level
      * a weak pull of the line's mid-height toward the measured terrain along
        the line, lowered by carve * bench_h (benches are cut into the ground)
    Facing sides linked by at least `min_rays` rays are then merged into one
    exact level, so each floor between two lines is perfectly flat.
    low_left[i]: True when the lower floor is on the left of line i (drawing direction).
    Returns dict(high, low, groups, n_links)."""
    from shapely.geometry import LineString

    n = len(lines)
    H = float(bench_h)
    rows, rhs, wts = [], [], []
    pair = {}

    def node(i, left):
        # 2*i = high side, 2*i+1 = low side
        return 2 * i + (1 if (left == bool(low_left[i])) else 0)

    # which line does a ray meet first? marched over a 1 m map of line ids (topology only;
    # the geometry stays vector)
    from rasterio import features
    from rasterio.transform import from_origin
    allp = np.concatenate(lines)
    gx0, gy0 = allp.min(0) - reach_m - 2
    gx1, gy1 = allp.max(0) + reach_m + 2
    gres = 1.0
    gW, gH = int(np.ceil((gx1 - gx0) / gres)), int(np.ceil((gy1 - gy0) / gres))
    gtf = from_origin(gx0, gy1, gres, gres)
    lab = features.rasterize([(LineString(a), i + 1) for i, a in enumerate(lines) if len(a) >= 2],
                             out_shape=(gH, gW), transform=gtf, all_touched=True, dtype="int32")
    dens = [_densify(a, 1.0) for a in lines]
    O, D, I, S = [], [], [], []
    for i, a in enumerate(lines):
        p = _densify(a, step_m)
        if len(p) < 2:
            continue
        t = np.gradient(p, axis=0)
        t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12
        nl = np.c_[-t[:, 1], t[:, 0]]
        zl = float(np.mean(sample(z0, transform, *p.T)))
        L = float(np.hypot(*np.diff(a, axis=0).T).sum())
        rows.append({2 * i: 1.0, 2 * i + 1: -1.0}); rhs.append(H); wts.append(wall_w * L)
        rows.append({2 * i: 0.5, 2 * i + 1: 0.5}); rhs.append(zl - carve * H); wts.append(anchor_w * L)
        for side in (1.0, -1.0):
            O.append(p); D.append(nl * side); I.append(np.full(len(p), i)); S.append(np.full(len(p), side))
    O, D, I, S = np.concatenate(O), np.concatenate(D), np.concatenate(I), np.concatenate(S)
    hit = np.zeros(len(O), "int64")
    hitd = np.zeros(len(O))
    alive = np.ones(len(O), bool)
    for k in np.arange(1.5, reach_m, 0.5):
        q = O + D * k
        c = ((q[:, 0] - gx0) / gres).astype(int)
        r = ((gy1 - q[:, 1]) / gres).astype(int)
        ok = alive & (c >= 0) & (c < gW) & (r >= 0) & (r < gH)
        v = np.zeros(len(O), "int64")
        v[ok] = lab[r[ok], c[ok]]
        new = ok & (v > 0) & (v - 1 != I)
        hit[new] = v[new] - 1
        hitd[new] = k
        alive &= ~new
    sel = np.nonzero(hitd > 0)[0]
    for idx in sel:
        i, j = int(I[idx]), int(hit[idx])
        o = O[idx]
        q = O[idx] + D[idx] * hitd[idx]
        bj = dens[j]
        kk = int(np.argmin(np.hypot(*(bj - q).T)))
        tj = bj[min(kk + 1, len(bj) - 1)] - bj[max(kk - 1, 0)]
        nj = np.array([-tj[1], tj[0]])
        from_left = float(np.dot(o - bj[kk], nj)) > 0
        a_ = node(i, S[idx] > 0)
        b_ = node(j, from_left)
        key = (min(a_, b_), max(a_, b_))
        pair[key] = pair.get(key, 0) + 1
    for (a_, b_), c in pair.items():
        rows.append({a_: 1.0, b_: -1.0}); rhs.append(0.0); wts.append(float(c) * step_m)
    # weighted least squares (dense: 2 unknowns per line)
    A = np.zeros((len(rows), 2 * n))
    for r, d in enumerate(rows):
        for k_, v in d.items():
            A[r, k_] = v
    w = np.sqrt(np.asarray(wts))
    sol, *_ = np.linalg.lstsq(A * w[:, None], np.asarray(rhs) * w, rcond=None)
    # merge sides that face each other into exact common levels
    parent = np.arange(2 * n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (a_, b_), c in pair.items():
        if c >= min_rays and abs(sol[a_] - sol[b_]) < 0.5 * H:
            ra, rb = find(a_), find(b_)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(k) for k in range(2 * n)])
    lev = sol.copy()
    for g in np.unique(roots):
        m = roots == g
        lev[m] = sol[m].mean()
    if log:
        resid = np.array([lev[2 * i] - lev[2 * i + 1] for i in range(n)])
        log.info("Bench levels from your lines: %d floor links, %d shared floors; wall heights median %.1f m "
                 "(10-90%%: %.1f-%.1f m)", len(pair), len(np.unique(roots)), float(np.median(resid)),
                 float(np.percentile(resid, 10)), float(np.percentile(resid, 90)))
    return {"high": lev[0::2], "low": lev[1::2], "groups": roots, "n_links": len(pair)}


def tin_from_levels(lines, low_left, levels, z0, transform, gap_m=0.3, string_step_m=1.0, zone_m=40.0,
                    coarse_m=10.0, log=None):
    """TIN of the whole site: crest and toe strings of every line at its constant
    levels (exact positions of the drawn lines), measured terrain grid points
    more than zone_m away from any line, and the site border."""
    import triangle
    from rasterio import features
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import unary_union

    H_, W_ = z0.shape
    res = transform.a
    x0, y1 = transform.c, transform.f
    x1, y0 = x0 + W_ * res, y1 - H_ * res
    seg3 = []
    for i, a in enumerate(lines):
        p = _densify(a, string_step_m)
        if len(p) < 2:
            continue
        t = np.gradient(p, axis=0)
        t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12
        nl = np.c_[-t[:, 1], t[:, 0]]
        sl = 1.0 if low_left[i] else -1.0                 # direction of the low side
        hi = p - nl * sl * gap_m
        lo = p + nl * sl * gap_m
        for q, z in ((hi, levels["high"][i]), (lo, levels["low"][i])):
            c = np.c_[q, np.full(len(q), z)]
            seg3 += [tuple(map(tuple, c[k:k + 2])) for k in range(len(c) - 1)]
    noded = unary_union(MultiLineString(seg3))
    geoms = list(noded.geoms) if hasattr(noded, "geoms") else [noded]
    sv, sz, ss = [], [], []
    k = 0
    for g in geoms:
        c = np.asarray(g.coords)
        if len(c) < 2:
            continue
        sv.append(c[:, :2])
        sz.append(c[:, 2])
        ss.append(np.c_[np.arange(k, k + len(c) - 1), np.arange(k + 1, k + len(c))])
        k += len(c)
    # terrain points away from the lines
    near = features.rasterize([(LineString(a).buffer(zone_m), 1) for a in lines if len(a) >= 2],
                              out_shape=(H_, W_), transform=transform).astype(bool)
    s = max(1, int(round(coarse_m / res)))
    rr, cc = np.mgrid[s // 2:H_:s, s // 2:W_:s]
    rr, cc = rr.ravel(), cc.ravel()
    keep = ~near[rr, cc]
    rr, cc = rr[keep], cc[keep]
    gxy = np.c_[x0 + (cc + 0.5) * res, y1 - (rr + 0.5) * res]
    bx = np.r_[np.arange(x0, x1, coarse_m), x1]
    by = np.r_[np.arange(y0, y1, coarse_m), y1]
    border = np.unique(np.r_[np.c_[bx, np.full_like(bx, y0)], np.c_[bx, np.full_like(bx, y1)],
                             np.c_[np.full_like(by, x0), by], np.c_[np.full_like(by, x1), by]], axis=0)
    V = np.concatenate(sv + [gxy, border])
    Z = np.concatenate(sz + [z0[rr, cc], sample(z0, transform, *border.T)])
    S = np.concatenate(ss) if ss else np.zeros((0, 2), int)
    key = np.round(V * 100).astype(np.int64)
    _, first, inv = np.unique(key[:, 0] * 10_000_000_000 + key[:, 1], return_index=True, return_inverse=True)
    V, Z = np.round(V[first], 2), Z[first]
    S = inv[S]
    S = S[S[:, 0] != S[:, 1]]
    S = np.unique(np.sort(S, axis=1), axis=0)
    out = triangle.triangulate({"vertices": V, "segments": S}, "pQ")
    TV = out["vertices"]
    Zall = np.empty(len(TV))
    Zall[:len(V)] = Z
    if len(TV) > len(V):
        Zall[len(V):] = sample(z0, transform, *TV[len(V):].T)
    if log:
        log.info("Vector terrain (TIN) of the whole site: %d vertices, %d triangles", len(TV), len(out["triangles"]))
    return {"verts": np.c_[TV, Zall], "tris": out["triangles"].astype(np.int64)}
