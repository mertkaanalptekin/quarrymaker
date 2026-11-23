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


def _arclen(p):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]


def line_levels(lines, low_left, z0, transform, bench_h, reach_m=60.0, step_m=4.0, carve=0.5,
                anchor_w=0.05, wall_w=4.0, min_rays=3, side_m=(8.0, 18.0), fill_tol=1.0, fill_w=40.0,
                piece_m=80.0, cont_w=1.0, tv_eps=0.5, iters=8, pairs=None, log=None):
    """Heights of the floors on both sides of every bench line.

    Every line is cut into pieces of about piece_m; each piece has one level on
    its high side and one on its low side, so bench edges are level piece by
    piece. Unknowns are solved by weighted least squares:
      * high - low = bench_h                       (one bench per wall)
      * consecutive pieces of a line keep the same levels, with a total-variation
        (reweighted L1) penalty: levels stay exactly equal over long runs and
        change in a few clean steps where the ground demands it (benches step
        down along a face);
      * rays cast from each piece to both sides meet the next line: the floor
        between them is shared, so the two facing sides have the same level;
      * a pull of each side toward the measured terrain side_m out on that side,
        lowered by carve * bench_h / 2, reweighted (x fill_w) where a side would
        sit more than fill_tol above the ground: benches are cut into the ground.
        side_m may list several distances (the floor reaches ~18 m flat).
    Facing sides linked by at least `min_rays` rays, and consecutive pieces that
    differ by less than tv_eps, are then merged into exact common levels.
    low_left[i]: True when the lower floor is on the left of line i (drawing direction).
    Returns dict(pieces=[(edges, high, low)] per line, high, low (length means per
    line), groups, n_links, pairs); use level_at() for heights along a line."""
    from shapely.geometry import LineString
    from rasterio import features
    from rasterio.transform import from_origin

    H = float(bench_h)
    nl_ = len(lines)
    lens = [float(_arclen(a)[-1]) for a in lines]
    npc = [max(1, int(round(L / piece_m))) for L in lens]
    first = np.r_[0, np.cumsum(npc)].astype(int)            # first piece index of each line
    edges = [np.linspace(0.0, lens[i], npc[i] + 1) for i in range(nl_)]
    n_pc = int(first[-1])

    def piece(i, s_):
        return int(first[i] + np.clip(np.searchsorted(edges[i], s_, side="right") - 1, 0, npc[i] - 1))

    def node(i, k, left):
        # 2*k = high side, 2*k+1 = low side of global piece k (on line i)
        return 2 * k + (1 if (left == bool(low_left[i])) else 0)

    rows, rhs, wts, anc, cont = [], [], [], [], []
    O, D, I, K, S = [], [], [], [], []
    for i, a in enumerate(lines):
        p = _densify(a, step_m)
        if len(p) < 2:
            continue
        t = np.gradient(p, axis=0)
        t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12
        nl = np.c_[-t[:, 1], t[:, 0]]
        sp = _arclen(p)
        kp = np.array([piece(i, v) for v in sp])
        for k in range(first[i], first[i + 1]):
            m = kp == k
            if not m.any():
                m = np.zeros(len(p), bool); m[np.argmin(np.abs(sp - edges[i][k - first[i]:k - first[i] + 2].mean()))] = True
            Lk = max(step_m * m.sum(), 1.0)
            rows.append({2 * k: 1.0, 2 * k + 1: -1.0}); rhs.append(H); wts.append(wall_w * Lk)
            for left in (True, False):
                for dm in np.atleast_1d(side_m):
                    q = p[m] + nl[m] * (dm if left else -dm)
                    zs = float(np.mean(sample(z0, transform, *q.T)))
                    anc.append((len(rows), node(i, k, left), zs))
                    rows.append({node(i, k, left): 1.0}); rhs.append(zs - 0.5 * carve * H)
                    wts.append(anchor_w * Lk / len(np.atleast_1d(side_m)))
            if k > first[i]:
                for side in (0, 1):
                    cont.append((len(rows), 2 * (k - 1) + side, 2 * k + side))
                    rows.append({2 * (k - 1) + side: 1.0, 2 * k + side: -1.0}); rhs.append(0.0)
                    wts.append(cont_w * piece_m)
        for side in (1.0, -1.0):
            O.append(p); D.append(nl * side); I.append(np.full(len(p), i)); K.append(kp); S.append(np.full(len(p), side))

    if pairs is None:
        # which line does a ray meet first? marched over a 1 m map of line ids
        # (topology only; the geometry stays vector)
        allp = np.concatenate(lines)
        gx0, gy0 = allp.min(0) - reach_m - 2
        gx1, gy1 = allp.max(0) + reach_m + 2
        gres = 1.0
        gW, gH = int(np.ceil((gx1 - gx0) / gres)), int(np.ceil((gy1 - gy0) / gres))
        gtf = from_origin(gx0, gy1, gres, gres)
        lab = features.rasterize([(LineString(a), i + 1) for i, a in enumerate(lines) if len(a) >= 2],
                                 out_shape=(gH, gW), transform=gtf, all_touched=True, dtype="int32")
        dens = [_densify(a, 1.0) for a in lines]
        dens_s = [_arclen(d) for d in dens]
        O, D, I, K, S = (np.concatenate(v) for v in (O, D, I, K, S))
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
        pairs = {}
        for idx in np.nonzero(hitd > 0)[0]:
            i, j = int(I[idx]), int(hit[idx])
            o = O[idx]
            q = O[idx] + D[idx] * hitd[idx]
            bj = dens[j]
            kk = int(np.argmin(np.hypot(*(bj - q).T)))
            tj = bj[min(kk + 1, len(bj) - 1)] - bj[max(kk - 1, 0)]
            nj = np.array([-tj[1], tj[0]])
            from_left = float(np.dot(o - bj[kk], nj)) > 0
            a_ = node(i, int(K[idx]), S[idx] > 0)
            b_ = node(j, piece(j, dens_s[j][kk]), from_left)
            key = (min(a_, b_), max(a_, b_))
            pairs[key] = pairs.get(key, 0) + 1
    for (a_, b_), c in pairs.items():
        rows.append({a_: 1.0, b_: -1.0}); rhs.append(0.0); wts.append(float(c) * step_m)

    # dense normal equations: 2 unknowns per piece (a few thousand at most)
    A = np.zeros((len(rows), 2 * n_pc))
    for r_, d in enumerate(rows):
        for k_, v in d.items():
            A[r_, k_] = v
    wts = np.asarray(wts, float)
    rhs = np.asarray(rhs, float)
    ar = np.array([r for r, _, _ in anc]); an = np.array([k for _, k, _ in anc]); az = np.array([z for _, _, z in anc])
    cr = np.array([r for r, _, _ in cont], int); ca = np.array([x for _, x, _ in cont], int)
    cb = np.array([x for _, _, x in cont], int)
    base_a, base_c = wts[ar].copy(), wts[cr].copy() if len(cr) else np.zeros(0)
    sol = np.zeros(2 * n_pc)
    for it in range(iters):
        N = (A * wts[:, None]).T @ A
        N[np.diag_indices_from(N)] += 1e-9 * (np.trace(N) / len(N))
        sol = np.linalg.solve(N, (A * wts[:, None]).T @ rhs)
        wts[ar] = np.where(sol[an] - az > fill_tol, base_a * fill_w, base_a)
        if len(cr):
            wts[cr] = base_c * np.minimum(1.0, tv_eps / (np.abs(sol[ca] - sol[cb]) + 1e-6))
    # merge into exact common levels: facing sides, and consecutive pieces that agree.
    # Closest pairs first; a merge that would put the crest and the toe of the same
    # piece on one level (a wall of zero height) is refused.
    parent = np.arange(2 * n_pc)
    members = {k: {k // 2: 1 << (k % 2)} for k in range(2 * n_pc)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        ma, mb = members[ra], members[rb]
        if len(ma) > len(mb):
            ra, rb, ma, mb = rb, ra, mb, ma
        if any((mb.get(k, 0) | v) == 3 for k, v in ma.items()):
            return
        for k, v in ma.items():
            mb[k] = mb.get(k, 0) | v
        del members[ra]
        parent[ra] = rb
    cand = [(abs(sol[a_] - sol[b_]), int(a_), int(b_)) for (a_, b_), c in pairs.items()
            if c >= min_rays and abs(sol[a_] - sol[b_]) < 0.5 * H]
    cand += [(abs(sol[a_] - sol[b_]), int(a_), int(b_)) for a_, b_ in zip(ca, cb) if abs(sol[a_] - sol[b_]) < tv_eps]
    for _, a_, b_ in sorted(cand):
        union(a_, b_)
    roots = np.array([find(k) for k in range(2 * n_pc)])
    lev = sol.copy()
    for g in np.unique(roots):
        m = roots == g
        lev[m] = sol[m].mean()
    pcs = [(edges[i], lev[2 * first[i]:2 * first[i + 1]:2], lev[2 * first[i] + 1:2 * first[i + 1]:2]) for i in range(nl_)]
    hi_m = np.array([np.average(h, weights=np.diff(e)) if e[-1] > 0 else h[0] for e, h, _ in pcs])
    lo_m = np.array([np.average(lo, weights=np.diff(e)) if e[-1] > 0 else lo[0] for e, _, lo in pcs])
    walls = lev[0::2] - lev[1::2]
    steps = int(sum(1 for a_, b_ in zip(ca, cb) if a_ % 2 == 0 and abs(lev[a_] - lev[b_]) > 0.05))
    if log:
        f = lev[an] - az
        log.info("Bench levels from your lines: %d pieces of ~%.0f m, %d floor links, %d shared floors, "
                 "%d level steps along lines; wall heights median %.1f m (10-90%%: %.1f-%.1f m)",
                 n_pc, piece_m, len(pairs), len(np.unique(roots)), steps, float(np.median(walls)),
                 float(np.percentile(walls, 10)), float(np.percentile(walls, 90)))
        log.info("Bench floors against the measured ground %s m out: median %+.1f m, 95%% below %+.1f m, "
                 "max %+.1f m", "/".join(f"{v:.0f}" for v in np.atleast_1d(side_m)), float(np.median(f)), float(np.percentile(f, 95)), float(f.max()))
    return {"pieces": pcs, "high": hi_m, "low": lo_m, "walls": walls, "groups": roots,
            "n_links": len(pairs), "n_steps": steps, "pairs": pairs}


def level_at(levels, i, pts, which, ramp_m=1.0):
    """Level of side `which` ('high' or 'low') of line i at points `pts` that
    follow the line in order (e.g. a densified or offset copy of it). Changes
    between pieces are short steps of ramp_m."""
    e, hi, lo = levels["pieces"][i]
    v = hi if which == "high" else lo
    if len(v) == 1:
        return np.full(len(pts), float(v[0]))
    s_ = _arclen(np.asarray(pts, float))
    s_ = s_ * (e[-1] / s_[-1]) if s_[-1] > 0 else s_
    xs, ys = [0.0], [float(v[0])]
    for k in range(1, len(v)):
        b = e[k]
        xs += [b - ramp_m / 2, b + ramp_m / 2]
        ys += [float(v[k - 1]), float(v[k])]
    xs.append(float(e[-1]) + 1.0); ys.append(float(v[-1]))
    return np.interp(s_, xs, ys)


def tin_from_levels(lines, low_left, levels, z0, transform, gap_m=0.3, string_step_m=1.0, zone_m=40.0,
                    coarse_m=10.0, fill_m=4.0, clear_m=1.5, blend_from_m=18.0, log=None):
    """TIN of the whole site: crest and toe strings of every line at its constant
    levels (exact positions of the drawn lines), measured terrain grid points
    more than zone_m away from any line, and the site border.

    The breakline network is cleaned before the constrained triangulation:
    strings simplified to 10 cm, snapped to a 5 cm grid and noded; the few
    segments that still cross or overlap another one (where lines cross) are
    left out of the constraints. Heights come from the nearest original string
    point. Coordinates are shifted to the grid's corner for numerical safety."""
    import shapely
    import triangle
    from rasterio import features
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    H_, W_ = z0.shape
    res = transform.a
    x0, y1 = transform.c, transform.f
    x1, y0 = x0 + W_ * res, y1 - H_ * res
    off = np.array([x0, y0])
    strings, pz = [], []
    for i, a in enumerate(lines):
        p = _densify(a - off, string_step_m)
        if len(p) < 2:
            continue
        t = np.gradient(p, axis=0)
        t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12
        nl = np.c_[-t[:, 1], t[:, 0]]
        sl = 1.0 if low_left[i] else -1.0                 # direction of the low side
        for q, which in ((p - nl * sl * gap_m, "high"), (p + nl * sl * gap_m, "low")):
            strings.append(LineString(q).simplify(0.1))
            pz.append(np.c_[q, level_at(levels, i, q, which)])
    PZ = np.concatenate(pz)
    U = shapely.set_precision(unary_union(strings), 0.05)
    geoms = list(U.geoms) if hasattr(U, "geoms") else [U]
    sv, ss = [], []
    k = 0
    for g in geoms:
        c = np.asarray(g.coords)[:, :2]
        if len(c) < 2:
            continue
        sv.append(c)
        ss.append(np.c_[np.arange(k, k + len(c) - 1), np.arange(k + 1, k + len(c))])
        k += len(c)
    SV = np.concatenate(sv)
    S = np.concatenate(ss)
    key = np.round(SV * 20).astype(np.int64)
    _, first, inv = np.unique(key[:, 0] * 10_000_000 + key[:, 1], return_index=True, return_inverse=True)
    SV = SV[first]
    S = inv[S]
    S = S[S[:, 0] != S[:, 1]]
    S = np.unique(np.sort(S, axis=1), axis=0)
    segs = shapely.linestrings(np.stack([SV[S[:, 0]], SV[S[:, 1]]], 1))
    a_, b_ = STRtree(segs).query(segs, predicate="intersects")
    m = a_ < b_
    a_, b_ = a_[m], b_[m]
    share = (S[a_, 0] == S[b_, 0]) | (S[a_, 0] == S[b_, 1]) | (S[a_, 1] == S[b_, 0]) | (S[a_, 1] == S[b_, 1])
    bad = np.zeros(len(S), bool)
    bad[a_[~share]] = True
    bad[b_[~share]] = True
    ov = shapely.overlaps(segs[a_[share]], segs[b_[share]])
    bad[a_[share][ov]] = True
    bad[b_[share][ov]] = True
    S = S[~bad]
    # heights of the string vertices: nearest original string point
    ptree = STRtree(shapely.points(PZ[:, :2]))
    qi, ti = ptree.query_nearest(shapely.points(SV), return_distance=False)
    _, firstq = np.unique(qi, return_index=True)          # ties give several matches: keep one
    SZ = PZ[ti[firstq], 2]
    # terrain points away from the lines
    near = features.rasterize([(LineString(a).buffer(zone_m), 1) for a in lines if len(a) >= 2],
                              out_shape=(H_, W_), transform=transform).astype(bool)
    s = max(1, int(round(coarse_m / res)))
    rr, cc = np.mgrid[s // 2:H_:s, s // 2:W_:s]
    rr, cc = rr.ravel(), cc.ravel()
    keep = ~near[rr, cc]
    rr, cc = rr[keep], cc[keep]
    gxy = np.c_[(cc + 0.5) * res, (H_ - rr - 0.5) * res]
    gz = z0[rr, cc]
    # fill the benched zone with floor points: each takes the level of the nearest
    # string (crest or toe), i.e. the floor on its own side of the nearest wall
    sf = max(1, int(round(fill_m / res)))
    fr, fc = np.mgrid[sf // 2:H_:sf, sf // 2:W_:sf]
    fr, fc = fr.ravel(), fc.ravel()
    inz = near[fr, fc]
    fr, fc = fr[inz], fc[inz]
    fxy = np.c_[(fc + 0.5) * res, (H_ - fr - 0.5) * res]
    (fi, ft), fd = ptree.query_nearest(shapely.points(fxy), return_distance=True)
    _, f1 = np.unique(fi, return_index=True)
    fd, ft = fd[f1], ft[f1]
    ok = fd > clear_m
    fxy, fz, fd = fxy[ok], PZ[ft[ok], 2], fd[ok]
    # toward the edge of the benched zone, hand over smoothly to the measured ground
    wgt = np.clip((zone_m - fd) / max(zone_m - blend_from_m, 1e-6), 0, 1)
    fz = wgt * fz + (1 - wgt) * sample(z0, transform, *(fxy + off).T)
    gxy = np.r_[gxy, fxy]
    gz = np.r_[gz, fz]
    bx = np.r_[np.arange(0, x1 - x0, coarse_m), x1 - x0]
    by = np.r_[np.arange(0, y1 - y0, coarse_m), y1 - y0]
    border = np.unique(np.r_[np.c_[bx, np.zeros_like(bx)], np.c_[bx, np.full_like(bx, y1 - y0)],
                             np.c_[np.zeros_like(by), by], np.c_[np.full_like(by, x1 - x0), by]], axis=0)
    V = np.concatenate([SV, gxy, border])
    Z = np.concatenate([SZ, gz, sample(z0, transform, *(border + off).T)])
    # all points unique (strings first, so segment indices stay valid)
    key = np.round(V * 20).astype(np.int64)
    _, first = np.unique(key[:, 0] * 10_000_000 + key[:, 1], return_index=True)
    keep = np.zeros(len(V), bool)
    keep[first] = True
    keep[:len(SV)] = True
    V, Z = V[keep], Z[keep]
    out = triangle.triangulate({"vertices": V, "segments": S}, "pcQ")
    TV = out["vertices"]
    Zall = np.empty(len(TV))
    Zall[:len(V)] = Z
    if len(TV) > len(V):
        Zall[len(V):] = sample(z0, transform, *(TV[len(V):] + off).T)
    if log:
        log.info("Vector terrain (TIN) of the whole site: %d vertices, %d triangles (%d breakline segments, "
                 "%d left out where lines cross)", len(TV), len(out["triangles"]), len(S), int(bad.sum()))
    return {"verts": np.c_[TV + off, Zall], "tris": out["triangles"].astype(np.int64)}


def raster_from_tin(tin, transform, shape, zone, z_outside):
    """Interpreted raster from the vector model (for the .tif outputs and checks):
    inside `zone` every cell takes the height of the TIN triangle it falls in
    (exact on flat floors); elsewhere z_outside."""
    from rasterio import features
    from shapely.geometry import Polygon

    V, T = tin["verts"], tin["tris"]
    cen = V[T, :2].mean(1)
    inv = ~transform
    c, r = inv * (cen[:, 0], cen[:, 1])
    c = np.clip(np.asarray(c).astype(int), 0, shape[1] - 1)
    r = np.clip(np.asarray(r).astype(int), 0, shape[0] - 1)
    sel = zone[r, c]
    zmean = V[T[sel], 2].mean(1)
    # rasterize triangle ids (+1), then look up heights (exact on flat floors)
    ids = np.nonzero(sel)[0]
    lab = features.rasterize(((Polygon(V[T[k], :2]), i + 1) for i, k in enumerate(ids)), out_shape=shape,
                             transform=transform, dtype="int32")
    out = np.asarray(z_outside, dtype="float64").copy()
    m = (lab > 0) & zone
    out[m] = zmean[lab[m] - 1]
    return out
