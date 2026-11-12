"""Plan linework for Rhino / CAD: raster masks -> simplified polylines and
polygons, written as DXF (R12, readable by Rhino, AutoCAD, QGIS) and GeoJSON.
numpy + shapely + rasterio only."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ----------------------------------------------------------------------
# Morphology
# ----------------------------------------------------------------------

def _filt(a, k, fn):
    """Separable min/max filter with a (2k+1) window, edge padded."""
    a = np.pad(a, k, mode="edge")
    a = fn(sliding_window_view(a, 2 * k + 1, axis=0), axis=-1)
    return fn(sliding_window_view(a, 2 * k + 1, axis=1), axis=-1)


def min_filter(a, k):
    return _filt(a, k, np.min)


def max_filter(a, k):
    return _filt(a, k, np.max)


def white_tophat(a, k):
    """Bright details narrower than the window (2k+1 px): a - opening(a)."""
    return a - max_filter(min_filter(a, k), k)


def thin(mask, max_iter=50):
    """Zhang-Suen thinning to 1-pixel centrelines (vectorised)."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            c = P[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(x.astype(np.int8) for x in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int8) for i in range(8))
            if step == 0:
                m = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & m
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


# ----------------------------------------------------------------------
# Raster -> vector
# ----------------------------------------------------------------------

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_polylines(skel, transform, min_len_m=10.0, simplify_m=0.5):
    """Chain the pixels of a 1-px line mask into polylines (CRS coordinates),
    split at real junctions, simplified (Douglas-Peucker).

    Uses m-connectivity: a diagonal link is ignored when the two pixels are
    already joined through a straight neighbour, so diagonal and zigzag lines
    (pixel staircases) are not mistaken for junctions. Returns LineStrings."""
    from shapely.geometry import LineString

    skel = thin(skel)
    rows, cols = np.nonzero(skel)
    if rows.size == 0:
        return []
    pix = set(zip(rows.tolist(), cols.tolist()))

    def nbrs(p):
        r, c = p
        out = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)) if (r + dr, c + dc) in pix]
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = (r + dr, c + dc)
            if q in pix and (r + dr, c) not in pix and (r, c + dc) not in pix:
                out.append(q)
        return out

    adj = {p: nbrs(p) for p in pix}
    used = set()
    paths = []

    def walk(start, nxt):
        path = [start, nxt]
        used.add(frozenset((start, nxt)))
        prev, cur = start, nxt
        while len(adj[cur]) == 2:
            step = [n for n in adj[cur] if n != prev and frozenset((cur, n)) not in used]
            if not step:
                break
            prev, cur = cur, step[0]
            used.add(frozenset((prev, cur)))
            path.append(cur)
        return path

    for p in pix:
        if len(adj[p]) != 2:
            for n in adj[p]:
                if frozenset((p, n)) not in used:
                    paths.append(walk(p, n))
    for p in pix:   # closed loops
        for n in adj[p]:
            if frozenset((p, n)) not in used:
                paths.append(walk(p, n))
    out = []
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    for path in paths:
        if len(path) < 2:
            continue
        rr = np.array([q[0] for q in path], float) + 0.5
        cc = np.array([q[1] for q in path], float) + 0.5
        ls = LineString(np.c_[a * cc + b * rr + c, d * cc + e * rr + f]).simplify(simplify_m, preserve_topology=False)
        if ls.length >= min_len_m:
            out.append(ls)
    return out


def mask_polygons(mask, transform, min_area_m2=2000.0, simplify_m=2.0):
    """Outlines of a boolean mask as simplified shapely polygons."""
    from rasterio import features
    from shapely.geometry import shape

    polys = []
    for g, v in features.shapes(mask.astype("uint8"), mask=mask, connectivity=8, transform=transform):
        if v != 1:
            continue
        p = shape(g).simplify(simplify_m, preserve_topology=True)
        if p.area >= min_area_m2 and not p.is_empty:
            polys.append(p)
    return polys


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

LAYER_COLORS = {"QUARRY_OUTLINE": 7, "BENCH_LINES": 1, "ROADS": 8, "AGRICULTURE": 3,
                "BLOCK_YARDS": 4, "ORIGIN": 2}


def write_dxf(path, layers, origin=(0.0, 0.0), z_of=None):
    """layers: {name: [shapely LineString | Polygon, ...]}. Coordinates are written
    relative to `origin` (local metres, as in the Rhino OBJ). z_of(x, y) -> z (array
    in, array out) makes 3D polylines for layers listed in z_of; others are 2D."""
    from shapely.geometry import LineString, MultiPolygon, Polygon

    z_of = z_of or {}
    ox, oy = origin
    L = []

    def w(code, val):
        L.append(f"{code}\n{val}")
    w(0, "SECTION"); w(2, "HEADER")
    w(9, "$ACADVER"); w(1, "AC1009")
    w(9, "$INSUNITS"); w(70, 6)          # metres (honoured by newer readers)
    w(0, "ENDSEC")
    w(0, "SECTION"); w(2, "TABLES")
    w(0, "TABLE"); w(2, "LAYER"); w(70, len(layers))
    for name in layers:
        w(0, "LAYER"); w(2, name); w(70, 0); w(62, LAYER_COLORS.get(name, 7)); w(6, "CONTINUOUS")
    w(0, "ENDTAB"); w(0, "ENDSEC")
    w(0, "SECTION"); w(2, "ENTITIES")

    def poly(name, coords, closed, zfn):
        coords = np.asarray(coords, float)
        if closed and len(coords) > 2 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        z = zfn(coords[:, 0], coords[:, 1]) if zfn else None
        w(0, "POLYLINE"); w(8, name); w(66, 1)
        w(70, (1 if closed else 0) | (8 if z is not None else 0))
        w(10, 0.0); w(20, 0.0); w(30, 0.0)
        for i, (x, y) in enumerate(coords):
            w(0, "VERTEX"); w(8, name)
            w(10, f"{x - ox:.2f}"); w(20, f"{y - oy:.2f}")      # centimetre precision
            w(30, f"{(z[i] if z is not None else 0.0):.2f}")
            if z is not None:
                w(70, 32)
        w(0, "SEQEND"); w(8, name)

    counts = {}
    for name, geoms in layers.items():
        zfn = z_of.get(name)
        n = 0
        for g in geoms:
            parts = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
            for part in parts:
                if isinstance(part, Polygon):
                    poly(name, part.exterior.coords, True, zfn)
                    for hole in part.interiors:
                        poly(name, hole.coords, True, zfn)
                    n += 1
                elif isinstance(part, LineString):
                    poly(name, part.coords, False, zfn)
                    n += 1
        counts[name] = n
    w(0, "ENDSEC"); w(0, "EOF")
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    return counts


def write_geojson(path, geoms, crs_name, props=None):
    import json

    from shapely.geometry import mapping
    feats = [{"type": "Feature", "properties": (props or {}), "geometry": mapping(g)} for g in geoms]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs_name}},
                   "features": feats}, fh)
