"""Rhino file of the interpreted quarry (stage 07), written with rhino3dm.

Layers (metres, local model coordinates = real - origin, like the OBJ):
  Quarry Benches (your drawing)    the bench lines as drawn (flat, z = 0)
  Quarry Benches 3D                the same lines lifted to the top of their wall
  Quarry model (interpreted mesh)  the block mesh (flat floors, vertical walls) of
                                   the excavated area only
  Quarry Brep (simplified)         HIDDEN: each bench floor as a planar face, and
                                   its walls as vertical extruded faces; floors are
                                   simplified to ~1.5 m (no 2 m grid steps)
  Info                             origin and CRS
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "blender_helpers"))


def _layer(m, r, name, color, visible=True, parent=None):
    L = r.Layer()
    L.Name = name
    L.Color = color
    L.Visible = visible
    if parent is not None:
        L.ParentLayerId = parent
    return m.Layers.Add(L)


def _attrs(r, layer, name=None):
    a = r.ObjectAttributes()
    a.LayerIndex = layer
    if name:
        a.Name = name
    return a


def write_quarry_3dm(out_path, z, transform, area, origin, crs, bench_h, user_lines=None, wall_m=2.5,
                     simplify_m=1.5, min_floor_m2=100.0, log=None):
    """z: interpreted surface (2-D, metres) on `transform` (north-up grid).
    area: bool mask of the excavated / benched cells to export.
    user_lines: list of (N, 2) arrays in CRS metres (optional)."""
    import rhino3dm as r
    from rasterio import features
    from shapely.geometry import LineString, Polygon, shape

    import blockmesh

    ox, oy = origin
    res = transform.a
    rows, cols = np.nonzero(area)
    if rows.size == 0:
        raise RuntimeError("no excavated area to export")
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    zc = np.asarray(z[r0:r1, c0:c1], dtype="float64")
    ac = area[r0:r1, c0:c1]
    x0 = transform.c + c0 * res - ox
    y0 = transform.f - r0 * res - oy

    m = r.File3dm()
    m.Settings.ModelUnitSystem = r.UnitSystem.Meters
    m.Settings.ModelAbsoluteTolerance = 0.001
    L_draw = _layer(m, r, "Quarry Benches (your drawing)", (255, 0, 0, 255))
    L_3d = _layer(m, r, "Quarry Benches 3D", (255, 140, 0, 255))
    L_mesh = _layer(m, r, "Quarry model (interpreted mesh)", (190, 180, 160, 255))
    L_brep = _layer(m, r, "Quarry Brep (simplified)", (120, 160, 200, 255), visible=False)
    L_info = _layer(m, r, "Info", (30, 90, 200, 255))
    stats = {}

    # --- lines ---------------------------------------------------------------
    if user_lines:
        H_, W_ = z.shape
        inv = ~transform
        for i, a in enumerate(user_lines):
            loc = a - np.array([ox, oy])
            m.Objects.AddCurve(r.PolylineCurve([r.Point3d(float(x), float(y), 0.0) for x, y in loc]),
                               _attrs(r, L_draw, f"bench_line_{i}"))
            cc, rr = inv * (a[:, 0], a[:, 1])
            ci = np.clip(np.round(np.asarray(cc) - 0.5).astype(int), 0, W_ - 1)
            ri = np.clip(np.round(np.asarray(rr) - 0.5).astype(int), 0, H_ - 1)
            top = np.full(len(a), -np.inf)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    top = np.maximum(top, z[np.clip(ri + dr, 0, H_ - 1), np.clip(ci + dc, 0, W_ - 1)])
            m.Objects.AddCurve(r.PolylineCurve([r.Point3d(float(x), float(y), float(t))
                                                for (x, y), t in zip(loc, top)]),
                               _attrs(r, L_3d, f"bench_line_{i}_3d"))
        stats["lines"] = len(user_lines)

    # --- mesh of the excavated area (block mesh, cells outside `area` dropped) ---
    bm = blockmesh.build(zc, x0, y0, res, wall_m)
    keep = ac.ravel()[bm["cell"]]
    st, tt, lp = bm["starts"][keep], bm["totals"][keep], bm["loops"]
    used = np.zeros(len(bm["co"]), bool)
    for k in range(4):
        sel = tt > k
        used[lp[st[sel] + k]] = True
    remap = -np.ones(len(bm["co"]), "int64")
    remap[used] = np.arange(int(used.sum()))
    me = r.Mesh()
    for x, y, zz in bm["co"][used]:
        me.Vertices.Add(float(x), float(y), float(zz))
    for s_, t_ in zip(st, tt):
        v = remap[lp[s_:s_ + t_]]
        if t_ == 4:
            me.Faces.AddFace(int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        else:
            me.Faces.AddFace(int(v[0]), int(v[1]), int(v[2]))
    me.Normals.ComputeNormals()
    me.Compact()
    m.Objects.AddMesh(me, _attrs(r, L_mesh, "quarry_interpreted_mesh"))
    stats["mesh_vertices"], stats["mesh_faces"] = int(used.sum()), int(keep.sum())

    # --- simplified Breps: flat floor pieces + their walls -------------------------
    Hh, Ww = zc.shape
    zp = np.pad(zc, 1, mode="edge")
    flat = np.ones(zc.shape, bool)
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = zp[1 + dr:Hh + 1 + dr, 1 + dc:Ww + 1 + dc]
        flat &= (np.abs(nb - zc) < 0.05) | (np.abs(nb - zc) > wall_m)
    flat &= ac
    zq = np.round(zc * 20).astype("int32")                   # 5 cm classes
    from rasterio.transform import Affine
    tfl = Affine(res, 0, x0, 0, -res, y0)
    n_floor = n_wall = 0
    for geom, val in features.shapes(zq, mask=flat, connectivity=4, transform=tfl):
        poly = shape(geom)
        if poly.area < min_floor_m2:
            continue
        poly = poly.simplify(simplify_m, preserve_topology=True)
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        zf = float(val) / 20.0
        ring = np.asarray(poly.exterior.coords)
        crv = r.PolylineCurve([r.Point3d(float(x), float(y), zf) for x, y in ring])
        b = r.Brep.CreateTrimmedPlane(r.Plane(r.Point3d(0, 0, zf), r.Vector3d(0, 0, 1)), crv)
        if b is None:
            continue
        m.Objects.AddBrep(b, _attrs(r, L_brep, f"floor_{zf:.2f}m"))
        n_floor += 1
        # walls: runs of the outline where the ground just outside is lower by more than wall_m
        ls = LineString(ring)
        step = max(res, 1.0)
        dists = np.arange(0, ls.length, step)
        pts = np.array([ls.interpolate(d).coords[0] for d in dists] + [ring[-1]])
        if len(pts) < 3:
            continue
        tang = np.gradient(pts, axis=0)
        nrm = np.c_[tang[:, 1], -tang[:, 0]] / (np.hypot(tang[:, 0], tang[:, 1])[:, None] + 1e-9)
        sign = 1.0 if poly.exterior.is_ccw else -1.0              # outward normal = right of a ccw ring
        out = pts + sign * nrm * 1.5 * res
        oc = np.clip(((out[:, 0] - x0) / res).astype(int), 0, Ww - 1)
        orr = np.clip(((y0 - out[:, 1]) / res).astype(int), 0, Hh - 1)
        drop = zf - zc[orr, oc]
        isw = drop > wall_m
        k = 0
        while k < len(pts):
            if not isw[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(pts) and isw[j + 1]:
                j += 1
            if j > k:
                hgt = float(np.median(drop[k:j + 1]))
                run = pts[k:j + 1]
                for a_, b_ in zip(run[:-1], run[1:]):
                    quad = [r.Point3d(a_[0], a_[1], zf), r.Point3d(b_[0], b_[1], zf),
                            r.Point3d(b_[0], b_[1], zf - hgt), r.Point3d(a_[0], a_[1], zf - hgt),
                            r.Point3d(a_[0], a_[1], zf)]
                    v = r.Vector3d(b_[0] - a_[0], b_[1] - a_[1], 0)
                    pl = r.Plane(r.Point3d(a_[0], a_[1], zf), v, r.Vector3d(0, 0, 1))
                    wb = r.Brep.CreateTrimmedPlane(pl, r.PolylineCurve(quad))
                    if wb is not None:
                        m.Objects.AddBrep(wb, _attrs(r, L_brep, f"wall_{hgt:.1f}m"))
                        n_wall += 1
            k = j + 1
    stats["brep_floors"], stats["brep_wall_faces"] = n_floor, n_wall

    m.Objects.AddTextDot(f"Quarry Maker origin\nE {ox:.0f}\nN {oy:.0f}\n{crs}\nbench height {bench_h:g} m",
                         r.Point3d(0, 0, float(np.nanmax(zc))), _attrs(r, L_info, "origin"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not m.Write(str(out_path), 7):
        raise RuntimeError(f"could not write {out_path}")
    if log:
        log.info("Rhino file %s: %d lines, mesh %d faces, Brep %d floors + %d wall faces (hidden layer)",
                 out_path.name, stats.get("lines", 0), stats["mesh_faces"], n_floor, n_wall)
    return stats


def write_vector_3dm(out_path, tin, floors, lines, levels, low_on_left, origin, crs, bench_h,
                     base_margin_m=10.0, gap_m=0.3, log=None):
    """Rhino file of the vector model built on the user's bench lines.
      Quarry Benches (your drawing)   the lines as drawn (z = 0)
      Quarry Benches 3D               crest (top) and toe (bottom) of every line at its level
      Quarry model (whole site)       TIN mesh of the whole site: flat floors, walls on the lines
      Quarry Brep solids (hidden)     one closed box-like solid per bench floor: the floor
                                      outline (sides on your lines) extruded down to a common base
    tin: dict(verts (N,3) CRS, tris); floors: list of (shapely Polygon in CRS, z)."""
    import rhino3dm as r

    import vectormesh as VM

    ox, oy = origin
    m = r.File3dm()
    m.Settings.ModelUnitSystem = r.UnitSystem.Meters
    m.Settings.ModelAbsoluteTolerance = 0.001
    L_draw = _layer(m, r, "Quarry Benches (your drawing)", (255, 0, 0, 255))
    L_3d = _layer(m, r, "Quarry Benches 3D", (255, 140, 0, 255))
    L_mesh = _layer(m, r, "Quarry model (whole site)", (190, 180, 160, 255))
    L_sol = _layer(m, r, "Quarry Brep solids (simplified)", (120, 160, 200, 255), visible=False)
    L_info = _layer(m, r, "Info", (30, 90, 200, 255))

    for i, a in enumerate(lines):
        loc = a - np.array([ox, oy])
        m.Objects.AddCurve(r.PolylineCurve([r.Point3d(float(x), float(y), 0.0) for x, y in loc]),
                           _attrs(r, L_draw, f"bench_line_{i}"))
        t = np.gradient(loc, axis=0)
        t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12
        nl = np.c_[-t[:, 1], t[:, 0]]
        sl = 1.0 if low_on_left[i] else -1.0
        for name, q, which in (("crest", loc - nl * sl * gap_m, "high"), ("toe", loc + nl * sl * gap_m, "low")):
            qd = VM._densify(q, 1.0)
            zz = VM.level_at(levels, i, qd, which)
            keep = np.r_[True, np.abs(np.diff(zz)) > 1e-3] | np.r_[np.abs(np.diff(zz)) > 1e-3, True]
            keep[[0, -1]] = True
            # original vertices plus the points where the level steps
            pts = np.r_[np.c_[q, VM.level_at(levels, i, q, which)], np.c_[qd[keep], zz[keep]]]
            order = np.argsort(np.r_[VM._arclen(q) / max(VM._arclen(q)[-1], 1e-9),
                                     VM._arclen(qd)[keep] / max(VM._arclen(qd)[-1], 1e-9)], kind="stable")
            pts = pts[order]
            m.Objects.AddCurve(r.PolylineCurve([r.Point3d(float(x), float(y), float(z)) for x, y, z in pts]),
                               _attrs(r, L_3d, f"bench_line_{i}_{name}"))

    V = tin["verts"] - np.array([ox, oy, 0.0])
    me = r.Mesh()
    for x, y, z in V:
        me.Vertices.Add(float(x), float(y), float(z))
    for a_, b_, c_ in tin["tris"]:
        me.Faces.AddFace(int(a_), int(b_), int(c_))
    me.Normals.ComputeNormals()
    me.Compact()
    m.Objects.AddMesh(me, _attrs(r, L_mesh, "quarry_model_whole_site"))

    base = (min(z for _, z in floors) if floors else float(V[:, 2].min())) - base_margin_m
    n_sol = 0
    for k, (poly, z) in enumerate(floors):
        ext = np.asarray(poly.exterior.coords) - np.array([ox, oy])
        crv = r.PolylineCurve([r.Point3d(float(x), float(y), float(z)) for x, y in ext])
        e = r.Extrusion.Create(crv, -(z - base), True)
        if e is None:
            continue
        for hole in poly.interiors:
            hc = np.asarray(hole.coords) - np.array([ox, oy])
            if len(hc) >= 4:
                e.AddInnerProfile(r.PolylineCurve([r.Point3d(float(x), float(y), float(z)) for x, y in hc]))
        m.Objects.AddExtrusion(e, _attrs(r, L_sol, f"bench_floor_{k}_{z:.2f}m"))
        n_sol += 1
    m.Objects.AddTextDot(f"Quarry Maker origin\nE {ox:.0f}\nN {oy:.0f}\n{crs}\nbench height {bench_h:g} m\n"
                         f"solids down to {base:.1f} m", r.Point3d(0, 0, float(V[:, 2].max())), _attrs(r, L_info, "origin"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not m.Write(str(out_path), 7):
        raise RuntimeError(f"could not write {out_path}")
    if log:
        log.info("Rhino file %s: %d lines, whole-site mesh %d triangles, %d bench solids (hidden layer)",
                 out_path.name, len(lines), len(tin["tris"]), n_sol)
    return {"lines": len(lines), "mesh_triangles": int(len(tin["tris"])), "solids": n_sol, "solid_base_m": base}
