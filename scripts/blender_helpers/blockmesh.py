"""Block mesh: a terrain grid with truly vertical walls (numpy only).

A height grid (one value per cell) becomes a surface where every cell is a
tile. Neighbouring cells whose heights differ by less than `wall_m` share their
corner vertices, so gentle ground stays smooth. Where they differ by more
(a quarry wall), the tiles get separate vertices at their own heights and a
vertical wall face is put between them: sharp crest and toe edges, flat floors.

build(z, x0, y0, res, wall_m) -> dict with
  co        (V, 3) float vertex coordinates (x right, y up, z)
  loops     (L,) int vertex index per face corner
  starts    (F,) int first loop of each face
  totals    (F,) int corners per face (3 or 4)
  cell      (F,) int index of the grid cell the face belongs to (row * w + col)
  is_wall   (F,) bool
x0, y0: coordinates of the north-west corner of cell (0, 0); rows go south.
"""
from __future__ import annotations

import numpy as np

# slots of the four cells around a corner: 0 NW, 1 NE, 2 SW, 3 SE


def build(z, x0: float, y0: float, res: float, wall_m: float = 2.5, smooth_iter: int = 6):
    z = np.asarray(z, dtype="float64")
    h, w = z.shape
    zp = np.pad(z, 1, mode="edge")
    # cells around each corner (h+1, w+1, 4)
    S = np.stack([zp[:-1, :-1], zp[:-1, 1:], zp[1:, :-1], zp[1:, 1:]], axis=-1)
    order = np.argsort(S, axis=-1, kind="stable")
    vals = np.take_along_axis(S, order, -1)
    gid_sorted = np.concatenate([np.zeros(S.shape[:2] + (1,), "int64"),
                                 np.cumsum(np.diff(vals, axis=-1) > wall_m, axis=-1)], axis=-1)
    grp = np.empty_like(gid_sorted)
    np.put_along_axis(grp, order, gid_sorted, -1)
    ngroups = gid_sorted[..., -1] + 1                        # 1..4 vertices at this corner
    # vertex height = mean of the cells in its group
    zg = np.zeros(S.shape[:2] + (4,))
    for g in range(4):
        m = grp == g
        cnt = m.sum(-1)
        zg[..., g] = np.where(cnt > 0, (S * m).sum(-1) / np.maximum(cnt, 1), np.nan)
    offset = (np.cumsum(ngroups.ravel()) - ngroups.ravel()).reshape(ngroups.shape)
    nv = int(ngroups.sum())
    rr, cc = np.mgrid[0:h + 1, 0:w + 1]
    cx = (x0 + cc * res).astype("float64")
    cy = (y0 - rr * res).astype("float64")
    if smooth_iter > 0:
        cx, cy = _straighten_walls(cx, cy, ngroups > 1, res, smooth_iter)
    co = np.zeros((nv, 3))
    for g in range(4):
        m = ngroups > g
        vid = offset[m] + g
        co[vid, 0] = cx[m]
        co[vid, 1] = cy[m]
        co[vid, 2] = zg[..., g][m]

    def vid(r, c, slot):
        return offset[r, c] + grp[r, c, slot]

    I, J = np.mgrid[0:h, 0:w]
    # floor tiles: NW -> SW -> SE -> NE (counter-clockwise from above)
    tiles = np.stack([vid(I, J, 3), vid(I + 1, J, 1), vid(I + 1, J + 1, 0), vid(I, J + 1, 2)], -1).reshape(-1, 4)
    tile_cell = (I * w + J).ravel()

    faces, cells = [tiles], [tile_cell]
    tris, tri_cells = [], []
    # walls between east-west neighbours (cell (i,j) | cell (i,j+1))
    i, j = np.mgrid[0:h, 0:w - 1]
    a_t, b_t = vid(i, j + 1, 2), vid(i, j + 1, 3)
    a_b, b_b = vid(i + 1, j + 1, 0), vid(i + 1, j + 1, 1)
    _walls(a_t, a_b, b_b, b_t, (i * w + j), faces, cells, tris, tri_cells)
    # walls between north-south neighbours (cell (i,j) above cell (i+1,j))
    i, j = np.mgrid[0:h - 1, 0:w]
    n_l, s_l = vid(i + 1, j, 1), vid(i + 1, j, 3)
    n_r, s_r = vid(i + 1, j + 1, 0), vid(i + 1, j + 1, 2)
    _walls(n_l, n_r, s_r, s_l, (i * w + j), faces, cells, tris, tri_cells)

    quads = np.concatenate(faces)
    qcell = np.concatenate(cells)
    tri = np.concatenate(tris) if tris else np.zeros((0, 3), "int64")
    tcell = np.concatenate(tri_cells) if tri_cells else np.zeros(0, "int64")
    nq, nt = len(quads), len(tri)
    loops = np.concatenate([quads.reshape(-1), tri.reshape(-1)])
    totals = np.concatenate([np.full(nq, 4), np.full(nt, 3)])
    starts = np.concatenate([[0], np.cumsum(totals)[:-1]])
    is_wall = np.ones(nq + nt, bool)
    is_wall[:len(tiles)] = False
    return {"co": co, "loops": loops.astype("int64"), "starts": starts.astype("int64"),
            "totals": totals.astype("int64"), "cell": np.concatenate([qcell, tcell]).astype("int64"),
            "is_wall": is_wall}


def _straighten_walls(cx, cy, wall, res, iters, lam=0.5, max_shift=0.7):
    """Slide the corners that lie on walls toward the mean of their neighbouring
    wall corners, so walls that run diagonally across the grid become straight
    instead of 2 m staircases in plan. All vertices at a corner move together,
    so walls stay vertical. Wall ends (one wall neighbour) stay put; shifts are
    capped at max_shift * res so tiles never fold."""
    x0, y0 = cx.copy(), cy.copy()
    wf = wall.astype("float64")
    for _ in range(iters):
        sx = np.zeros_like(cx)
        sy = np.zeros_like(cy)
        n = np.zeros_like(cx)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            nb = np.roll(np.roll(wf, -dr, 0), -dc, 1)
            # no wrap-around at the grid border
            if dr == 1:
                nb[-1, :] = 0
            if dr == -1:
                nb[0, :] = 0
            if dc == 1:
                nb[:, -1] = 0
            if dc == -1:
                nb[:, 0] = 0
            sx += nb * np.roll(np.roll(cx, -dr, 0), -dc, 1)
            sy += nb * np.roll(np.roll(cy, -dr, 0), -dc, 1)
            n += nb
        move = wall & (n >= 2)
        ax = np.where(n > 0, sx / np.maximum(n, 1), cx)
        ay = np.where(n > 0, sy / np.maximum(n, 1), cy)
        cx = np.where(move, cx + lam * (ax - cx), cx)
        cy = np.where(move, cy + lam * (ay - cy), cy)
    lim = max_shift * res
    cx = x0 + np.clip(cx - x0, -lim, lim)
    cy = y0 + np.clip(cy - y0, -lim, lim)
    return cx, cy


def _walls(p1, p2, p3, p4, cell, faces, cells, tris, tri_cells):
    """Edge between two cells with corner vertices (p1 top/left of cell A,
    p2 bottom of A, p3 bottom of B, p4 top of B). A wall exists where the two
    cells use different vertices at either end of the edge."""
    d_a = p1 != p4
    d_b = p2 != p3
    quad = d_a & d_b
    faces.append(np.stack([p1[quad], p2[quad], p3[quad], p4[quad]], -1))
    cells.append(cell[quad])
    t1 = d_a & ~d_b                     # wall closes to a point at the second end
    tris.append(np.stack([p1[t1], p2[t1], p4[t1]], -1))
    tri_cells.append(cell[t1])
    t2 = ~d_a & d_b
    tris.append(np.stack([p1[t2], p2[t2], p3[t2]], -1))
    tri_cells.append(cell[t2])
