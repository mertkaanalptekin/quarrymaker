"""Stage 09 - quarry expansion over time (Esri World Imagery Wayback).

Esri's Wayback archive keeps every published version of World Imagery since
2014. Many versions show the same photo of a given place, so the stage
1. samples a few tiles of the site in every version and groups versions
   whose tiles are identical (= the same photo),
2. downloads each distinct photo of the site (zoom `wayback.zoom`),
3. runs the same quarry-area method as stage 07 on each,
4. reports the quarry area per date and maps where/when the quarry grew.

Dates are the Wayback release in which a photo first appeared; the photo
itself was captured somewhat earlier. Outputs in terrain/20_expansion/vNNN/
and exchange/rhino/vNNN/plan/quarry_expansion.dxf.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
DEFAULTS = {"enabled": True, "zoom": 16, "max_versions": 10, "sample_zoom": 15, "throttle_s": 0.05,
            "min_area_ha": 5.0, "min_match": 0.45, "persistence": 0.6}
ATTRIBUTION = ("Esri World Imagery Wayback - Source: Esri, Maxar, Earthstar Geographics, and the GIS "
               "User Community. Powered by Esri.")


def _get(sess, url, path=None, tries=4):
    import requests
    if path is not None and path.exists():
        return path.read_bytes()
    err = None
    for k in range(tries):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 200:
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(r.content)
                return r.content
            if r.status_code == 404:
                return b""
            err = f"HTTP {r.status_code}"
        except requests.RequestException as ex:
            err = str(ex)
        time.sleep(2 ** k)
    raise common.PipelineError(f"Wayback download failed ({err}): {url}")


def main(argv=None) -> int:
    import numpy as np
    import rasterio
    import requests
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    import linework as lw
    import terraces as T

    args, cfg, paths, log = common.stage_context(__doc__, "09", argv)
    wc = dict(DEFAULTS)
    wc.update(cfg.get("wayback") or {})
    if not args.force and common.stage_up_to_date(paths, cfg, "09"):
        log.info("Stage 09 up to date - skipping.")
        return 0
    if not wc["enabled"]:
        common.finish_stage(paths, cfg, "09", ["Expansion over time disabled (wayback.enabled: false)."], [], [], log)
        return 0
    imag = _imagery_module()
    meta = common.read_json(paths.base / "meta.json")
    e, crs = meta["extent"], meta["crs"]
    ox, oy = meta["origin_x"], meta["origin_y"]
    cache = paths.raw / "wayback"
    sess = requests.Session()
    sess.headers["User-Agent"] = "quarry-terrain-pipeline/1.0"

    conf = json.loads(_get(sess, CONFIG_URL, cache / "config.json").decode("utf-8"))
    rel = sorted(({"id": int(k), "date": v["itemTitle"].split("Wayback ")[-1].rstrip(")"),
                   "url": v["itemURL"]} for k, v in conf.items()), key=lambda r: r["date"])
    log.info("Wayback archive: %d versions (%s .. %s)", len(rel), rel[0]["date"], rel[-1]["date"])

    # 1. which versions show a different photo of the site? (sample tiles in the quarry)
    cfg_t = dict(cfg)
    cfg_t["imagery"] = dict(cfg.get("imagery") or {}, zoom=int(wc["sample_zoom"]))
    z, x0, y0, x1, y1 = imag.tile_range(cfg_t)
    xs = sorted({x0 + (x1 - x0) // 4, (x0 + x1) // 2, x1 - (x1 - x0) // 4})
    ys = sorted({y0 + (y1 - y0) // 4, (y0 + y1) // 2, y1 - (y1 - y0) // 4})
    samples = [(xs[0], ys[1]), (xs[1], ys[1]), (xs[2], ys[1]), (xs[1], ys[0]), (xs[1], ys[2])]
    groups = []
    for n, r in enumerate(rel):
        sig = []
        for (tx, ty) in samples:
            b = _get(sess, r["url"].format(level=z, row=ty, col=tx), cache / "samples" / str(r["id"]) / f"{z}_{tx}_{ty}")
            sig.append(hashlib.sha1(b).hexdigest()[:12])
            time.sleep(float(wc["throttle_s"]))
        sig = tuple(sig)
        if groups and groups[-1]["sig"] == sig:
            groups[-1]["last"] = r["date"]
        else:
            groups.append({"sig": sig, "first": r["date"], "last": r["date"], "id": r["id"], "url": r["url"]})
        if (n + 1) % 40 == 0:
            log.info("  compared %d/%d versions, %d different photos so far", n + 1, len(rel), len(groups))
    # a signature can come back later (rollback); keep the first appearance only
    seen, versions = set(), []
    for g in groups:
        if g["sig"] not in seen:
            seen.add(g["sig"])
            versions.append(g)
    log.info("Different photos of the site: %d", len(versions))
    if len(versions) > int(wc["max_versions"]):
        idx = np.unique(np.linspace(0, len(versions) - 1, int(wc["max_versions"])).round().astype(int))
        versions = [versions[i] for i in idx]
        log.info("Using %d of them, spread over time (wayback.max_versions)", len(versions))

    # 2-3. download each photo and find the quarry area on it
    res = 2.0
    W, H = int(round((e["xmax"] - e["xmin"]) / res)), int(round((e["ymax"] - e["ymin"]) / res))
    tf = from_origin(e["xmin"], e["ymax"], res, res)
    z0 = np.zeros((H, W), "float32")
    with rasterio.open(paths.processed / "dem_utm.tif") as s:
        reproject(rasterio.band(s, 1), z0, src_transform=s.transform, src_crs=s.crs, dst_transform=tf,
                  dst_crs=crs, resampling=Resampling.cubic)
    z0 = z0.astype(float)
    cfg_d = dict(cfg)
    cfg_d["imagery"] = dict(cfg.get("imagery") or {}, zoom=int(wc["zoom"]))
    zd, X0, Y0, X1, Y1 = imag.tile_range(cfg_d)
    out = paths.root / "terrain" / "20_expansion" / paths.run_id
    out.mkdir(parents=True, exist_ok=True)
    photos = []
    for k, v in enumerate(versions):
        log.info("Photo %d/%d (Wayback %s): %d tiles", k + 1, len(versions), v["first"],
                 (X1 - X0 + 1) * (Y1 - Y0 + 1))
        mosaic = np.zeros(((Y1 - Y0 + 1) * 256, (X1 - X0 + 1) * 256, 3), "uint8")
        for j, ty in enumerate(range(Y0, Y1 + 1)):
            for i, tx in enumerate(range(X0, X1 + 1)):
                tp = cache / "tiles" / str(v["id"]) / str(zd) / f"{tx}_{ty}"
                was_cached = tp.exists()
                b = _get(sess, v["url"].format(level=zd, row=ty, col=tx), tp)
                if b:
                    import io
                    with Image.open(io.BytesIO(b)) as im:
                        mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = np.asarray(im.convert("RGB"))
                if not was_cached:
                    time.sleep(float(wc["throttle_s"]))
        bx0, _, _, by1 = imag.tile_bounds_merc(X0, Y0, zd)
        px = 2 * imag.WEBM / 2 ** zd / 256
        rgb = np.zeros((3, H, W), "uint8")
        for b_ in range(3):
            reproject(mosaic[..., b_], rgb[b_], src_transform=from_origin(bx0, by1, px, px), src_crs="EPSG:3857",
                      dst_transform=tf, dst_crs=crs, resampling=Resampling.average)
        photos.append(rgb.transpose(1, 2, 0))

    # quality: structure of each photo vs the newest one (mismatched mosaics, clouds, haze)
    def structure(a):
        g = T.gauss(a.astype("float32").mean(-1), 8 / res)
        gy, gx = np.gradient(g)
        return np.hypot(gx, gy)[::4, ::4].ravel()
    ref_s = structure(photos[-1])
    keep = []
    for k, v in enumerate(versions):
        v["match"] = round(float(np.corrcoef(structure(photos[k]), ref_s)[0, 1]), 2)
        ok = v["match"] >= float(wc["min_match"]) or k == len(versions) - 1
        log.info("  %s: structure match with the newest photo %.2f%s", v["first"], v["match"],
                 "" if ok else " -> left out (mismatched, cloudy or hazy)")
        if ok:
            keep.append(k)
    dropped_versions = [versions[k]["first"] for k in range(len(versions)) if k not in keep]
    versions = [versions[k] for k in keep]
    photos = [photos[k] for k in keep]

    # same colours, same threshold: match every photo to the newest, use its threshold
    def match_hist(a, ref):
        out_ = np.empty_like(a)
        for b_ in range(3):
            src, tgt = a[..., b_].ravel(), ref[..., b_].ravel()
            sv, si, sc = np.unique(src, return_inverse=True, return_counts=True)
            tv, tc = np.unique(tgt, return_counts=True)
            out_[..., b_] = np.interp(np.cumsum(sc) / src.size, np.cumsum(tc) / tgt.size, tv)[si].reshape(a.shape[:2])
        return out_
    ref_mask, _, ref_info = T.quarry_area(photos[-1], z0, res, tf, min_area_ha=float(wc["min_area_ha"]))
    thr = ref_info["threshold"]
    log.info("Threshold from the newest photo: %.3f", thr)
    masks, rgbs = [], []
    for k, v in enumerate(versions):
        rgb = photos[k] if k == len(versions) - 1 else match_hist(photos[k], photos[-1])
        m = ref_mask if k == len(versions) - 1 else T.quarry_area(rgb, z0, res, tf, min_area_ha=float(wc["min_area_ha"]),
                                                                   threshold=thr)[0]
        v["area_ha"] = round(float(m.sum()) * res * res / 1e4, 1)
        log.info("  %s: quarry area %.1f ha", v["first"], v["area_ha"])
        masks.append(m)
        rgbs.append(np.asarray(Image.fromarray(photos[k]).resize((W // 4, H // 4))))
        with rasterio.open(out / f"quarry_area_{v['first']}.tif", "w", driver="GTiff", width=W, height=H,
                           count=1, dtype="uint8", crs=crs, transform=tf, compress="deflate") as ds:
            ds.write(m.astype("uint8"), 1)
    del photos

    # 4. growth: the first photo on which each cell is quarry
    # a cell counts as quarried from photo k when it is quarry there AND on most
    # later photos (one-off misdetections do not create early growth)
    first = np.full((H, W), -1, "int16")
    stack = np.stack(masks).astype("uint8")
    later = np.cumsum(stack[::-1], axis=0)[::-1]          # quarry count from k to the end
    for k in range(len(masks)):
        n_after = len(masks) - k
        persistent = stack[k].astype(bool) & (later[k] >= np.ceil(float(wc["persistence"]) * n_after))
        first[(first < 0) & persistent] = k
    del stack, later
    rows = []
    for k, v in enumerate(versions):
        new = (first == k).sum() * res * res / 1e4
        cum = float(((first >= 0) & (first <= k)).sum()) * res * res / 1e4
        v["persistent_ha"] = round(cum, 1)
        rows.append({"wayback_date": v["first"], "shown_until": v["last"], "quarry_area_ha": v["area_ha"],
                     "persistent_quarried_ha": round(cum, 1),
                     "new_since_previous_ha": round(float(new), 1) if k else None})
    common.write_json(out / "expansion.json", {"versions": rows, "left_out": dropped_versions,
                                                "attribution": ATTRIBUTION,
                                                "note": "Dates are Wayback release dates; photos were "
                                                        "captured somewhat earlier."})
    (out / "expansion.csv").write_text("wayback_date,shown_until,quarry_area_ha,persistent_quarried_ha,new_since_previous_ha\n" + "".join(
        f"{r['wayback_date']},{r['shown_until']},{r['quarry_area_ha']},{r['persistent_quarried_ha']},{r['new_since_previous_ha'] or ''}\n"
        for r in rows), encoding="utf-8")

    layers, files = {}, [out / "expansion.json", out / "expansion.csv"]
    for k, v in enumerate(versions):
        polys = lw.mask_polygons(masks[k], tf, min_area_m2=20000, simplify_m=3.0)
        layers[f"QUARRY_{v['first']}"] = polys
        p = out / f"quarry_outline_{v['first']}.geojson"
        lw.write_geojson(p, polys, crs, {"wayback_date": v["first"]})
        files.append(p)
    plan = paths.rhino / "plan"
    plan.mkdir(parents=True, exist_ok=True)
    dxf = plan / "quarry_expansion.dxf"
    lw.write_dxf(dxf, layers, origin=(ox, oy))
    files.append(dxf)

    # check images: growth map, area chart, photos with outlines
    from PIL import ImageDraw
    import colorsys
    cols = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(0.66 * (1 - k / max(1, len(versions) - 1)), 0.85, 0.95))
            for k in range(len(versions))]
    base = rgbs[-1].astype(float) * 0.6 + 40
    small_first = np.asarray(Image.fromarray((first + 1).astype("uint8")).resize((W // 4, H // 4), Image.NEAREST)).astype(int) - 1
    for k in range(len(versions)):
        base[small_first == k] = base[small_first == k] * 0.35 + np.array(cols[k]) * 0.65
    gm = Image.fromarray(base.astype("uint8"))
    d = ImageDraw.Draw(gm)
    for k, v in enumerate(versions):
        d.rectangle([8, 8 + 16 * k, 260, 22 + 16 * k], fill=(255, 255, 255))
        d.rectangle([10, 10 + 16 * k, 22, 20 + 16 * k], fill=cols[k])
        d.text((28, 9 + 16 * k), f"{v['first']}: {v['area_ha']:.0f} ha", fill=(0, 0, 0))
    p_map = paths.checks / "09_expansion_map.png"
    gm.save(p_map)
    ch = Image.new("RGB", (900, 380), "white")
    d = ImageDraw.Draw(ch)
    amax = max(v["persistent_ha"] for v in versions) or 1
    for k, v in enumerate(versions):
        x = 60 + k * (800 / max(1, len(versions)))
        hgt = 300 * v["persistent_ha"] / amax
        d.rectangle([x, 340 - hgt, x + 800 / max(1, len(versions)) - 12, 340], fill=cols[k])
        d.text((x, 345), v["first"], fill=(0, 0, 0))
        d.text((x, 325 - hgt), f"{v['persistent_ha']:.0f} ha", fill=(0, 0, 0))
    d.text((60, 10), "Quarried area by date (quarry on that photo and on most later ones) - Esri Wayback", fill=(0, 0, 0))
    p_chart = paths.checks / "09_area_per_date.png"
    ch.save(p_chart)
    summary = [f"{len(versions)} different photos of the site in the Wayback archive ({rel[0]['date']} .. {rel[-1]['date']})"]
    summary += [f"{r['wayback_date']}: quarried {r['persistent_quarried_ha']:.0f} ha (outline on this photo {r['quarry_area_ha']:.0f} ha)"
                + (f", +{r['new_since_previous_ha']:.0f} ha new" if r['new_since_previous_ha'] else "") for r in rows]
    if dropped_versions:
        summary.append("Left out (mismatched, cloudy or hazy photo): " + ", ".join(dropped_versions))
    summary.append(f"Outlines per date: {common.rel(paths, dxf)} and GeoJSON in {common.rel(paths, out)}")
    common.finish_stage(paths, cfg, "09", summary, files, [p_map, p_chart], log)
    return 0


def _imagery_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch_imagery", Path(__file__).resolve().parent / "02_fetch_imagery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    common.run_main(main, "09")
