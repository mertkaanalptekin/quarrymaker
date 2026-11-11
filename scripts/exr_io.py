"""Minimal OpenEXR reader/writer for 32-bit float, uncompressed, scanline
images (no external dependency). Used for the Gaea heightmap in real metres.

write_exr(path, arr)  arr: (H, W) float -> channels R, G, B all equal
                      (single-channel EXRs are not read by every tool).
read_exr(path)        -> dict {channel: (H, W) float32}; only reads files of
                      the kind written here (FLOAT, NO_COMPRESSION, scanline).
"""
from __future__ import annotations

import struct
from pathlib import Path

MAGIC = 20000630
FLOAT = 2


def _attr(name: str, typ: str, data: bytes) -> bytes:
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data


def write_exr(path, arr, channels=("B", "G", "R")) -> None:
    import numpy as np

    a = np.ascontiguousarray(arr, dtype="<f4")
    if a.ndim != 2:
        raise ValueError("write_exr expects a 2-D array")
    h, w = a.shape
    chans = sorted(channels)  # EXR stores channels in alphabetical order
    chlist = b"".join(c.encode() + b"\0" + struct.pack("<iB3xii", FLOAT, 0, 1, 1) for c in chans) + b"\0"
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = (struct.pack("<ii", MAGIC, 2)
              + _attr("channels", "chlist", chlist)
              + _attr("compression", "compression", b"\0")
              + _attr("dataWindow", "box2i", box)
              + _attr("displayWindow", "box2i", box)
              + _attr("lineOrder", "lineOrder", b"\0")
              + _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
              + _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
              + _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
              + b"\0")
    line_bytes = w * 4 * len(chans)
    block = 8 + line_bytes
    first = len(header) + 8 * h
    offsets = struct.pack(f"<{h}Q", *[first + y * block for y in range(h)])
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(offsets)
        for y in range(h):
            row = a[y].tobytes()
            fh.write(struct.pack("<ii", y, line_bytes))
            fh.write(row * len(chans))


def read_exr(path) -> dict:
    import numpy as np

    data = Path(path).read_bytes()
    magic, version = struct.unpack_from("<ii", data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path} is not an OpenEXR file")
    if version & 0x200:
        raise ValueError("tiled EXR not supported by this reader")
    pos = 8
    attrs = {}
    while data[pos] != 0:
        end = data.index(b"\0", pos)
        name = data[pos:end].decode()
        pos = end + 1
        end = data.index(b"\0", pos)
        typ = data[pos:end].decode()
        pos = end + 1
        (size,) = struct.unpack_from("<i", data, pos)
        pos += 4
        attrs[name] = (typ, data[pos:pos + size])
        pos += size
    pos += 1
    if attrs["compression"][1] != b"\0":
        raise ValueError("compressed EXR not supported by this reader")
    raw = attrs["channels"][1]
    chans, p = [], 0
    while raw[p] != 0:
        end = raw.index(b"\0", p)
        nm = raw[p:end].decode()
        ptype, _, _, _ = struct.unpack_from("<iB3xii", raw, end + 1)
        if ptype != FLOAT:
            raise ValueError("only FLOAT channels supported by this reader")
        chans.append(nm)
        p = end + 1 + 16
    x0, y0, x1, y1 = struct.unpack("<iiii", attrs["dataWindow"][1])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    offsets = struct.unpack_from(f"<{h}Q", data, pos)
    out = {c: np.empty((h, w), dtype="float32") for c in chans}
    for off in offsets:
        y, size = struct.unpack_from("<ii", data, off)
        line = np.frombuffer(data, dtype="<f4", count=w * len(chans), offset=off + 8)
        for i, c in enumerate(chans):
            out[c][y - y0] = line[i * w:(i + 1) * w]
    return out
