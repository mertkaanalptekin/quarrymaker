"""Read and update config.yaml without losing its comments.

The app changes only the values it manages (area, Blender path, bench
height, pit outline ...). Every other line, comment and manual edit in
config.yaml stays exactly as it was.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(float(v)) if isinstance(v, float) else str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(yaml_scalar(x) for x in v) + "]"
    s = str(v)
    if re.fullmatch(r"[A-Za-z0-9_./:\\-]+", s) and s.lower() not in ("null", "true", "false", "yes", "no"):
        return s
    return json.dumps(s)  # a JSON string is a valid YAML double-quoted string


def set_value(text: str, section: str | None, key: str, value) -> str:
    """Set `key` (inside top-level `section`, or top-level if None) in YAML text."""
    lines = text.splitlines()
    val = yaml_scalar(value)
    if section is None:
        start, end, indent = 0, len(lines), ""
    else:
        start = next((i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(section)}\s*:", ln)), None)
        if start is None:
            lines += [f"{section}:", f"  {key}: {val}"]
            return "\n".join(lines) + "\n"
        start += 1
        end = start
        while end < len(lines) and (lines[end].startswith((" ", "\t")) or not lines[end].strip()):
            end += 1
        indent = "  "
    pat = re.compile(rf"^({re.escape(indent)}{re.escape(key)}\s*:\s*)([^#]*?)(\s+#.*)?$")
    for i in range(start, end):
        if section is None and lines[i].startswith((" ", "\t")):
            continue
        m = pat.match(lines[i])
        if m:
            comment = m.group(3) or ""
            lines[i] = f"{m.group(1)}{val}{comment}"
            return "\n".join(lines) + "\n"
    # key missing: append at the end of the section
    ins = end
    while ins > start and not lines[ins - 1].strip():
        ins -= 1
    lines.insert(ins, f"{indent}{key}: {val}")
    return "\n".join(lines) + "\n"


def update_config(path: Path, updates: dict) -> None:
    """updates: {"aoi.west": 27.4, "blender.path": "...", "crs_epsg": 32635, ...}"""
    text = Path(path).read_text(encoding="utf-8")
    for dotted, value in updates.items():
        if "." in dotted:
            section, key = dotted.split(".", 1)
        else:
            section, key = None, dotted
        text = set_value(text, section, key, value)
    tmp = Path(path).with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def utm_epsg_for(lon: float, lat: float) -> int:
    """WGS84 UTM zone EPSG code for a point (e.g. 32635 = UTM 35N)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone
