"""Expected-geometry references from STL and G-code.

Current scope (MVP): compare *gross dimensions* — footprint and height — of the
observed object against the intended part. This catches the most common
failures (detached part, print stopped early, blob/spaghetti) cheaply and is
robust to partial visibility from a single fixed camera.

Not yet implemented (needs Open3D + calibrated extrinsics, see docs/PLAN.md):
full point-cloud registration (ICP) of the observed cloud against a sampled
STL surface, and per-region deviation maps. A single camera sees only the
top/front faces, so registration must use visibility-aware comparison and
multiple observations before declaring a defect.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from printbot.types import ObjectEstimate


@dataclass(frozen=True)
class ExpectedPart:
    footprint_mm: tuple[float, float]   # (long, short) side of XY bounding rect
    height_mm: float
    source: str


@dataclass
class ReferenceCheck:
    consistent: bool
    reasons: list[str] = field(default_factory=list)
    height_ratio: float = float("nan")
    footprint_ratio: float = float("nan")


# --------------------------------------------------------------------- STL

def load_stl(path: str | Path) -> np.ndarray:
    """Load an STL (binary or ASCII) -> (N, 3, 3) triangle vertices."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].astype(np.float64)
    text = data.decode("ascii", errors="replace")
    verts = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not verts or len(verts) % 3:
        raise ValueError(f"{path}: not a valid STL")
    return np.array(verts, dtype=np.float64).reshape(-1, 3, 3)


def expected_from_stl(path: str | Path) -> ExpectedPart:
    """Assumes the STL is already in print orientation (Z up) as sliced."""
    v = load_stl(path).reshape(-1, 3)
    ext = v.max(axis=0) - v.min(axis=0)
    a, b = sorted(ext[:2], reverse=True)
    return ExpectedPart((float(a), float(b)), float(ext[2]), f"stl:{Path(path).name}")


# ------------------------------------------------------------------ G-code

_WORD = re.compile(r"([GMXYZEF])(-?\d*\.?\d+)", re.IGNORECASE)


def gcode_extrusion_points(lines: Iterable[str]) -> np.ndarray:
    """Return Nx3 end points of extruding G0/G1 moves (handles G90/G91, M82/M83)."""
    pos = {"X": 0.0, "Y": 0.0, "Z": 0.0, "E": 0.0}
    abs_xyz, abs_e = True, True
    pts: list[tuple[float, float, float]] = []
    for raw in lines:
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        words = {k.upper(): float(v) for k, v in _WORD.findall(line)}
        g = words.get("G")
        m = words.get("M")
        if g == 90: abs_xyz, abs_e = True, True; continue
        if g == 91: abs_xyz, abs_e = False, False; continue
        if m == 82: abs_e = True; continue
        if m == 83: abs_e = False; continue
        if g == 92:
            for k in "XYZE":
                if k in words: pos[k] = words[k]
            continue
        if g not in (0, 1):
            continue
        extruded = False
        for k in "XYZ":
            if k in words:
                pos[k] = words[k] if abs_xyz else pos[k] + words[k]
        if "E" in words:
            e = words["E"]
            de = e - pos["E"] if abs_e else e
            pos["E"] = e if abs_e else pos["E"] + e
            extruded = de > 1e-6 and any(k in words for k in "XY")
        if extruded:
            pts.append((pos["X"], pos["Y"], pos["Z"]))
    return np.array(pts, dtype=np.float64).reshape(-1, 3)


def expected_from_gcode(path: str | Path, skirt_trim_pct: float = 1.0) -> ExpectedPart:
    """Footprint/height from extrusion moves.

    Skirts/brims/purge lines inflate the XY extent; ``skirt_trim_pct`` trims
    that percentile from each side, which is only a rough mitigation — prefer
    the STL when available.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        p = gcode_extrusion_points(f)
    if len(p) == 0:
        raise ValueError(f"{path}: no extrusion moves found")
    lo = np.percentile(p[:, :2], skirt_trim_pct, axis=0)
    hi = np.percentile(p[:, :2], 100 - skirt_trim_pct, axis=0)
    a, b = sorted(hi - lo, reverse=True)
    return ExpectedPart((float(a), float(b)), float(p[:, 2].max()), f"gcode:{Path(path).name}")


# ---------------------------------------------------------------- compare

def compare(obs: ObjectEstimate, exp: ExpectedPart,
            height_tol: float = 0.15, footprint_tol: float = 0.2) -> Optional[ReferenceCheck]:
    """Compare observed gross dimensions to expected. Returns None if the
    observation lacks 3D data (cannot judge — never a defect by default)."""
    if not np.isfinite(obs.max_height_mm) or not np.isfinite(obs.footprint_mm[0]):
        return None
    hr = obs.max_height_mm / exp.height_mm if exp.height_mm > 0 else float("nan")
    fr = obs.footprint_mm[0] / exp.footprint_mm[0] if exp.footprint_mm[0] > 0 else float("nan")
    reasons = []
    if abs(hr - 1) > height_tol:
        reasons.append(f"height {obs.max_height_mm:.1f} mm vs expected {exp.height_mm:.1f} mm ({hr:.2f}x)")
    if abs(fr - 1) > footprint_tol:
        reasons.append(f"footprint {obs.footprint_mm[0]:.1f} mm vs expected {exp.footprint_mm[0]:.1f} mm ({fr:.2f}x)")
    return ReferenceCheck(consistent=not reasons, reasons=reasons, height_ratio=hr, footprint_ratio=fr)
