"""Depth geometry: deprojection, plane fitting, PCD parsing.

Pure NumPy — no Open3D required. Open3D can be added later for ICP
registration (see ``printbot.inspection.reference``).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from printbot.types import Intrinsics, Plane

log = logging.getLogger(__name__)


def deproject(depth_mm: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Depth image -> HxWx3 camera-frame points (mm). Invalid depth -> NaN."""
    h, w = depth_mm.shape
    if (w, h) != (intr.width, intr.height):
        raise ValueError(
            f"depth image {w}x{h} does not match intrinsics {intr.width}x{intr.height}; "
            "depth must be aligned to the colour image"
        )
    z = depth_mm.astype(np.float64)
    z[z <= 0] = np.nan
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - intr.ppx) * z / intr.fx
    y = (v - intr.ppy) * z / intr.fy
    return np.dstack([x, y, z])


def project(points_mm: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Nx3 camera-frame points -> Nx2 pixel coordinates (u, v)."""
    p = np.asarray(points_mm, dtype=np.float64)
    u = p[:, 0] * intr.fx / p[:, 2] + intr.ppx
    v = p[:, 1] * intr.fy / p[:, 2] + intr.ppy
    return np.stack([u, v], axis=1)


def _plane_from_points(p: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    n = np.cross(p[1] - p[0], p[2] - p[0])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n = n / norm
    return n, -float(n @ p[0])


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int = 200,
    inlier_mm: float = 1.5,
    rng: Optional[np.random.Generator] = None,
    max_points: int = 20000,
) -> Optional[tuple[Plane, np.ndarray]]:
    """Robustly fit a plane to Nx3 points. Returns (plane, inlier_bool_mask) or None.

    The plane normal is oriented toward the camera origin, so positive signed
    distance means "between plate and camera", i.e. height above the plate.
    The final plane is refined by least squares (SVD) on the inliers.
    """
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) < 3:
        return None
    rng = rng or np.random.default_rng(0)
    sample = pts if len(pts) <= max_points else pts[rng.choice(len(pts), max_points, replace=False)]

    best_n, best_d, best_count = None, 0.0, -1
    for _ in range(iterations):
        cand = _plane_from_points(sample[rng.choice(len(sample), 3, replace=False)])
        if cand is None:
            continue
        n, d = cand
        count = int((np.abs(sample @ n + d) < inlier_mm).sum())
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None:
        return None

    inl = np.abs(sample @ best_n + best_d) < inlier_mm
    centroid = sample[inl].mean(axis=0)
    _, _, vt = np.linalg.svd(sample[inl] - centroid, full_matrices=False)
    n = vt[-1]
    d = -float(n @ centroid)
    if d < 0:  # orient toward camera at origin: signed distance of origin = d must be > 0
        n, d = -n, -d
    plane = Plane(normal=n, d=d)
    full_inliers = np.zeros(len(points), dtype=bool)
    finite = np.isfinite(points).all(axis=1)
    full_inliers[finite] = np.abs(plane.signed_distance(points[finite])) < inlier_mm
    return plane, full_inliers


def height_map(points_hw3: np.ndarray, plane: Plane) -> np.ndarray:
    """HxW height above plane (mm); NaN where depth invalid."""
    h, w, _ = points_hw3.shape
    flat = points_hw3.reshape(-1, 3)
    out = np.full(len(flat), np.nan)
    ok = np.isfinite(flat).all(axis=1)
    out[ok] = plane.signed_distance(flat[ok])
    return out.reshape(h, w)


# --------------------------------------------------------------------------- PCD

_PCD_NUMPY = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2",
              ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}


def parse_pcd(data: bytes, units: str = "auto") -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Parse PCD bytes (ASCII or binary) -> (Nx3 xyz in mm, Nx3 uint8 rgb or None).

    ``units``: "mm", "m", or "auto". Viam documents millimetres as its internal
    unit, but whether serialized PCD from ``Camera.get_point_cloud`` is in
    metres is NOT verified in our references — "auto" treats data whose median
    |z| < 10 as metres. Verify on hardware and pin the setting.
    """
    header_end = 0
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    n_points = 0
    encoding = None
    for _ in range(32):
        nl = data.index(b"\n", header_end)
        line = data[header_end:nl].decode("ascii", errors="replace").strip()
        header_end = nl + 1
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        key = key.upper()
        if key == "FIELDS":
            fields = vals
        elif key == "SIZE":
            sizes = [int(v) for v in vals]
        elif key == "TYPE":
            types = vals
        elif key == "COUNT":
            counts = [int(v) for v in vals]
        elif key == "POINTS":
            n_points = int(vals[0])
        elif key == "DATA":
            encoding = vals[0].lower()
            break
    if encoding is None or not fields:
        raise ValueError("invalid PCD: missing FIELDS or DATA header")
    counts = counts or [1] * len(fields)
    if any(c != 1 for c in counts):
        raise ValueError("PCD fields with COUNT>1 are not supported")

    if encoding == "ascii":
        body = data[header_end:].decode("ascii").split()
        arr = np.array(body, dtype=np.float64).reshape(-1, len(fields))
        cols = {f: arr[:, i] for i, f in enumerate(fields)}
        rgb_raw = cols.get("rgb")
        if rgb_raw is not None:
            rgb_raw = rgb_raw.astype(np.float32).view(np.uint32) if np.any(rgb_raw % 1) else rgb_raw.astype(np.uint32)
    elif encoding == "binary":
        dtype = np.dtype([(f, _PCD_NUMPY[(t, s)]) for f, t, s in zip(fields, types, sizes)])
        rec = np.frombuffer(data, dtype=dtype, count=n_points, offset=header_end)
        cols = {f: rec[f] for f in fields}
        rgb_raw = cols.get("rgb")
        if rgb_raw is not None:
            rgb_raw = np.ascontiguousarray(rgb_raw).view(np.uint32)
    else:
        raise ValueError(f"PCD encoding '{encoding}' not supported (ascii/binary only)")

    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)
    if units == "m" or (units == "auto" and len(xyz) and np.nanmedian(np.abs(xyz[:, 2])) < 10):
        xyz *= 1000.0
    rgb = None
    if rgb_raw is not None:
        r = (rgb_raw >> 16) & 0xFF
        g = (rgb_raw >> 8) & 0xFF
        b = rgb_raw & 0xFF
        rgb = np.stack([r, g, b], axis=1).astype(np.uint8)
    return xyz, rgb
