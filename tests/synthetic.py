"""Synthetic RGB-D scenes: a pinhole camera looking straight down at a plate.

Camera at 400 mm above the plate; plate is a rectangle of configurable colour;
background is a lighter colour farther away. Objects are drawn into both the
colour and depth images consistently, so perception can be tested end to end.
"""

from __future__ import annotations

import cv2
import numpy as np

from printbot.types import Frame, Intrinsics

W, H = 640, 480
INTR = Intrinsics(W, H, 600.0, 600.0, 320.0, 240.0)
PLATE_Z = 400.0
PLATE_RECT = (120, 80, 520, 400)  # u0, v0, u1, v1
# camera looks down: cam +Z -> world -Z, cam +Y -> world -Y; plate surface at world z = 0
T_WORLD_CAM = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, PLATE_Z], [0, 0, 0, 1]], dtype=float)


def mm_to_px(mm: float, z: float) -> float:
    return mm * INTR.fx / z


def base_scene(plate_bgr=(40, 40, 40), bg_bgr=(200, 200, 200), noise_mm=0.3, seed=0):
    rng = np.random.default_rng(seed)
    bgr = np.full((H, W, 3), bg_bgr, dtype=np.uint8)
    depth = np.full((H, W), 450.0)
    u0, v0, u1, v1 = PLATE_RECT
    bgr[v0:v1, u0:u1] = plate_bgr
    depth[v0:v1, u0:u1] = PLATE_Z
    depth += rng.normal(0, noise_mm, depth.shape)
    return bgr, depth, rng


def add_box(bgr, depth, center_px=(320, 240), size_mm=(30, 20), height_mm=15, color=(0, 0, 255), angle_deg=0.0):
    z_top = PLATE_Z - height_mm
    w = mm_to_px(size_mm[0], z_top)
    h = mm_to_px(size_mm[1], z_top)
    box = cv2.boxPoints((center_px, (w, h), angle_deg)).astype(np.int32)
    m = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(m, [box], 1)
    bgr[m > 0] = color
    depth[m > 0] = z_top
    return m > 0


def add_spaghetti(bgr, depth, rng, n_strands=9, color=(0, 0, 255)):
    u0, v0, u1, v1 = PLATE_RECT
    m = np.zeros(depth.shape, np.uint8)
    for _ in range(n_strands):
        pts = np.cumsum(rng.normal(0, 12, (8, 2)), axis=0) + rng.uniform([u0 + 60, v0 + 60], [u1 - 60, v1 - 60])
        strand = np.zeros_like(m)
        cv2.polylines(strand, [pts.astype(np.int32)], False, 1, thickness=3)
        hgt = rng.uniform(3, 8)
        depth[strand > 0] = PLATE_Z - hgt
        m |= strand
    bgr[m > 0] = color
    return m > 0


def frame(bgr, depth, with_depth=True) -> Frame:
    d = np.clip(depth, 0, 65535).astype(np.uint16) if with_depth else None
    return Frame(bgr=bgr, depth_mm=d, intrinsics=INTR if with_depth else None, source="synthetic")


def part_frame(**kw) -> Frame:
    bgr, depth, _ = base_scene(seed=kw.pop("seed", 0))
    add_box(bgr, depth, **kw)
    return frame(bgr, depth)


def spaghetti_frame(seed=1) -> Frame:
    bgr, depth, rng = base_scene(seed=seed)
    add_spaghetti(bgr, depth, rng)
    return frame(bgr, depth)


def empty_frame(seed=2) -> Frame:
    bgr, depth, _ = base_scene(seed=seed)
    return frame(bgr, depth)


class SequenceSource:
    """Frame source returning frames from a list, repeating the last one."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0

    async def get_frame(self):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return f
