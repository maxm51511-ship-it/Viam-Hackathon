"""Build-plate segmentation (colour + depth).

Strategy
--------
1. HSV thresholding with one or more configurable ranges finds *candidate*
   exposed-plate pixels. Nothing assumes a particular plate or filament colour —
   profiles are data in ``config/settings.json``.
2. The largest candidate region (after morphology) defines the plate boundary
   (convex hull). Objects on the plate punch holes in the colour mask; the hull
   restores them as "inside the plate".
3. If depth is available, a RANSAC plane is fitted to the *exposed* plate
   pixels only. Every pixel inside the plate boundary whose height above that
   plane exceeds ``object_min_height_mm`` is material. Depth decides — colour
   only seeds the plane — so filament that happens to match the plate colour,
   shadows and reflections are handled.
4. Without depth, falls back to "inside hull AND not plate-coloured", flagged
   in ``notes`` so downstream logic can lower confidence.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from printbot.perception.geometry import deproject, fit_plane_ransac, height_map
from printbot.types import Frame, PlateSegmentation

log = logging.getLogger(__name__)


class PlateSegmenter:
    def __init__(self, cfg: dict[str, Any]):
        self.hsv_ranges = [
            (np.array(r["lower"], dtype=np.uint8), np.array(r["upper"], dtype=np.uint8))
            for r in cfg["hsv_ranges"]
        ]
        self.roi = np.array(cfg["roi_polygon_px"], dtype=np.int32) if cfg.get("roi_polygon_px") else None
        k = int(cfg.get("morph_kernel_px", 7))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.min_plate_area_frac = float(cfg.get("min_plate_area_frac", 0.05))
        self.min_height = float(cfg.get("object_min_height_mm", 1.5))
        self.ransac_iterations = int(cfg.get("ransac_iterations", 200))
        self.ransac_inlier = float(cfg.get("ransac_inlier_mm", 1.5))
        self.min_speck_px = int(cfg.get("min_speck_px", 20))

    # ------------------------------------------------------------------ colour
    def color_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for lo, hi in self.hsv_ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        if self.roi is not None:
            roi = np.zeros_like(mask)
            cv2.fillPoly(roi, [self.roi], 255)
            mask &= roi
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask > 0

    def plate_boundary(self, color_mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(color_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self.min_plate_area_frac * color_mask.size:
            return None
        return cv2.convexHull(biggest).reshape(-1, 2).astype(np.int32)

    # ------------------------------------------------------------------ main
    def segment(self, frame: Frame) -> PlateSegmentation:
        notes: list[str] = []
        cmask = self.color_mask(frame.bgr)
        polygon = self.plate_boundary(cmask)
        if polygon is None and self.roi is not None:
            polygon = self.roi
            notes.append("plate colour region too small; using configured ROI as plate boundary")
        if polygon is None:
            empty = np.zeros(cmask.shape, dtype=bool)
            return PlateSegmentation(empty, None, empty, None, None, False,
                                     ["build plate not found: check HSV profile / lighting / ROI"])

        inside = np.zeros(cmask.shape, dtype=np.uint8)
        cv2.fillPoly(inside, [polygon], 1)
        inside = inside.astype(bool)
        exposed = cmask & inside

        if frame.has_depth:
            pts = deproject(frame.depth_mm, frame.intrinsics)
            fit = fit_plane_ransac(pts[exposed], self.ransac_iterations, self.ransac_inlier)
            if fit is not None:
                plane, _ = fit
                hmap = height_map(pts, plane)
                obj = inside & (np.nan_to_num(hmap, nan=-np.inf) > self.min_height)
                # Do NOT morphologically open here: it erases thin filament strands
                # (1-3 px wide), which are exactly what spaghetti detection needs.
                obj = remove_small_components(obj, self.min_speck_px)
                invalid_frac = float(np.isnan(hmap[inside]).mean())
                if invalid_frac > 0.2:
                    notes.append(f"{invalid_frac:.0%} of plate pixels have no depth (glare/occlusion)")
                return PlateSegmentation(inside, polygon, obj, hmap, plane, True, notes)
            notes.append("plane fit failed; falling back to colour-only object mask")

        notes.append("colour-only segmentation (no depth): lower reliability")
        obj = remove_small_components(inside & ~cmask, self.min_speck_px)
        return PlateSegmentation(inside, polygon, obj, None, None, False, notes)


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Drop connected components smaller than ``min_px`` (depth speckle) while
    preserving thin structures, unlike morphological opening."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]
