"""Turn an object mask (+ height map) into ``ObjectEstimate`` blobs.

Each estimate carries shape descriptors the classifier uses to separate a
compact printed part from spaghetti/loose filament:

* solidity  = area / convex-hull area  (spaghetti is stringy -> low)
* extent    = area / min-area-rect area
* height_cv = std/mean of height inside the mask (spaghetti is bumpy)
* n_fragments = number of nearby pieces merged into this blob
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from printbot.perception.geometry import deproject
from printbot.types import Frame, ObjectEstimate, PlateSegmentation


def extract_objects(frame: Frame, seg: PlateSegmentation, cfg: dict[str, Any]) -> list[ObjectEstimate]:
    min_area = int(cfg.get("min_area_px", 150))
    dil = int(cfg.get("merge_dilate_px", 5))
    raw = seg.object_mask.astype(np.uint8)
    if raw.sum() == 0:
        return []

    # Merge nearby fragments (spaghetti strands) into one blob for counting,
    # but measure area on the raw mask.
    merged = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)))
    n, labels = cv2.connectedComponents(merged)
    points = deproject(frame.depth_mm, frame.intrinsics) if frame.has_depth else None

    out: list[ObjectEstimate] = []
    for lbl in range(1, n):
        region = (labels == lbl) & seg.object_mask
        area = int(region.sum())
        if area < min_area:
            continue
        n_frag, _ = cv2.connectedComponents(region.astype(np.uint8))
        ys, xs = np.nonzero(region)
        pts2d = np.stack([xs, ys], axis=1).astype(np.float32)
        hull_area = float(cv2.contourArea(cv2.convexHull(pts2d)))
        rect = cv2.minAreaRect(pts2d)
        rect_area = float(rect[1][0] * rect[1][1])
        est = ObjectEstimate(
            mask=region,
            pixel_area=area,
            bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            centroid_px=(float(xs.mean()), float(ys.mean())),
            min_rect=(tuple(rect[0]), tuple(rect[1]), float(rect[2])),
            # pixel-centre hulls slightly underestimate area -> clamp ratios to 1
            solidity=min(1.0, area / hull_area) if hull_area > 0 else float("nan"),
            extent=min(1.0, area / rect_area) if rect_area > 0 else float("nan"),
            n_fragments=max(1, n_frag - 1),
        )
        w, h = rect[1]
        est.yaw_deg = float(rect[2] if w >= h else rect[2] + 90.0)
        if seg.height_mm is not None and points is not None:
            _add_3d(est, region, seg.height_mm, points)
        out.append(est)
    out.sort(key=lambda e: e.pixel_area, reverse=True)
    return out


def _add_3d(est: ObjectEstimate, region: np.ndarray, hmap: np.ndarray, points: np.ndarray) -> None:
    hs = hmap[region]
    hs = hs[np.isfinite(hs)]
    if hs.size == 0:
        return
    est.max_height_mm = float(np.percentile(hs, 98))  # robust to depth speckle
    est.height_cv = float(hs.std() / hs.mean()) if hs.mean() > 0 else float("nan")

    # Top surface = points within the top 15 % of height. Its centre is a
    # *candidate* reference for grasping — not a validated grasp point.
    top = region & (np.nan_to_num(hmap, nan=-np.inf) >= 0.85 * est.max_height_mm)
    tp = points[top]
    tp = tp[np.isfinite(tp).all(axis=1)]
    if len(tp):
        est.top_center_cam_mm = np.median(tp, axis=0)

    # Metric footprint from the min-area-rect of all valid object points
    # (approximation: assumes camera roughly looks down on the plate).
    rp = points[region]
    rp = rp[np.isfinite(rp).all(axis=1)]
    if len(rp) >= 5:
        r = cv2.minAreaRect(rp[:, :2].astype(np.float32))
        a, b = sorted(r[1], reverse=True)
        est.footprint_mm = (float(a), float(b))


def largest(objects: list[ObjectEstimate]) -> Optional[ObjectEstimate]:
    return objects[0] if objects else None
