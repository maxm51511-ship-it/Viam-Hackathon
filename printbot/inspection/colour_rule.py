"""Demo colour rule: RED material = failed print, WHITE material = good print.

Used for the hackathon demo, where real failed prints are not available. Red
prints or red filament scraps stand in for failures. Depth still decides
*what* is material (see PlateSegmenter), so the plate colour does not matter;
colour only labels each detected object.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from printbot.types import ObjectEstimate, Verdict


def colour_fractions(bgr: np.ndarray, mask: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
    """Return (red_fraction, white_fraction) of the masked pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[mask]
    if len(hsv) == 0:
        return 0.0, 0.0
    h = hsv[:, 0].astype(float) * 2.0          # OpenCV hue 0-179 -> degrees
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0
    red = (s >= cfg["red_s_min"]) & (v >= cfg["red_v_min"]) & ((h <= cfg["red_hue_max"]) | (h >= cfg["red_hue_min"]))
    white = (s <= cfg["white_s_max"]) & (v >= cfg["white_v_min"])
    return float(red.mean()), float(white.mean())


def colour_vote(bgr: np.ndarray, obj: ObjectEstimate, cfg: dict[str, Any]) -> tuple[Verdict, float, str]:
    red, white = colour_fractions(bgr, obj.mask, cfg)
    frac, dom = cfg["colour_frac"], cfg["colour_dominance"]
    if red >= frac and red >= dom * white:
        return Verdict.FAILED, min(0.95, 0.6 + red * 0.4), f"RED material {red:.0%} -> failed (demo rule)"
    if white >= frac and white >= dom * red:
        return Verdict.SUCCESS, min(0.95, 0.6 + white * 0.4), f"WHITE material {white:.0%} -> good (demo rule)"
    return Verdict.UNCERTAIN, 0.3, f"colour unclear (red {red:.0%}, white {white:.0%})"
