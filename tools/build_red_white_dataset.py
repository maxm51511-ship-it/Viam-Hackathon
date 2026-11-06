"""Build a YOLO detection dataset (classes: failed=red, good=white) from three
Viam dataset exports. Fully offline - no robot connection, no hardware risk.

Sources:
  data/                 - 5 images, mixed shapes (cube/sphere/pyramid/
                           cylinder/cuboid), HAND-VERIFIED boxes (see below).
  raw_exports/good_only - "good prints only" capture: every detected object
                           is trivially labelled "good" (ground truth comes
                           from the dataset's own identity, not a colour
                           guess - these are white parts by construction).
  raw_exports/mixed     - white + red parts together: objects are found
                           automatically (simple, well-separated spheres/
                           cylinder - no plate-engraving contamination like
                           the first set had), then each box is labelled via
                           the repo's official printbot.inspection.colour_rule
                           thresholds.

Why the first 5 images use hand-verified boxes instead of automatic
detection: the plate's embossed brand text and engraved logo graphic sit in
the same brightness/saturation range as real white plastic, and no HSV
threshold cleanly separated them there (3 attempts tried; see git history).
Depth would fix this properly (flat engraving vs. raised plastic) but these
exports have no matching depth frame. The newer captures don't have this
problem because their objects sit away from the engraving.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from printbot.config import DEFAULTS  # noqa: E402
from printbot.inspection.colour_rule import colour_fractions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = DEFAULTS["classifier"]["colour"]
OUT = ROOT / "datasets" / "red_white"
DEBUG_DIR = ROOT / "debug"
CLASSES = {"failed": 0, "good": 1}

MIN_OBJECT_PX = 1500
MAX_BBOX_PX = 70000      # rejects a bounding box bigger than any single real part
MIN_BOX_SIDE_PX = 40     # rejects thin edge/bezel slivers (one dimension tiny)
MIN_FILL_RATIO = 0.15   # rejects thin snaking lines: cv2 CC_STAT_AREA counts actual
                         # "on" pixels, not bbox area, so a line bridging the whole
                         # frame can have a huge bbox but tiny pixel count. Real
                         # spheres only hit ~0.23-0.27 here (shading means only
                         # the bright highlight passes the colour threshold, not
                         # the whole silhouette) vs. ~0.03-0.06 for the bridging
                         # blob - set well below the former, well above the latter.
OPEN_KERNEL_PX = 7

# Plate gold/tan HSV thresholds (PIL 0-255 HSV), calibrated against this
# camera/lighting in detect_parts.py.
PLATE_HUE_MAX = 32
PLATE_SAT_RANGE = (12, 150)
PLATE_VAL_RANGE = (50, 235)
PLATE_MIN_BLOB_PX = 5000

# --- Source 1: original 5-image export, hand-verified boxes -----------------
ORIGINAL_DIR = ROOT / "data"
ORIGINAL_BOXES = {
    "2026-11-06T17_14_57.026Z_2026-11-06T17_14_57.026Z+capV7Lwzx9v1BulHeueT7wlHhP4iZ7RgiwxYvsKIqgXo9eWuJYjyGLH0q1cMZ6Xk": [
        (472, 399, 612, 524),   # red cylinder
        (693, 407, 887, 599),   # red cuboid
    ],
    "2026-11-06T17_15_09.055Z_2026-11-06T17_15_09.055Z+MNIpXVmPVLgbjMAHAJ5dSp8nYYdhZZ5k1CJQgytZMLijCQWOrYR8Upw35FrFKKpc": [
        (585, 360, 740, 545),   # white pyramid, small red patch
    ],
    "2026-11-06T17_15_16.59Z_2026-11-06T17_15_16.59Z+WqbMHAvutYIH9nRBvu2n7AtvRrhZDMncEkWLSFWAsMG6A4MJtgmoGWUEazyyzVKU": [
        (565, 235, 665, 400),   # red cylinder
        (478, 505, 638, 692),   # red cuboid
        (740, 468, 895, 602),   # white pyramid, small red patch
    ],
    "2026-11-06T17_15_30.244Z_2026-11-06T17_15_30.244Z+J9YDF9iW37paDxb2vffFlLgCu7TI93zEmtxvF7TSeWjb3eTK3kkOljXVvcA3d4JI": [
        (561, 220, 734, 393),   # red sphere
        (729, 450, 903, 627),   # red sphere
    ],
    "2026-11-06T17_15_52.254Z_2026-11-06T17_15_52.254Z+MfZdsR9oF8VS5KMPuPD8BSpgJErW1WoNLff7PGxmaPveWNrTnqcJRv1XLtE20V9g": [
        (688, 276, 836, 422),   # white pyramid, small red patch
        (420, 437, 635, 615),   # roughly half red / half white part
    ],
}
# Held out for validation; token split - too few images for a real one.
ORIGINAL_VAL_SUBSTR = "WqbMHAvutYIH9nRBvu2n7AtvRrhZDMncEkWLSFWAsMG6A4MJtgmoGWUEazyyzVKU"

# --- Sources 2 & 3: auto-detected ------------------------------------------
GOOD_ONLY_DIR = ROOT / "raw_exports" / "good_only" / "data"
MIXED_DIR = ROOT / "raw_exports" / "mixed" / "data"

# --- Source 4: empty plate, pure negatives (zero object boxes) -------------
# These directly address the plate's engraved logo/text being mistaken for
# material - exactly the failure mode that broke automatic detection on
# source 1. No detection needed: every image here has zero real objects.
EMPTY_PLATE_DIR = ROOT / "raw_exports" / "empty_plate" / "data"


def largest_component_mask(mask: np.ndarray, min_px: int):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if areas[best - 1] < min_px:
        return None
    return labels == best


def plate_silhouette(rgb: np.ndarray) -> np.ndarray:
    hsv = np.array(Image.fromarray(rgb).convert("HSV"))
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    gold = (
        (hue <= PLATE_HUE_MAX)
        & (sat >= PLATE_SAT_RANGE[0]) & (sat <= PLATE_SAT_RANGE[1])
        & (val >= PLATE_VAL_RANGE[0]) & (val <= PLATE_VAL_RANGE[1])
    )
    plate = largest_component_mask(gold, PLATE_MIN_BLOB_PX)
    if plate is None:
        return np.ones(rgb.shape[:2], dtype=bool)
    dilated = cv2.dilate(plate.astype(np.uint8), np.ones((25, 25), np.uint8))
    return dilated.astype(bool)


def object_mask(hsv_pil: np.ndarray, cfg: dict) -> np.ndarray:
    h = hsv_pil[:, :, 0].astype(float) * 2.0
    s = hsv_pil[:, :, 1] / 255.0
    v = hsv_pil[:, :, 2] / 255.0
    red = (s >= cfg["red_s_min"]) & (v >= cfg["red_v_min"]) & ((h <= cfg["red_hue_max"]) | (h >= cfg["red_hue_min"]))
    white_mid = (s <= 0.18) & (v >= 0.65)
    return red | white_mid


def merge_overlapping_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Union any two boxes that substantially overlap - e.g. a part with a
    strong shading/colour split (red top, white base) can fragment into two
    separate components after the morphological opening; they're one object."""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax0, ay0, ax1, ay1 = boxes[i]
                bx0, by0, bx1, by1 = boxes[j]
                ix0, iy0 = max(ax0, bx0), max(ay0, by0)
                ix1, iy1 = min(ax1, bx1), min(ay1, by1)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                inter = (ix1 - ix0) * (iy1 - iy0)
                area_a = (ax1 - ax0) * (ay1 - ay0)
                area_b = (bx1 - bx0) * (by1 - by0)
                if inter / min(area_a, area_b) > 0.15:
                    boxes[i] = (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def detect_boxes(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    valid = plate_silhouette(rgb)
    hsv_pil = np.array(Image.fromarray(rgb).convert("HSV"))
    mask = object_mask(hsv_pil, CFG) & valid
    kernel = np.ones((OPEN_KERNEL_PX, OPEN_KERNEL_PX), np.uint8)
    mask_clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w_, h_ = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        bbox_area = w_ * h_
        if area < MIN_OBJECT_PX or bbox_area > MAX_BBOX_PX:
            continue
        if w_ < MIN_BOX_SIDE_PX or h_ < MIN_BOX_SIDE_PX:
            continue
        if area / bbox_area < MIN_FILL_RATIO:
            continue
        x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        boxes.append((x0, y0, x0 + w_ - 1, y0 + h_ - 1))
    return merge_overlapping_boxes(boxes)


def hsv_mask_for_box(bgr: np.ndarray, x0, y0, x1, y1, cfg: dict):
    sub_bgr = bgr[y0:y1 + 1, x0:x1 + 1]
    hsv = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(float) * 2.0
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    red = (s >= cfg["red_s_min"]) & (v >= cfg["red_v_min"]) & ((h <= cfg["red_hue_max"]) | (h >= cfg["red_hue_min"]))
    white = (s <= cfg["white_s_max"]) & (v >= cfg["white_v_min"])
    return sub_bgr, (red | white)


def classify_box(bgr: np.ndarray, x0, y0, x1, y1, cfg: dict):
    sub_bgr, mask = hsv_mask_for_box(bgr, x0, y0, x1, y1, cfg)
    red, white = colour_fractions(sub_bgr, mask, cfg)
    frac, dom = cfg["colour_frac"], cfg["colour_dominance"]
    if red >= frac and red >= dom * white:
        return "failed", red, white
    if white >= frac and white >= dom * red:
        return "good", red, white
    return ("failed" if red >= white else "good"), red, white


def write_example(fname, rgb, bgr, boxes_with_labels, split, report):
    Image.fromarray(rgb).save(OUT / "images" / split / f"{fname}.jpg", quality=95)
    h, w = rgb.shape[:2]
    lines = []
    entry = {"file": fname, "split": split, "objects": []}
    debug = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(debug)
    for (x0, y0, x1, y1, label, red_f, white_f) in boxes_with_labels:
        cls = CLASSES[label]
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        entry["objects"].append({"bbox": [x0, y0, x1, y1], "label": label,
                                  "red_frac": round(red_f, 3), "white_frac": round(white_f, 3)})
        colour = (255, 0, 0) if label == "failed" else (0, 128, 255)
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=4)
        draw.text((x0, max(0, y0 - 14)), f"{label} r{red_f:.2f}/w{white_f:.2f}", fill=colour)
    debug.save(DEBUG_DIR / f"{fname}.jpg")
    with open(OUT / "labels" / split / f"{fname}.txt", "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    report.append(entry)
    print(f"{fname} [{split}]: {len(boxes_with_labels)} object(s)")
    for o in entry["objects"]:
        print(f"   {o['label']:6s} red={o['red_frac']:.2f} white={o['white_frac']:.2f} bbox={o['bbox']}")


def main():
    for split in ("train", "val"):
        os.makedirs(OUT / "images" / split, exist_ok=True)
        os.makedirs(OUT / "labels" / split, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    report = []

    # Source 1: hand-verified boxes, labelled via official colour_rule.
    for fname, boxes in ORIGINAL_BOXES.items():
        rgb = np.array(Image.open(ORIGINAL_DIR / fname).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        split = "val" if ORIGINAL_VAL_SUBSTR in fname else "train"
        labelled = []
        for (x0, y0, x1, y1) in boxes:
            label, red_f, white_f = classify_box(bgr, x0, y0, x1, y1, CFG)
            labelled.append((x0, y0, x1, y1, label, red_f, white_f))
        write_example(fname, rgb, bgr, labelled, split, report)

    # Source 2: good-only capture - auto-detect boxes, force label "good".
    # Ground truth is the dataset's own identity, not a colour guess; the
    # red/white fractions are still computed and printed for a sanity check
    # (a high red_frac here would mean a detection bug, not a real failure).
    if GOOD_ONLY_DIR.is_dir():
        files = sorted(p.name for p in GOOD_ONLY_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
        for i, fname in enumerate(files):
            rgb = np.array(Image.open(GOOD_ONLY_DIR / fname).convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            boxes = detect_boxes(rgb)
            labelled = []
            for (x0, y0, x1, y1) in boxes:
                _, red_f, white_f = classify_box(bgr, x0, y0, x1, y1, CFG)
                if red_f > 0.35:
                    print(f"   WARNING: {fname} box {(x0, y0, x1, y1)} has red_frac={red_f:.2f} "
                          f"in a 'good-only' capture - check this box by eye before trusting it.")
                labelled.append((x0, y0, x1, y1, "good", red_f, white_f))
            split = "val" if i == 0 else "train"  # token split, see README
            write_example(f"good_{fname}", rgb, bgr, labelled, split, report)

    # Source 3: mixed capture - auto-detect boxes, label via colour_rule.
    if MIXED_DIR.is_dir():
        files = sorted(p.name for p in MIXED_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
        for i, fname in enumerate(files):
            rgb = np.array(Image.open(MIXED_DIR / fname).convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            boxes = detect_boxes(rgb)
            labelled = []
            for (x0, y0, x1, y1) in boxes:
                label, red_f, white_f = classify_box(bgr, x0, y0, x1, y1, CFG)
                labelled.append((x0, y0, x1, y1, label, red_f, white_f))
            split = "val" if i == 0 else "train"  # token split, see README
            write_example(f"mixed_{fname}", rgb, bgr, labelled, split, report)

    # Source 4: empty plate - pure negatives, zero boxes, no detection needed.
    if EMPTY_PLATE_DIR.is_dir():
        files = sorted(p.name for p in EMPTY_PLATE_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
        for i, fname in enumerate(files):
            rgb = np.array(Image.open(EMPTY_PLATE_DIR / fname).convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            split = "val" if i == 0 else "train"  # token split, see README
            write_example(f"empty_{fname}", rgb, bgr, [], split, report)

    with open(OUT / "bootstrap_report.json", "w") as f:
        json.dump(report, f, indent=2)

    n_train = sum(1 for e in report if e["split"] == "train")
    n_val = sum(1 for e in report if e["split"] == "val")
    n_obj = sum(len(e["objects"]) for e in report)
    with open(OUT / "data.yaml", "w") as f:
        f.write(
            "# Bootstrap dataset from 3 Viam exports: red filament = failed print demo\n"
            "# marker, white filament = good print demo marker. See README.md for\n"
            "# provenance per source and limitations.\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: failed\n"
            "  1: good\n"
        )
    print(f"\n{n_train} train / {n_val} val images, {n_obj} labelled objects total.")
    print(f"Wrote {OUT}/data.yaml, {OUT}/bootstrap_report.json, and debug/*.jpg (visual check)")


if __name__ == "__main__":
    main()
