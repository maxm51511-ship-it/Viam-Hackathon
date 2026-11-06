"""Settings loading.

Settings live in ``config/settings.json`` (tunable thresholds, resource names,
named poses). Secrets never go there — they come from environment variables
(see ``.env.example``). Missing keys fall back to ``DEFAULTS`` so a partial
settings file is valid.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.json"

DEFAULTS: dict[str, Any] = {
    "resources": {
        # Names from the hackathon machine config (Viam overview PDF): arm, gripper, cam.
        "arm": "arm",
        "gripper": "gripper",
        "camera": "cam",
        "motion": "builtin",
        "vision": None,          # optional Viam vision service name
    },
    "camera": {
        "color_source_name": None,   # None = pick first non-depth image
        "depth_source_name": None,   # None = pick first VIAM_RAW_DEPTH image
        "pcd_units": "auto",         # "mm" | "m" | "auto" (unverified upstream; see docs)
    },
    "plate": {
        # HSV thresholds for the exposed build plate (OpenCV ranges H 0-179, S/V 0-255).
        # A list of ranges lets one profile cover e.g. textured PEI with highlights.
        "hsv_ranges": [{"lower": [0, 0, 0], "upper": [179, 80, 90]}],
        "roi_polygon_px": None,      # optional fixed polygon [[u,v],...] restricting the search
        "morph_kernel_px": 7,
        "min_plate_area_frac": 0.05,
        "object_min_height_mm": 1.5,  # height above plane that counts as material
        "min_speck_px": 20,           # drop smaller connected blobs (depth noise)
        "ransac_iterations": 200,
        "ransac_inlier_mm": 1.5,
    },
    "objects": {
        "min_area_px": 150,
        "merge_dilate_px": 5,
    },
    "classifier": {
        # "colour_demo": red material = failed, white = good (hackathon demo)
        # "geometry":    shape heuristics (+ YOLO / STL reference when available)
        "mode": "colour_demo",
        "colour": {"red_hue_max": 20, "red_hue_min": 330, "red_s_min": 0.50, "red_v_min": 0.20,
                   "white_s_max": 0.25, "white_v_min": 0.60, "colour_frac": 0.35, "colour_dominance": 2.0},
        "spaghetti_max_solidity": 0.55,
        "spaghetti_min_fragments": 4,
        "success_min_solidity": 0.80,
        "success_min_height_mm": 3.0,
        "yolo_min_confidence": 0.5,
        "min_observations": 3,
        "agreement_frac": 0.8,
        "min_confidence_to_act": 0.7,
    },
    "yolo": {
        "weights": "models/print_defects-seg.pt",
        "imgsz": 640,
        "confidence": 0.35,
        "iou": 0.5,
        "device": None,
    },
    "robot": {
        # Motion is OFF unless explicitly enabled at runtime AND via env var.
        "motion_enabled": False,
        "named_poses": {},           # e.g. {"home": {...}, "success_bin": {...}} in world frame, mm/deg
        "obstacles": [],             # [{"label","center":{x,y,z},"dims_mm":{x,y,z}}] in world frame
        "approach_offset_mm": 80.0,
        # UFactory gripper geometry — UNKNOWN until measured; grasp planning refuses while null.
        "gripper": {"max_opening_mm": None, "finger_length_mm": None,
                    "jaw_clearance_mm": 10.0, "plate_margin_mm": 3.0},
        # How the print is released from the plate before pickup:
        #   "operator" = human confirms the part is loose (only option until a mechanism is validated)
        "release_mode": "operator",
        # Failed prints: "human" = always hand off; "grasp" = try gripper pickup to failed bin.
        # Sweeping is not implemented (needs a validated compliant end effector).
        "failed_removal_mode": "human",
    },
    "logging": {
        "event_log": "logs/events.jsonl",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load settings JSON merged over DEFAULTS. A missing file yields DEFAULTS."""
    p = Path(path) if path else Path(os.environ.get("PRINTBOT_SETTINGS", DEFAULT_SETTINGS_PATH))
    if not p.exists():
        return copy.deepcopy(DEFAULTS)
    with p.open("r", encoding="utf-8") as f:
        user = json.load(f)
    return _deep_merge(DEFAULTS, user)


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Minimal ``.env`` loader (KEY=VALUE lines) — avoids a python-dotenv dependency.
    Existing environment variables take precedence."""
    p = Path(path) if path else REPO_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
