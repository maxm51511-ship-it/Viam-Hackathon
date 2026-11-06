"""Top-down parallel-jaw grasp planning (MVP).

Given an object's top-surface centre and long-axis direction in the WORLD
frame, propose a top-down grasp across the object's short side and check it
against gripper and geometry constraints. Every rejection carries a reason so
the orchestrator can hand off to a human instead of guessing.

Assumptions (must be verified on hardware — see docs/PLAN.md):
* world +Z points up (away from the build plate);
* the gripper frame's +Z points out of the fingers, so "pointing down" is the
  Viam orientation vector (0, 0, -1), with theta rotating about it;
* the object's top centre approximates a reasonable grasp centre. This is NOT
  the centre of mass and ignores fragile features — structural checks
  (Part Three B) are a later refinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class GraspPlan:
    feasible: bool
    reasons: list[str] = field(default_factory=list)
    pre_grasp: Optional[dict] = None   # pose dict in world frame (mm / deg)
    grasp: Optional[dict] = None
    lift: Optional[dict] = None
    jaw_width_mm: float = float("nan")


def _pose(p: np.ndarray, theta_deg: float) -> dict:
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
            "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": float(theta_deg)}


def plan_top_down_grasp(
    top_center_world: np.ndarray,
    long_axis_world: np.ndarray,
    footprint_mm: tuple[float, float],
    height_mm: float,
    gripper: dict[str, Any],
    approach_offset_mm: float = 80.0,
) -> GraspPlan:
    reasons: list[str] = []
    max_open = gripper.get("max_opening_mm")
    finger_len = gripper.get("finger_length_mm")
    clearance = float(gripper.get("jaw_clearance_mm", 10.0))
    plate_margin = float(gripper.get("plate_margin_mm", 3.0))
    if max_open is None or finger_len is None:
        reasons.append("gripper spec unknown: set robot.gripper.max_opening_mm and finger_length_mm")
    if not all(math.isfinite(v) for v in (*footprint_mm, height_mm)):
        reasons.append("object has no metric size (no depth)")
    if reasons:
        return GraspPlan(False, reasons)

    short = footprint_mm[1]
    width = short + clearance
    if width > max_open:
        reasons.append(f"object short side {short:.1f} mm + clearance exceeds gripper opening {max_open} mm")
    if height_mm < plate_margin + 2.0:
        reasons.append(f"object too flat ({height_mm:.1f} mm) for a top-down pinch above the plate")

    # Grasp as deep as fingers allow but stay plate_margin above the plate.
    depth = min(float(finger_len) * 0.8, height_mm - plate_margin)
    grasp_pt = np.asarray(top_center_world, dtype=np.float64) - np.array([0, 0, max(depth, 0.0)])

    # Close the jaws across the short side: jaw axis ⟂ long axis.
    ax = np.asarray(long_axis_world, dtype=np.float64)
    yaw = math.degrees(math.atan2(ax[1], ax[0])) + 90.0
    yaw = (yaw + 180.0) % 360.0 - 180.0

    pre = grasp_pt + np.array([0, 0, depth + approach_offset_mm])
    lift = grasp_pt + np.array([0, 0, approach_offset_mm + height_mm])
    if reasons:
        return GraspPlan(False, reasons, jaw_width_mm=width)
    return GraspPlan(True, [f"top-down grasp, jaw width {width:.1f} mm, depth {depth:.1f} mm, yaw {yaw:.1f} deg"],
                     _pose(pre, yaw), _pose(grasp_pt, yaw), _pose(lift, yaw), width)
