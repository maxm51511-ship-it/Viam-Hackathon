"""Shared data contracts.

These types are the ONLY coupling between perception, inspection, robot control
and the orchestrator. They are plain dataclasses with NumPy payloads and carry
no Viam or hardware dependencies, so every module can be unit-tested offline.

Coordinate conventions
----------------------
* Pixel coordinates: (u, v) = (column, row), origin top-left.
* Camera frame: RealSense optical convention, +X right, +Y down, +Z forward
  (out of the lens). Units are millimetres.
* World/robot frame: whatever the Viam frame system defines as "world". Only
  ``printbot.robot.transforms`` converts camera -> world; nothing else should.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics of the colour-aligned depth image (pixels)."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass
class Frame:
    """One synchronized capture from the camera.

    ``depth_mm`` must be aligned to ``bgr`` (same resolution, same pixel grid).
    Zero depth means "no reading" (RealSense convention).
    """

    bgr: np.ndarray                      # HxWx3 uint8, OpenCV BGR order
    depth_mm: Optional[np.ndarray]       # HxW uint16/float, millimetres, 0 = invalid
    intrinsics: Optional[Intrinsics]
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"

    @property
    def has_depth(self) -> bool:
        return self.depth_mm is not None and self.intrinsics is not None


@dataclass(frozen=True)
class Plane:
    """Plane n·p + d = 0 in camera frame (mm); ``normal`` is unit length and
    oriented toward the camera so that positive signed distance = "above plate"."""

    normal: np.ndarray
    d: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.d


@dataclass
class PlateSegmentation:
    plate_mask: np.ndarray               # HxW bool — pixels belonging to exposed/occluded plate region
    plate_polygon: Optional[np.ndarray]  # Nx2 int32 pixel polygon of plate boundary
    object_mask: np.ndarray              # HxW bool — material on top of the plate
    height_mm: Optional[np.ndarray]      # HxW float — height above plate plane (NaN where unknown)
    plane: Optional[Plane]
    used_depth: bool
    notes: list[str] = field(default_factory=list)


class MaterialClass(str, enum.Enum):
    """Label vocabulary shared by heuristics, YOLO dataset and classifier."""

    BUILD_PLATE = "build_plate"
    PRINTED_OBJECT = "printed_object"
    SPAGHETTI = "spaghetti"
    LOOSE_FILAMENT = "loose_filament"
    OTHER_DEFECT = "other_defect"


@dataclass
class Detection2D:
    """A detected region in image space, from YOLO or from heuristics."""

    label: MaterialClass
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None    # HxW bool
    source: str = "heuristic"            # "heuristic" | "yolo"


@dataclass
class ObjectEstimate:
    """A connected blob of material on the plate with its 3D description.

    All 3D quantities are in the CAMERA frame (mm). ``grasp_*`` is a candidate,
    not a validated grasp — validation happens in the grasp planner.
    """

    mask: np.ndarray
    pixel_area: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    min_rect: tuple[tuple[float, float], tuple[float, float], float]  # cv2.minAreaRect output
    max_height_mm: float = float("nan")
    footprint_mm: tuple[float, float] = (float("nan"), float("nan"))  # (long, short) side
    yaw_deg: float = float("nan")         # long-axis angle in image plane
    top_center_cam_mm: Optional[np.ndarray] = None  # 3-vector, top surface centre
    # shape descriptors used by the heuristic classifier
    solidity: float = float("nan")        # area / convex-hull area
    extent: float = float("nan")          # area / min-rect area
    height_cv: float = float("nan")       # coeff. of variation of height inside mask
    n_fragments: int = 1


class Verdict(str, enum.Enum):
    EMPTY = "empty"                # nothing on the plate
    SUCCESS = "success"
    FAILED = "failed"
    UNCERTAIN = "uncertain"        # -> human inspection, never auto-manipulate


@dataclass
class Inspection:
    """Result of inspecting ONE frame."""

    verdict: Verdict
    confidence: float
    reasons: list[str]
    objects: list[ObjectEstimate]
    detections: list[Detection2D]
    timestamp: float = field(default_factory=time.time)


@dataclass
class Assessment:
    """Final decision aggregated over several inspections.

    ``needs_human`` is True whenever automatic manipulation is not justified.
    """

    verdict: Verdict
    confidence: float
    reasons: list[str]
    n_observations: int
    needs_human: bool
    primary_object: Optional[ObjectEstimate] = None
