"""Camera-frame -> world-frame conversion.

This is the ONLY place perception output crosses into robot coordinates. It
relies on the camera's ``frame`` (parent + translation + orientation) being
configured in the Viam machine config; without calibrated extrinsics the
result is wrong, not just imprecise.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PointTransformer(Protocol):
    async def to_world(self, xyz_cam_mm: np.ndarray) -> np.ndarray: ...


class ViamFrameTransformer:
    def __init__(self, robot, camera_frame: str, world_frame: str = "world"):
        self.robot = robot
        self.camera_frame = camera_frame
        self.world_frame = world_frame

    async def to_world(self, xyz_cam_mm: np.ndarray) -> np.ndarray:
        from viam.proto.common import Pose, PoseInFrame

        x, y, z = (float(v) for v in xyz_cam_mm)
        q = PoseInFrame(reference_frame=self.camera_frame, pose=Pose(x=x, y=y, z=z, o_z=1, theta=0))
        res = await self.robot.transform_pose(q, self.world_frame)
        return np.array([res.pose.x, res.pose.y, res.pose.z])


class FixedTransformer:
    """Rigid transform from a 4x4 matrix (tests, offline replay, calibration checks)."""

    def __init__(self, T_world_cam: np.ndarray):
        self.T = np.asarray(T_world_cam, dtype=np.float64)

    async def to_world(self, xyz_cam_mm: np.ndarray) -> np.ndarray:
        p = np.append(np.asarray(xyz_cam_mm, dtype=np.float64), 1.0)
        return (self.T @ p)[:3]
