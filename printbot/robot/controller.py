"""Collision-aware robot control wrapper (Viam motion service).

SAFETY MODEL
------------
Physical motion requires ALL of:
  1. ``settings.robot.motion_enabled == true``
  2. environment variable ``PRINTBOT_ALLOW_MOTION=1``
  3. ``RobotController(..., allow_motion=True)``
Otherwise every motion command is logged as DRY-RUN and not sent.

* Moves go through ``motion.move`` with the GRIPPER as the component (per the
  UFactory/Viam guidance) and a WorldState built from configured obstacles.
  Direct ``arm.move_to_position`` is deliberately not exposed: it ignores
  obstacles.
* Any exception during a motion command triggers ``stop_all()``.
* A software stop is NOT an emergency stop. The physical E-stop must be within
  reach during every hardware test.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)


class MotionRefused(RuntimeError):
    pass


class RobotController:
    def __init__(self, robot, settings: dict[str, Any], allow_motion: bool = False, events=None):
        self.robot = robot
        self.res = settings["resources"]
        self.rcfg = settings["robot"]
        self.events = events
        self.live = bool(allow_motion and self.rcfg.get("motion_enabled")
                         and os.environ.get("PRINTBOT_ALLOW_MOTION") == "1")
        self._motion = self._gripper = None
        if robot is not None:
            from viam.components.gripper import Gripper
            from viam.services.motion import Motion

            self._motion = Motion.from_robot(robot, self.res["motion"])
            self._gripper = Gripper.from_robot(robot, self.res["gripper"])
        log.info("RobotController live=%s", self.live)

    # ------------------------------------------------------------ helpers
    def _log(self, kind: str, **data):
        if self.events:
            self.events.record(kind, dry_run=not self.live, **data)
        log.info("%s%s %s", "" if self.live else "[DRY-RUN] ", kind, data)

    def world_state(self):
        from viam.proto.common import (Geometry, GeometriesInFrame, Pose, RectangularPrism, Vector3,
                                       WorldState)

        geoms = []
        for ob in self.rcfg.get("obstacles", []):
            c, d = ob["center"], ob["dims_mm"]
            geoms.append(Geometry(center=Pose(x=c["x"], y=c["y"], z=c["z"], o_z=1, theta=0),
                                  box=RectangularPrism(dims_mm=Vector3(x=d["x"], y=d["y"], z=d["z"])),
                                  label=ob.get("label", "obstacle")))
        return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=geoms)]) if geoms else None

    def named_pose(self, name: str) -> dict:
        p = self.rcfg.get("named_poses", {}).get(name)
        if p is None:
            raise MotionRefused(f"named pose '{name}' not configured in settings.robot.named_poses")
        return p

    # ------------------------------------------------------------ motion
    async def move_gripper_to(self, pose: dict, frame: str = "world", label: str = "") -> bool:
        self._log("move", label=label, frame=frame, pose=pose)
        if not self.live:
            return True
        from viam.proto.common import Pose, PoseInFrame

        dest = PoseInFrame(reference_frame=frame, pose=Pose(**pose))
        try:
            return await self._motion.move(component_name=self.res["gripper"], destination=dest,
                                           world_state=self.world_state())
        except Exception:
            await self.stop()
            raise

    async def go_named(self, name: str) -> bool:
        return await self.move_gripper_to(self.named_pose(name), label=name)

    async def open_gripper(self) -> None:
        self._log("gripper_open")
        if self.live:
            await self._gripper.open()

    async def grab(self) -> bool:
        self._log("gripper_grab")
        if not self.live:
            return True
        return await self._gripper.grab()

    async def is_holding(self) -> Optional[bool]:
        if not self.live:
            return None
        status = await self._gripper.is_holding_something()
        return bool(getattr(status, "is_holding_something", False))

    async def stop(self) -> None:
        self._log("stop_all")
        if self.robot is not None:
            try:
                await self.robot.stop_all()
            except Exception as e:  # report, never mask the original failure
                log.error("stop_all failed: %s", e)
