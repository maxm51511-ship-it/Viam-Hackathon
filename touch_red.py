"""Approved motion: move the gripper to touch the red cuboid, two slow steps.

Step 1: hover 100mm above the part (gripper pointing straight down).
Step 2: descend to the part's top surface. No grasp, no further movement.

Re-detects immediately before moving (the bed is live / changing), matches
the red cuboid by proximity to the approved target, and ABORTS instead of
moving if nothing matches closely enough - it will not guess at a different
part. Obstacles already in the frame system (table, walls, ceiling) are
applied automatically by the "builtin" motion service.
"""

import asyncio
import math

import detect_parts as dp
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

# Approved in chat for the red cuboid (bottom-left, on its white stand).
APPROVED_TARGET = (139.2, -525.9, 84.4)
MATCH_TOLERANCE_MM = 120.0
HOVER_CLEARANCE_MM = 100.0

MOTION_SERVICE_NAME = "builtin"
GRIPPER_NAME = "gripper"


async def gripper_world_position(robot):
    p = PoseInFrame(reference_frame=GRIPPER_NAME, pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0))
    wp = (await robot.transform_pose(p, "world")).pose
    return wp.x, wp.y, wp.z


async def main():
    robot = await dp.connect()
    try:
        scene = await dp.detect(robot)
        candidates = []
        for d in scene["red_detections"]:
            cx = int((d.x_min + d.x_max) / 2)
            cy = int((d.y_min + d.y_max) / 2)
            depth_mm = dp.median_depth(scene["depth_arr"], cx, cy)
            if depth_mm is None:
                continue
            px, py, pz = dp.pixel_to_camera_point(cx, cy, depth_mm, scene["intrinsics"])
            cam_pose = PoseInFrame(reference_frame="cam", pose=Pose(x=px, y=py, z=pz, o_x=0, o_y=0, o_z=1, theta=0))
            wp = (await robot.transform_pose(cam_pose, "world")).pose
            candidates.append(wp)

        def dist(wp):
            return math.dist((wp.x, wp.y, wp.z), APPROVED_TARGET)

        if not candidates:
            print("ABORT: no red detections with valid depth right now. Nothing sent to the arm.")
            return

        best = min(candidates, key=dist)
        d = dist(best)
        print(f"Re-detected candidate closest to approved target: "
              f"x={best.x:.1f} y={best.y:.1f} z={best.z:.1f}  (off by {d:.1f}mm)")
        if d > MATCH_TOLERANCE_MM:
            print(f"ABORT: closest match is {d:.1f}mm from the approved target "
                  f"(tolerance {MATCH_TOLERANCE_MM}mm). Scene has likely changed too much "
                  f"since approval - nothing sent to the arm. Re-run detection and re-confirm.")
            return

        touch = Pose(x=best.x, y=best.y, z=best.z, o_x=0.04, o_y=0.02, o_z=-1.0, theta=23.7)
        hover = Pose(x=best.x, y=best.y, z=best.z + HOVER_CLEARANCE_MM, o_x=0.04, o_y=0.02, o_z=-1.0, theta=23.7)

        motion = MotionClient.from_robot(robot, MOTION_SERVICE_NAME)

        print(f"\nStep 1/2: hover at x={hover.x:.1f} y={hover.y:.1f} z={hover.z:.1f}")
        ok = await motion.move(
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame="world", pose=hover),
        )
        gx, gy, gz = await gripper_world_position(robot)
        print(f"  move() returned success={ok}; gripper now at x={gx:.1f} y={gy:.1f} z={gz:.1f}")
        if not ok:
            print("ABORT: hover move reported failure, not attempting the descent.")
            return

        print(f"\nStep 2/2: descend to touch x={touch.x:.1f} y={touch.y:.1f} z={touch.z:.1f}")
        ok = await motion.move(
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame="world", pose=touch),
        )
        gx, gy, gz = await gripper_world_position(robot)
        print(f"  move() returned success={ok}; gripper now at x={gx:.1f} y={gy:.1f} z={gz:.1f}")
        print("\nDone. No grasp attempted, no further movement.")
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
