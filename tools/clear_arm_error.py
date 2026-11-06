"""Sends arm.do_command({"clear_error": True}) to reset a latched servo fault,
then re-checks state. NOT purely read-only - confirm the arm/gripper are
physically clear of any obstruction before running this; releasing a latched
fault re-enables the servos, and if something is still in contact/under
tension that can cause a sudden jerk.

    python tools/clear_arm_error.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from printbot.robot.connection import connect  # noqa: E402
from viam.components.arm import Arm


async def main():
    robot = await connect()
    try:
        arm = Arm.from_robot(robot, "arm")
        try:
            resp = await arm.do_command({"clear_error": True})
            print("clear_error response:", resp)
        except Exception as e:
            print("clear_error call failed:", e)
            return

        # Re-check that normal operation is restored.
        try:
            state = await arm.do_command({"get_state": True})
            print("get_state after clear:", state)
        except Exception as e:
            print("get_state after clear failed:", e)

        try:
            pos = await arm.get_end_position()
            print(f"end position: x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f}")
        except Exception as e:
            print("get_end_position failed:", e)

        try:
            moving = await arm.is_moving()
            print("is_moving:", moving)
        except Exception as e:
            print("is_moving failed:", e)
    finally:
        await robot.close()


asyncio.run(main())
