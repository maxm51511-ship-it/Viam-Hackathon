"""READ-ONLY arm diagnostics: error state, controller state, end position,
is_moving. Sends no motion or state-changing commands.

    python tools/diag_arm.py
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
            err = await arm.do_command({"get_error": True})
            print("get_error:", err)
        except Exception as e:
            print("get_error call itself failed:", e)
        try:
            state = await arm.do_command({"get_state": True})
            print("get_state:", state)
        except Exception as e:
            print("get_state call itself failed:", e)
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
