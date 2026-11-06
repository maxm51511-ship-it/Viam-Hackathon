"""Run one inspect -> classify -> (approve) -> pick/place cycle.

DRY-RUN BY DEFAULT: motion commands are logged, not sent. Live motion needs
ALL of: --allow-motion flag, PRINTBOT_ALLOW_MOTION=1, and
robot.motion_enabled=true in config/settings.json. Only enable after the
motion has been validated against a simulated arm and the E-stop is in reach.

    python tools/run_cycle.py                       # live camera, dry-run robot
    python tools/run_cycle.py --frames data/captures/s1   # offline replay, no machine needed
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from printbot.app.approval import ConsoleApprover  # noqa: E402
from printbot.app.events import EventLog  # noqa: E402
from printbot.app.state_machine import Orchestrator  # noqa: E402
from printbot.config import load_settings  # noqa: E402
from printbot.inspection.classifier import PrintClassifier  # noqa: E402
from printbot.inspection.reference import expected_from_gcode, expected_from_stl  # noqa: E402
from printbot.perception.sources import FileFrameSource, ViamCameraSource  # noqa: E402
from printbot.robot.controller import RobotController  # noqa: E402
from printbot.robot.transforms import ViamFrameTransformer  # noqa: E402


async def main(a) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = load_settings()
    ev = EventLog(s["logging"]["event_log"])
    expected = expected_from_stl(a.stl) if a.stl else expected_from_gcode(a.gcode) if a.gcode else None
    yolo = None
    if a.yolo:
        from printbot.perception.yolo import YoloSegmenter
        yolo = YoloSegmenter(s["yolo"])

    robot = None
    if a.frames:
        frames = FileFrameSource(a.frames)
        from printbot.robot.transforms import FixedTransformer
        import numpy as np
        transformer = FixedTransformer(np.eye(4))  # offline: world == camera (planning output not meaningful)
        ctl = RobotController(None, s, allow_motion=False, events=ev)
    else:
        from printbot.robot.connection import connect
        robot = await connect()
        frames = ViamCameraSource(robot, s["resources"]["camera"], s["camera"]["color_source_name"],
                                  s["camera"]["depth_source_name"])
        transformer = ViamFrameTransformer(robot, s["resources"]["camera"])
        ctl = RobotController(robot, s, allow_motion=a.allow_motion, events=ev)
    try:
        orch = Orchestrator(s, frames, PrintClassifier(s, yolo=yolo, expected=expected), transformer, ctl,
                            ConsoleApprover(), ev)
        r = await orch.run_cycle()
        print(f"\nRESULT: {r.final_state.value} - {r.message}")
        print(" -> ".join(st.value for st in r.history))
        return 0 if r.final_state.value in ("COMPLETED", "HUMAN_REVIEW", "ABORTED") else 1
    finally:
        if robot is not None:
            await robot.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", help="offline replay directory instead of live camera")
    p.add_argument("--stl")
    p.add_argument("--gcode")
    p.add_argument("--yolo", action="store_true")
    p.add_argument("--allow-motion", action="store_true")
    raise SystemExit(asyncio.run(main(p.parse_args())))
