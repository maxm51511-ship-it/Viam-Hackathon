"""Capture RGB-D frames from the Viam camera for testing and YOLO dataset building.

    python tools/capture_frames.py --n 20 --interval 1.0 --out data/captures/session1 --tag spaghetti

Saves <stem>_color.png, <stem>_depth.png (uint16 mm), <stem>_meta.json.
Read-only: no motion.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from printbot.config import load_settings  # noqa: E402
from printbot.perception.sources import ViamCameraSource, save_frame  # noqa: E402
from printbot.robot.connection import connect  # noqa: E402


async def main(a) -> None:
    s = load_settings()
    robot = await connect()
    try:
        src = ViamCameraSource(robot, s["resources"]["camera"],
                               s["camera"]["color_source_name"], s["camera"]["depth_source_name"])
        out = Path(a.out)
        for i in range(a.n):
            f = await src.get_frame()
            stem = f"{a.tag}_{time.strftime('%Y%m%d_%H%M%S')}_{i:03d}"
            save_frame(f, out, stem)
            print(f"saved {stem} depth={'yes' if f.depth_mm is not None else 'NO'}")
            await asyncio.sleep(a.interval)
    finally:
        await robot.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--out", default="data/captures")
    p.add_argument("--tag", default="frame")
    asyncio.run(main(p.parse_args()))
