"""READ-ONLY machine check: lists resources, frame system, camera properties
and one image capture. Sends no motion commands.

    python tools/check_machine.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from printbot.config import load_settings  # noqa: E402
from printbot.robot.connection import connect  # noqa: E402


async def main() -> int:
    s = load_settings()
    robot = await connect()
    try:
        names = robot.resource_names
        print("== resources")
        for n in sorted(names, key=lambda r: (r.subtype, r.name)):
            print(f"  {n.namespace}:{n.type}:{n.subtype}  {n.name}")
        present = {n.name for n in names}
        for role, name in s["resources"].items():
            if name and name not in present:
                print(f"  !! configured {role} '{name}' NOT on machine")

        print("== frame system")
        for fc in await robot.get_frame_system_config():
            f = fc.frame
            print(f"  {f.reference_frame:<20} parent={f.pose_in_observer_frame.reference_frame:<12} "
                  f"pose={f.pose_in_observer_frame.pose}".replace("\n", " "))

        cam_name = s["resources"]["camera"]
        if cam_name in present:
            from viam.components.camera import Camera

            cam = Camera.from_robot(robot, cam_name)
            props = await cam.get_properties()
            print("== camera properties\n", props)
            images, _ = await cam.get_images()
            print("== camera images")
            for im in images:
                print(f"  name={getattr(im, 'name', '?')} mime={im.mime_type} {im.width}x{im.height}")
    finally:
        await robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
