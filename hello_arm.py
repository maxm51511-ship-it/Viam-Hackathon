import asyncio
import os

from dotenv import load_dotenv
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.robot.client import RobotClient

load_dotenv()

ARM_NAME = "arm"
CAMERA_NAME = "cam"


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    return await RobotClient.at_address(os.environ["VIAM_ADDRESS"], opts)


async def main():
    robot = await connect()
    try:
        print("Connected. Resources:")
        for r in robot.resource_names:
            print("  ", r.subtype, r.name)

        arm = Arm.from_robot(robot, ARM_NAME)
        pose = await arm.get_end_position()
        print(f"Arm end position: x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f}")

        cam = Camera.from_robot(robot, CAMERA_NAME)
        images, meta = await cam.get_images()
        for img in images:
            mime = getattr(img, "mime_type", "") or ""
            ext = "png" if "png" in mime else "jpg"
            filename = f"camera_test_{img.name}.{ext}"
            with open(filename, "wb") as f:
                f.write(img.data)
            print(f"Saved {filename}")
    finally:
        await robot.close()


asyncio.run(main())
