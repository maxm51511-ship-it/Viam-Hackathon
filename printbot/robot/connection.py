"""Viam machine connection from environment variables (never from source).

Required env vars (see ``.env.example``):
    VIAM_MACHINE_ADDRESS   e.g. my-machine-main.xxxx.viam.cloud
    VIAM_API_KEY
    VIAM_API_KEY_ID
"""

from __future__ import annotations

import os

from printbot.config import load_dotenv


class ConfigError(RuntimeError):
    pass


async def connect():
    """Return a connected ``RobotClient``. Caller must ``await robot.close()``."""
    from viam.robot.client import RobotClient

    # Register every resource API we use BEFORE connecting: RobotClient only
    # creates handles at connect time for APIs already imported, otherwise
    # e.g. Motion.from_robot(robot, "builtin") raises ResourceNotFoundError.
    import viam.components.arm  # noqa: F401
    import viam.components.camera  # noqa: F401
    import viam.components.gripper  # noqa: F401
    import viam.services.motion  # noqa: F401
    import viam.services.vision  # noqa: F401

    load_dotenv()
    missing = [k for k in ("VIAM_MACHINE_ADDRESS", "VIAM_API_KEY", "VIAM_API_KEY_ID") if not os.environ.get(k)]
    if missing:
        raise ConfigError(f"missing environment variables: {', '.join(missing)} (copy .env.example to .env)")
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)
