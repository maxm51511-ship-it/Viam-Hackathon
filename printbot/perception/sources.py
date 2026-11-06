"""Frame acquisition.

* ``ViamCameraSource`` — live RealSense (or any Viam camera) via ``Camera.get_images``.
* ``FileFrameSource``  — replays frames saved by ``tools/capture_frames.py``
  (``<stem>_color.png`` + ``<stem>_depth.png`` (uint16 mm) + ``<stem>_meta.json``),
  so perception can be developed and regression-tested without hardware.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

from printbot.types import Frame, Intrinsics

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    async def get_frame(self) -> Frame: ...


# --------------------------------------------------------------------- files

def save_frame(frame: Frame, directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / f"{stem}_color.png"), frame.bgr)
    meta = {"timestamp": frame.timestamp, "source": frame.source, "intrinsics": None}
    if frame.depth_mm is not None:
        cv2.imwrite(str(directory / f"{stem}_depth.png"), frame.depth_mm.astype(np.uint16))
    if frame.intrinsics is not None:
        meta["intrinsics"] = frame.intrinsics.__dict__
    (directory / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
    return directory / f"{stem}_color.png"


def load_frame(color_path: Path) -> Frame:
    color_path = Path(color_path)
    stem = color_path.name.removesuffix("_color.png")
    bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read colour image {color_path}")
    dpath = color_path.with_name(f"{stem}_depth.png")
    depth = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED) if dpath.exists() else None
    mpath = color_path.with_name(f"{stem}_meta.json")
    intr, ts = None, time.time()
    if mpath.exists():
        meta = json.loads(mpath.read_text())
        ts = meta.get("timestamp", ts)
        if meta.get("intrinsics"):
            intr = Intrinsics(**meta["intrinsics"])
    return Frame(bgr=bgr, depth_mm=depth, intrinsics=intr, timestamp=ts, source=f"file:{color_path.name}")


class FileFrameSource:
    """Cycles through ``*_color.png`` files in a directory."""

    def __init__(self, directory: str | Path):
        self.paths = sorted(Path(directory).glob("*_color.png"))
        if not self.paths:
            raise FileNotFoundError(f"no *_color.png frames in {directory}")
        self._i = 0

    async def get_frame(self) -> Frame:
        p = self.paths[self._i % len(self.paths)]
        self._i += 1
        return load_frame(p)


# ---------------------------------------------------------------------- viam

class ViamCameraSource:
    """Live frames from a Viam camera component (e.g. ``viam:camera:realsense``).

    Assumes the camera returns a colour image and a ``VIAM_RAW_DEPTH`` image
    aligned to it. Alignment and which source names the RealSense module uses
    must be verified on the machine (``tools/check_machine.py`` prints them).
    """

    def __init__(self, robot, camera_name: str, color_source: Optional[str] = None,
                 depth_source: Optional[str] = None):
        from viam.components.camera import Camera  # local import: keep module importable without SDK

        self.camera = Camera.from_robot(robot, camera_name)
        self.camera_name = camera_name
        self.color_source = color_source
        self.depth_source = depth_source
        self._intr: Optional[Intrinsics] = None

    async def intrinsics(self) -> Optional[Intrinsics]:
        if self._intr is None:
            props = await self.camera.get_properties()
            ip = props.intrinsic_parameters
            if ip.width_px > 0 and ip.focal_x_px > 0:
                self._intr = Intrinsics(ip.width_px, ip.height_px, ip.focal_x_px, ip.focal_y_px,
                                        ip.center_x_px, ip.center_y_px)
            else:
                log.warning("camera %s reports no intrinsics; depth deprojection disabled", self.camera_name)
        return self._intr

    async def get_frame(self) -> Frame:
        from viam.media.video import CameraMimeType

        images, _ = await self.camera.get_images()
        bgr, depth = None, None
        for img in images:
            name = getattr(img, "name", "")
            is_depth = img.mime_type == CameraMimeType.VIAM_RAW_DEPTH
            if is_depth and depth is None and (self.depth_source in (None, name)):
                depth = np.array(img.bytes_to_depth_array(), dtype=np.uint16)
            elif not is_depth and bgr is None and (self.color_source in (None, name)):
                bgr = _decode_color(img)
        if bgr is None:
            raise RuntimeError(f"camera '{self.camera_name}' returned no colour image "
                               f"(got: {[(getattr(i, 'name', '?'), i.mime_type) for i in images]})")
        intr = await self.intrinsics()
        if depth is not None and intr is not None and depth.shape != (intr.height, intr.width):
            log.warning("depth %s != intrinsics %dx%d — is depth aligned to colour?",
                        depth.shape, intr.width, intr.height)
        return Frame(bgr=bgr, depth_mm=depth, intrinsics=intr, source=f"viam:{self.camera_name}")


def _decode_color(img) -> np.ndarray:
    from viam.media.video import CameraMimeType

    if img.mime_type == CameraMimeType.VIAM_RGBA:
        rgba = np.frombuffer(img.data[12:], dtype=np.uint8).reshape(img.height, img.width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    buf = np.frombuffer(img.data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot decode colour image of type {img.mime_type}")
    return bgr
