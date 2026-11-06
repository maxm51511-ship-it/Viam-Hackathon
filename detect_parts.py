"""READ-ONLY vision check: find red (failed) and white (good) parts on the bed.

Red detection uses the machine's configured "color-detector" vision service
(tuned to #c8202a). White has no well-defined hue, so the built-in hue-based
detector can't target it; white regions are found locally instead, by
thresholding the color image for low-saturation / high-brightness pixels.

Saves an annotated image (annotated.jpg) with red boxes in red and white
blobs outlined in blue, and prints the 3D "touch" target for the best red
detection in both the camera frame and the arm's world frame.

Does NOT move the arm or gripper. transform_pose and the detector/camera
calls are all read-only.
"""

import asyncio
import os

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

load_dotenv()

CAMERA_NAME = "cam"
RED_DETECTOR_NAME = "color-detector"

# The Bambu "Textured PEI Plate" is gold/tan: hue ~10-20deg, moderate
# saturation, medium brightness (PIL HSV is 0-255 per channel). Everything
# else in frame (desk, laptop, carpet, basket) reads as cooler/grayer.
PLATE_HUE_MAX = 32
PLATE_SAT_RANGE = (12, 150)
PLATE_VAL_RANGE = (50, 235)
PLATE_MIN_BLOB_PX = 5000

# White parts are near-zero saturation and distinctly brighter than the gold
# plate (whose own highlights only reach ~155 value); search is restricted to
# inside the plate's bounding box so background whites never qualify.
WHITE_MAX_SATURATION = 0.12
WHITE_MIN_VALUE = 0.75
WHITE_MIN_BLOB_PX = 150

# Vision-service "red" detections smaller than this are almost always noise
# (skin tone, fabric) rather than an actual part.
RED_MIN_BLOB_PX = 400


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    return await RobotClient.at_address(os.environ["VIAM_ADDRESS"], opts)


def connected_components(mask: np.ndarray, min_px: int):
    """Flood-fill a boolean mask into connected blobs, each >= min_px pixels.
    Returns [(x_min,y_min,x_max,y_max,pixel_count), ...]."""
    visited = np.zeros_like(mask, dtype=bool)
    boxes = []
    h, w = mask.shape
    for y0 in range(h):
        for x0 in np.nonzero(mask[y0] & ~visited[y0])[0]:
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            ys, xs = [], []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(ys) >= min_px:
                boxes.append((min(xs), min(ys), max(xs), max(ys), len(ys)))
    return boxes


def find_plate_bbox(rgb: np.ndarray):
    """Locate the gold PEI plate as the largest gold-ish connected blob.
    Returns (x_min,y_min,x_max,y_max), or the full frame if none is found."""
    hsv = np.array(Image.fromarray(rgb).convert("HSV"))
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (
        (hue <= PLATE_HUE_MAX)
        & (sat >= PLATE_SAT_RANGE[0]) & (sat <= PLATE_SAT_RANGE[1])
        & (val >= PLATE_VAL_RANGE[0]) & (val <= PLATE_VAL_RANGE[1])
    )
    blobs = connected_components(mask, PLATE_MIN_BLOB_PX)
    if not blobs:
        h, w = mask.shape
        return (0, 0, w - 1, h - 1)
    x0, y0, x1, y1, _ = max(blobs, key=lambda b: b[4])
    return (x0, y0, x1, y1)


def find_white_boxes(rgb: np.ndarray, plate_bbox):
    """Return [(x_min,y_min,x_max,y_max)] for white blobs via HSV thresholding,
    restricted to inside the plate so background whites never qualify."""
    px0, py0, px1, py1 = plate_bbox
    hsv = np.array(Image.fromarray(rgb).convert("HSV"), dtype=np.float32)
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    mask = (sat < WHITE_MAX_SATURATION) & (val > WHITE_MIN_VALUE)
    mask[:py0, :] = mask[py1 + 1 :, :] = False
    mask[:, :px0] = mask[:, px1 + 1 :] = False
    return [b[:4] for b in connected_components(mask, WHITE_MIN_BLOB_PX)]


def median_depth(depth_arr, cx, cy, radius=3):
    """Median of valid (nonzero) depth samples in a small patch, in mm."""
    h, w = len(depth_arr), len(depth_arr[0])
    vals = [
        depth_arr[y][x]
        for y in range(max(0, cy - radius), min(h, cy + radius + 1))
        for x in range(max(0, cx - radius), min(w, cx + radius + 1))
        if depth_arr[y][x] > 0
    ]
    return float(np.median(vals)) if vals else None


def pixel_to_camera_point(u, v, depth_mm, intrinsics):
    """Pinhole unprojection: pixel + depth -> 3D point in the camera frame (mm)."""
    x = (u - intrinsics.center_x_px) * depth_mm / intrinsics.focal_x_px
    y = (v - intrinsics.center_y_px) * depth_mm / intrinsics.focal_y_px
    return x, y, depth_mm


async def detect(robot):
    """Run one detection pass. Returns a dict with everything needed both to
    report on the scene and to compute a 3D touch target. Fully read-only."""
    cam = Camera.from_robot(robot, CAMERA_NAME)
    props = await cam.get_properties()
    intrinsics = props.intrinsic_parameters

    detector = VisionClient.from_robot(robot, RED_DETECTOR_NAME)
    raw_red = await detector.get_detections_from_camera(CAMERA_NAME)

    images, _meta = await cam.get_images()
    color_img = next(i for i in images if i.name == "color")
    depth_img = next(i for i in images if i.name == "depth")
    depth_arr = depth_img.bytes_to_depth_array()
    rgb = np.array(Image.open(__import__("io").BytesIO(color_img.data)).convert("RGB"))

    plate_bbox = find_plate_bbox(rgb)
    px0, py0, px1, py1 = plate_bbox

    def on_plate(d):
        cx, cy = (d.x_min + d.x_max) / 2, (d.y_min + d.y_max) / 2
        return px0 <= cx <= px1 and py0 <= cy <= py1

    def big_enough(d):
        return (d.x_max - d.x_min) * (d.y_max - d.y_min) >= RED_MIN_BLOB_PX

    red_detections = [d for d in raw_red if on_plate(d) and big_enough(d)]
    white_boxes = find_white_boxes(rgb, plate_bbox)

    return {
        "intrinsics": intrinsics,
        "depth_arr": depth_arr,
        "rgb": rgb,
        "plate_bbox": plate_bbox,
        "red_detections": red_detections,
        "red_dropped": len(raw_red) - len(red_detections),
        "white_boxes": white_boxes,
    }


def annotate(scene, path="annotated.jpg"):
    px0, py0, px1, py1 = scene["plate_bbox"]
    im = Image.fromarray(scene["rgb"])
    draw = ImageDraw.Draw(im)
    draw.rectangle([px0, py0, px1, py1], outline=(0, 255, 0), width=3)
    draw.text((px0, max(0, py0 - 14)), "plate", fill=(0, 255, 0))
    for d in scene["red_detections"]:
        draw.rectangle([d.x_min, d.y_min, d.x_max, d.y_max], outline=(255, 0, 0), width=4)
        draw.text((d.x_min, max(0, d.y_min - 14)), f"RED fail {d.confidence:.2f}", fill=(255, 0, 0))
    for b in scene["white_boxes"]:
        draw.rectangle(list(b), outline=(0, 128, 255), width=4)
        draw.text((b[0], max(0, b[1] - 14)), "WHITE good", fill=(0, 128, 255))
    im.save(path)
    return path


async def best_red_world_pose(robot, scene):
    """Project the largest red detection to a world-frame Pose (gripper
    pointing straight down at its top surface), or None if there's nothing
    valid to touch. Read-only (uses transform_pose, no motion)."""
    red = scene["red_detections"]
    if not red:
        return None, None
    best = max(red, key=lambda d: (d.x_max - d.x_min) * (d.y_max - d.y_min))
    cx = int((best.x_min + best.x_max) / 2)
    cy = int((best.y_min + best.y_max) / 2)
    depth_mm = median_depth(scene["depth_arr"], cx, cy)
    if depth_mm is None:
        return None, best
    px, py, pz = pixel_to_camera_point(cx, cy, depth_mm, scene["intrinsics"])
    cam_pose = PoseInFrame(
        reference_frame="cam",
        pose=Pose(x=px, y=py, z=pz, o_x=0, o_y=0, o_z=1, theta=0),
    )
    world_pose = await robot.transform_pose(cam_pose, "world")
    return world_pose.pose, best


async def main():
    robot = await connect()
    try:
        scene = await detect(robot)
        px0, py0, px1, py1 = scene["plate_bbox"]
        print(f"Plate bbox: ({px0},{py0})-({px1},{py1})")
        print(f"Red detections ({RED_DETECTOR_NAME}): {len(scene['red_detections'])} "
              f"({scene['red_dropped']} dropped as off-plate/too small)")
        for d in scene["red_detections"]:
            print(f"  conf={d.confidence:.2f} bbox=({d.x_min},{d.y_min})-({d.x_max},{d.y_max})")
        print(f"White blobs (local HSV threshold, on-plate only): {len(scene['white_boxes'])}")
        for b in scene["white_boxes"]:
            print(f"  bbox={b}")

        path = annotate(scene)
        print(f"\nSaved {path}")

        wp, best = await best_red_world_pose(robot, scene)
        if best is None:
            print("\nNo red (failed) part detected - nothing to propose touching.")
        elif wp is None:
            print("\nBest red detection has no valid depth - can't compute a 3D point.")
        else:
            print(
                "PROPOSED touch target in world/arm frame (NOT sent - awaiting approval):\n"
                f"  x={wp.x:.1f} y={wp.y:.1f} z={wp.z:.1f}  "
                f"o=({wp.o_x:.2f},{wp.o_y:.2f},{wp.o_z:.2f}) theta={wp.theta:.1f}"
            )
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
