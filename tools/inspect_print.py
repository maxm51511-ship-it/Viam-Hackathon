"""Run inspection on saved frames (offline) or the live camera, and write an
annotated image per frame. Never moves the robot.

    python tools/inspect_print.py --frames data/captures/session1
    python tools/inspect_print.py --live --n 3
    python tools/inspect_print.py --frames data/captures/s1 --stl part.stl --yolo
    python tools/inspect_print.py --demo            # synthetic scenes, no hardware
"""

import argparse
import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from printbot.config import load_settings  # noqa: E402
from printbot.inspection.classifier import Assessor, PrintClassifier  # noqa: E402
from printbot.inspection.reference import expected_from_gcode, expected_from_stl  # noqa: E402
from printbot.perception.sources import FileFrameSource, ViamCameraSource  # noqa: E402
from printbot.types import Verdict  # noqa: E402

COLORS = {Verdict.SUCCESS: (0, 200, 0), Verdict.FAILED: (0, 0, 255),
          Verdict.UNCERTAIN: (0, 200, 255), Verdict.EMPTY: (200, 200, 200)}


def annotate(frame, insp, seg_polygon=None) -> np.ndarray:
    img = frame.bgr.copy()
    for o in insp.objects:
        overlay = img.copy()
        overlay[o.mask] = COLORS[insp.verdict]
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        box = cv2.boxPoints(o.min_rect).astype(np.int32)
        cv2.polylines(img, [box], True, (255, 255, 0), 1)
        u, v = map(int, o.centroid_px)
        cv2.putText(img, f"h={o.max_height_mm:.1f} sol={o.solidity:.2f}", (u + 5, v), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
    if seg_polygon is not None:
        cv2.polylines(img, [seg_polygon], True, (255, 0, 255), 1)
    cv2.putText(img, f"{insp.verdict.value} {insp.confidence:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                COLORS[insp.verdict], 2)
    for i, r in enumerate(insp.reasons[:4]):
        cv2.putText(img, r[:90], (10, 50 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


async def main(a) -> None:
    s = load_settings()
    expected = expected_from_stl(a.stl) if a.stl else expected_from_gcode(a.gcode) if a.gcode else None
    yolo = None
    if a.yolo:
        from printbot.perception.yolo import YoloSegmenter
        yolo = YoloSegmenter(s["yolo"])
    clf = PrintClassifier(s, yolo=yolo, expected=expected)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    robot = None
    if a.live:
        from printbot.robot.connection import connect
        robot = await connect()
        src = ViamCameraSource(robot, s["resources"]["camera"], s["camera"]["color_source_name"],
                               s["camera"]["depth_source_name"])
    elif a.demo:
        sys.path.insert(0, str(ROOT / "tests"))
        import synthetic as syn
        frames = []
        for kind in ("white_part", "red_part", "red_spaghetti", "empty"):
            bgr, depth, rng = syn.base_scene(seed=len(frames) + 3)
            if kind == "white_part":
                syn.add_box(bgr, depth, color=(235, 235, 235))
            elif kind == "red_part":
                syn.add_box(bgr, depth, color=(30, 30, 210), angle_deg=25)
            elif kind == "red_spaghetti":
                syn.add_spaghetti(bgr, depth, rng, color=(30, 30, 210))
            frames.append(syn.frame(bgr, depth))
        src = syn.SequenceSource(frames)
        a.n = len(frames)
    else:
        src = FileFrameSource(a.frames)
        a.n = a.n or len(src.paths)
    try:
        inspections = []
        for i in range(a.n):
            f = await src.get_frame()
            insp = clf.inspect(f)
            inspections.append(insp)
            seg = clf.segmenter.segment(f)
            p = out / f"inspect_{i:03d}.png"
            cv2.imwrite(str(p), annotate(f, insp, seg.plate_polygon))
            print(f"[{i}] {f.source}: {insp.verdict.value} conf={insp.confidence:.2f} -> {p}")
            for r in insp.reasons:
                print(f"      - {r}")
        if not a.demo:
            asm = Assessor(s["classifier"]).assess(inspections)
            print(f"\nASSESSMENT: {asm.verdict.value} conf={asm.confidence:.2f} needs_human={asm.needs_human}")
    finally:
        if robot is not None:
            await robot.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--frames")
    g.add_argument("--live", action="store_true")
    g.add_argument("--demo", action="store_true")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--stl")
    p.add_argument("--gcode")
    p.add_argument("--yolo", action="store_true")
    p.add_argument("--out", default="logs/inspect")
    asyncio.run(main(p.parse_args()))
