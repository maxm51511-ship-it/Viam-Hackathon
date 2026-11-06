import struct

from printbot.config import _deep_merge

import numpy as np
import pytest

import synthetic as syn
from printbot.inspection.classifier import NO_DEPTH_CONFIDENCE_CAP, Assessor, PrintClassifier
from printbot.inspection.reference import (ExpectedPart, compare, expected_from_gcode, expected_from_stl,
                                           gcode_extrusion_points)
from printbot.types import Detection2D, Inspection, MaterialClass, Verdict


def test_part_is_success(settings):
    insp = PrintClassifier(settings).inspect(syn.part_frame())
    assert insp.verdict == Verdict.SUCCESS and insp.confidence >= 0.7


def test_spaghetti_is_failed(settings):
    insp = PrintClassifier(settings).inspect(syn.spaghetti_frame())
    assert insp.verdict == Verdict.FAILED


def test_empty_plate(settings):
    assert PrintClassifier(settings).inspect(syn.empty_frame()).verdict == Verdict.EMPTY


def test_no_depth_caps_confidence(settings):
    bgr, depth, _ = syn.base_scene()
    syn.add_box(bgr, depth)
    insp = PrintClassifier(settings).inspect(syn.frame(bgr, depth, with_depth=False))
    assert insp.confidence <= NO_DEPTH_CONFIDENCE_CAP


def test_yolo_disagreement_gives_uncertain(settings):
    class FakeYolo:
        def predict(self, bgr):
            return [Detection2D(MaterialClass.SPAGHETTI, 0.9, (0, 0, 10, 10), source="yolo")]

    insp = PrintClassifier(settings, yolo=FakeYolo()).inspect(syn.part_frame())
    assert insp.verdict == Verdict.UNCERTAIN


def test_reference_mismatch_flags_failure(settings):
    exp = ExpectedPart((30, 20), 40.0, "test")  # part should be 40 mm tall, observed 15 mm
    insp = PrintClassifier(settings, expected=exp).inspect(syn.part_frame(height_mm=15))
    assert insp.verdict == Verdict.UNCERTAIN  # heuristics say success, reference says failed -> conflict


def _insp(v, c=0.9):
    return Inspection(v, c, [], [], [])


def test_assessor_requires_multiple_observations(settings):
    a = Assessor(settings["classifier"]).assess([_insp(Verdict.FAILED)])
    assert a.needs_human and a.verdict == Verdict.UNCERTAIN


def test_assessor_inconsistent_observations(settings):
    a = Assessor(settings["classifier"]).assess([_insp(Verdict.FAILED), _insp(Verdict.SUCCESS), _insp(Verdict.FAILED)])
    assert a.needs_human and a.verdict == Verdict.UNCERTAIN


def test_assessor_consistent(settings):
    a = Assessor(settings["classifier"]).assess([_insp(Verdict.SUCCESS)] * 3)
    assert not a.needs_human and a.verdict == Verdict.SUCCESS


def test_assessor_low_confidence_needs_human(settings):
    a = Assessor(settings["classifier"]).assess([_insp(Verdict.SUCCESS, 0.5)] * 3)
    assert a.needs_human and a.verdict == Verdict.UNCERTAIN


def _box_tris(dx, dy, dz):
    v = np.array([[x, y, z] for x in (0, dx) for y in (0, dy) for z in (0, dz)], float)
    return np.stack([v[[0, 1, 2]], v[[7, 6, 5]], v[[3, 4, 1]]])  # extents suffice for the test


def test_stl_binary_and_ascii(tmp_path):
    tris = _box_tris(40, 25, 12)
    b = tmp_path / "b.stl"
    with b.open("wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", 0, 0, 1) + t.astype("<f4").tobytes() + b"\0\0")
    a = tmp_path / "a.stl"
    a.write_text("solid s\n" + "".join(
        "facet normal 0 0 1\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t) + "endloop\nendfacet\n"
        for t in tris) + "endsolid s\n")
    for p in (a, b):
        e = expected_from_stl(p)
        assert e.footprint_mm == pytest.approx((40, 25)) and e.height_mm == pytest.approx(12)


def test_gcode_extents(tmp_path):
    g = tmp_path / "p.gcode"
    g.write_text("G90\nM82\nG92 E0\nG1 Z0.2 F600\nG0 X10 Y10\nG1 X50 Y10 E1.0\nG1 X50 Y30 E2.0\n"
                 "G1 X10 Y30 E3.0\nG1 Z10.2\nG1 X50 Y30 E4.0 ; top\nG1 X60 Y60 ; travel, no extrusion\n")
    pts = gcode_extrusion_points(g.read_text().splitlines())
    assert pts[:, 2].max() == pytest.approx(10.2)
    e = expected_from_gcode(g, skirt_trim_pct=0)
    assert e.footprint_mm == pytest.approx((40, 20)) and e.height_mm == pytest.approx(10.2)


def test_compare_without_depth_returns_none(settings):
    bgr, depth, _ = syn.base_scene()
    syn.add_box(bgr, depth)
    from printbot.perception.objects import extract_objects
    from printbot.perception.plate import PlateSegmenter
    f = syn.frame(bgr, depth, with_depth=False)
    obj = extract_objects(f, PlateSegmenter(settings["plate"]).segment(f), settings["objects"])[0]
    assert compare(obj, ExpectedPart((30, 20), 15, "t")) is None


# ------------------------------------------------------------ colour demo mode
WHITE, RED, PINK = (235, 235, 235), (30, 30, 210), (180, 150, 255)


def _scene(color=None, spaghetti=False, plate=(40, 40, 40), seed=0):
    bgr, depth, rng = syn.base_scene(plate_bgr=plate, seed=seed)
    if spaghetti:
        syn.add_spaghetti(bgr, depth, rng, color=RED)
    elif color is not None:
        syn.add_box(bgr, depth, color=color)
    return syn.frame(bgr, depth)


def test_demo_white_is_success(demo_settings):
    assert PrintClassifier(demo_settings).inspect(_scene(WHITE)).verdict == Verdict.SUCCESS


def test_demo_red_part_is_failed(demo_settings):
    assert PrintClassifier(demo_settings).inspect(_scene(RED)).verdict == Verdict.FAILED


def test_demo_red_spaghetti_is_failed(demo_settings):
    assert PrintClassifier(demo_settings).inspect(_scene(spaghetti=True, seed=3)).verdict == Verdict.FAILED


def test_demo_other_colour_is_uncertain(demo_settings):
    assert PrintClassifier(demo_settings).inspect(_scene(PINK)).verdict == Verdict.UNCERTAIN


def test_demo_white_part_on_light_plate(demo_settings):
    s = _deep_merge(demo_settings, {"plate": {"hsv_ranges": [{"lower": [0, 0, 150], "upper": [179, 60, 255]}]}})
    # plate (200) and part (235) both "white-ish": depth must still separate them
    insp = PrintClassifier(s).inspect(_scene(WHITE, plate=(200, 200, 200)))
    assert insp.verdict == Verdict.SUCCESS and len(insp.objects) == 1
