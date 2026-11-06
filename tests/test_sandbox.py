"""Offline tests of sandbox/print_inspector.py against synthetic point clouds."""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

import synthetic as syn
from printbot.perception.geometry import deproject

spec = importlib.util.spec_from_file_location(
    "print_inspector", Path(__file__).resolve().parent.parent / "sandbox" / "print_inspector.py")
pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi)

WHITE, RED = (235, 235, 235), (30, 30, 210)   # BGR
GOLD = (55, 160, 200)                           # BGR of a gold PEI sheet (RGB 200,160,55)
ANY_PLATE = dict(pi.CONFIG, plate_colour=None)  # colour check off


def to_pcd(bgr, depth, metres=True, rgb_type="F"):
    pts = deproject(depth, syn.INTR).reshape(-1, 3)
    rgb = bgr[..., ::-1].reshape(-1, 3).astype(np.uint32)
    ok = np.isfinite(pts).all(axis=1)
    pts, rgb = pts[ok], rgb[ok]
    if metres:
        pts = pts / 1000.0
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    t = "F" if rgb_type == "F" else "U"
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4" if t == "F" else "<u4")]
    rec = np.zeros(len(pts), dtype=dt)
    rec["x"], rec["y"], rec["z"] = pts.T
    rec["rgb"] = packed.astype("<u4").view("<f4") if t == "F" else packed
    hdr = (f"VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F {t}\nCOUNT 1 1 1 1\nWIDTH {len(pts)}\n"
           f"HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA binary\n")
    return hdr.encode() + rec.tobytes()


def scene(kind, plate=GOLD, seed=0):
    bgr, depth, rng = syn.base_scene(plate_bgr=plate, seed=seed)
    if kind == "white_part":
        syn.add_box(bgr, depth, size_mm=(30, 20), height_mm=15, color=WHITE)
    elif kind == "red_part":
        syn.add_box(bgr, depth, size_mm=(30, 20), height_mm=15, color=RED)
    elif kind == "red_spaghetti":
        syn.add_spaghetti(bgr, depth, rng, color=RED)
    elif kind == "pink_part":
        syn.add_box(bgr, depth, color=(180, 150, 255))
    return bgr, depth


@pytest.mark.parametrize("rgb_type", ["F", "U"])
def test_white_part_is_good(rgb_type):
    obs = pi.analyze(to_pcd(*scene("white_part"), rgb_type=rgb_type))
    assert obs["plate_found"] and len(obs["objects"]) == 1
    o = obs["objects"][0]
    assert o["verdict"] == "GOOD"
    assert o["height_mm"] == pytest.approx(15, abs=1.5)
    assert o["top_center_cam_mm"][2] == pytest.approx(385, abs=2)


def test_red_spaghetti_is_failed():
    obs = pi.analyze(to_pcd(*scene("red_spaghetti", seed=3)))
    assert obs["objects"] and all(o["verdict"] == "FAILED" for o in obs["objects"])


def test_red_part_on_white_plate_uses_depth_not_colour():
    obs = pi.analyze(to_pcd(*scene("red_part", plate=(230, 230, 230))), ANY_PLATE)
    assert [o["verdict"] for o in obs["objects"]] == ["FAILED"]


def test_white_part_on_white_plate_still_found_by_depth():
    obs = pi.analyze(to_pcd(*scene("white_part", plate=(230, 230, 230))), ANY_PLATE)
    assert [o["verdict"] for o in obs["objects"]] == ["GOOD"]


def test_other_colour_is_uncertain():
    obs = pi.analyze(to_pcd(*scene("pink_part")))
    assert [o["verdict"] for o in obs["objects"]] == ["UNCERTAIN"]


def test_empty_plate():
    obs = pi.analyze(to_pcd(*scene("empty")))
    assert obs["plate_found"] and obs["objects"] == []
    assert pi.decide([obs] * 3, 3)[0] == "EMPTY"


def test_decide_requires_agreement():
    good = pi.analyze(to_pcd(*scene("white_part")))
    bad = pi.analyze(to_pcd(*scene("red_part")))
    assert pi.decide([good] * 3, 3)[0] == "GOOD"
    assert pi.decide([bad] * 3, 3)[0] == "FAILED"
    assert pi.decide([good, bad, good], 3)[0] == "UNCERTAIN"
    assert pi.decide([good], 3)[0] == "UNCERTAIN"


def test_mm_pcd_and_speed():
    data = to_pcd(*scene("white_part"), metres=False)
    t = time.time()
    obs = pi.analyze(data)
    assert obs["objects"][0]["verdict"] == "GOOD"
    assert time.time() - t < 10  # pure-python budget per capture at 640x480


# ------------------------------------------------------------ plate colour
def test_gold_plate_recognised_and_reported():
    obs = pi.analyze(to_pcd(*scene("white_part")))
    assert obs["plate_found"] and obs["plate"]["colour"] == "gold/yellow"
    assert any("plate: gold/yellow" in n for n in obs["notes"])


def test_plate_of_wrong_colour_is_rejected():
    obs = pi.analyze(to_pcd(*scene("white_part", plate=(40, 40, 40))))   # grey plate, gold expected
    assert not obs["plate_found"] and "plate colour" in obs["notes"][-1]
    assert pi.decide([obs] * 3, 3)[0] == "UNCERTAIN"


def test_object_beside_the_plate_is_ignored():
    bgr, depth = scene("white_part")
    syn.add_box(bgr, depth, center_px=(80, 240), size_mm=(20, 20), height_mm=15, color=RED)  # on the table area
    obs = pi.analyze(to_pcd(bgr, depth))
    assert [o["verdict"] for o in obs["objects"]] == ["GOOD"]


def test_closer_white_surface_is_not_taken_as_plate():
    rnd = __import__("random").Random(0)
    white = [(rnd.uniform(-80, 80), rnd.uniform(-80, 0), 350.0, 230, 230, 230) for _ in range(3000)]
    gold = [(rnd.uniform(-80, 80), rnd.uniform(0, 80), 400.0, 200, 160, 55) for _ in range(3000)]
    (n, d, frac, pts), reason = pi.find_plate_plane(white + gold, pi.CONFIG)
    assert reason is None and d == pytest.approx(400, abs=3)
