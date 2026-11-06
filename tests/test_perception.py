import numpy as np
import pytest

import synthetic as syn
from printbot.perception.geometry import deproject, fit_plane_ransac, parse_pcd, project
from printbot.perception.objects import extract_objects
from printbot.perception.plate import PlateSegmenter
from printbot.perception.sources import load_frame, save_frame


def test_deproject_project_roundtrip():
    depth = np.full((syn.H, syn.W), 400.0)
    pts = deproject(depth, syn.INTR)
    uv = project(pts.reshape(-1, 3)[[0, 1000, 5000]], syn.INTR)
    assert np.allclose(uv, [[0, 0], [1000 % syn.W, 1000 // syn.W], [5000 % syn.W, 5000 // syn.W]])


def test_deproject_rejects_misaligned_depth():
    with pytest.raises(ValueError, match="aligned"):
        deproject(np.ones((10, 10)), syn.INTR)


def test_ransac_plane_tilted_with_outliers():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-100, 100, (3000, 2))
    z = 400 + 0.1 * xy[:, 0] + rng.normal(0, 0.3, 3000)
    pts = np.column_stack([xy, z])
    pts[:300, 2] -= 20  # outliers (objects)
    plane, inl = fit_plane_ransac(pts, iterations=200, inlier_mm=1.5)
    assert inl[300:].mean() > 0.95 and inl[:300].mean() < 0.05
    assert plane.signed_distance(np.array([[0, 0, 380.0]]))[0] == pytest.approx(20, abs=1.0)  # above plate = +


def _pcd(points, binary=True):
    hdr = ("VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
           f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\n"
           f"DATA {'binary' if binary else 'ascii'}\n")
    if binary:
        rec = np.zeros(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
        rec["x"], rec["y"], rec["z"] = points.T
        rec["rgb"] = (255 << 16) | (128 << 8) | 7
        return hdr.encode() + rec.tobytes()
    body = "\n".join(f"{x} {y} {z} {(255 << 16) | (128 << 8) | 7}" for x, y, z in points)
    return (hdr + body + "\n").encode()


@pytest.mark.parametrize("binary", [True, False])
def test_parse_pcd_metres_auto_to_mm(binary):
    pts_m = np.array([[0.1, 0.2, 0.4], [0.0, 0.0, 0.5]])
    xyz, rgb = parse_pcd(_pcd(pts_m, binary), units="auto")
    assert np.allclose(xyz, pts_m * 1000, atol=1e-3)
    assert rgb[0].tolist() == [255, 128, 7]


def test_parse_pcd_explicit_mm():
    xyz, _ = parse_pcd(_pcd(np.array([[100.0, 0, 400]])), units="mm")
    assert xyz[0, 2] == pytest.approx(400)


def test_plate_and_part_segmentation(settings):
    f = syn.part_frame(size_mm=(30, 20), height_mm=15)
    seg = PlateSegmenter(settings["plate"]).segment(f)
    assert seg.used_depth and seg.plate_polygon is not None
    objs = extract_objects(f, seg, settings["objects"])
    assert len(objs) == 1
    o = objs[0]
    assert o.max_height_mm == pytest.approx(15, abs=1.0)
    assert o.footprint_mm[0] == pytest.approx(30, abs=3) and o.footprint_mm[1] == pytest.approx(20, abs=3)
    assert o.solidity > 0.9 and o.n_fragments == 1
    assert o.top_center_cam_mm[2] == pytest.approx(385, abs=1.0)


def test_filament_same_colour_as_plate_found_by_depth(settings):
    """Colour alone cannot see this part; depth must."""
    bgr, depth, _ = syn.base_scene()
    syn.add_box(bgr, depth, color=(42, 42, 42))
    seg = PlateSegmenter(settings["plate"]).segment(syn.frame(bgr, depth))
    assert len(extract_objects(syn.frame(bgr, depth), seg, settings["objects"])) == 1


def test_colour_only_fallback_flags_low_reliability(settings):
    bgr, depth, _ = syn.base_scene()
    syn.add_box(bgr, depth)
    seg = PlateSegmenter(settings["plate"]).segment(syn.frame(bgr, depth, with_depth=False))
    assert not seg.used_depth and any("colour-only" in n for n in seg.notes)
    assert seg.object_mask.sum() > 500


def test_plate_not_found(settings):
    bgr, depth, _ = syn.base_scene(plate_bgr=(200, 200, 200))
    seg = PlateSegmenter(settings["plate"]).segment(syn.frame(bgr, depth))
    assert seg.plate_polygon is None and "not found" in seg.notes[0]


def test_frame_save_load_roundtrip(tmp_path):
    f = syn.part_frame()
    p = save_frame(f, tmp_path, "x")
    g = load_frame(p)
    assert np.array_equal(g.bgr, f.bgr) and np.array_equal(g.depth_mm, f.depth_mm)
    assert g.intrinsics == f.intrinsics
