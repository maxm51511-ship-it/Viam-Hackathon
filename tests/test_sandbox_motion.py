"""Offline tests of the sandbox MOTION steps with fake robot/motion/gripper/camera.
Nothing here talks to a real machine."""

import asyncio
import copy

import pytest
from viam.proto.common import Pose, PoseInFrame

from test_sandbox import RED, WHITE, pi, scene, to_pcd

SCAN = {"x": 300.0, "y": 0.0, "z": 400.0, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": 0.0}
GOOD_BIN = {"x": 0.0, "y": 350.0, "z": 250.0, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": 0.0}
FAILED_BIN = {"x": 0.0, "y": -350.0, "z": 250.0, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": 0.0}


class FakeRobot:
    def __init__(self, plate_world_z=0.0):
        self.stopped = 0
        self.plate_z = plate_world_z

    async def transform_pose(self, q, dest):
        # camera at the scan pose, looking straight down, plate 400 mm below it
        p = q.pose
        return PoseInFrame(reference_frame=dest,
                           pose=Pose(x=SCAN["x"] + p.x, y=-p.y, z=self.plate_z + 400.0 - p.z, o_z=1))

    async def stop_all(self, extra=None):
        self.stopped += 1


class FakeMotion:
    def __init__(self, fail_on=None):
        self.pose = dict(SCAN, z=350.0)
        self.calls = []
        self.fail_on = fail_on

    async def get_pose(self, component_name, destination_frame, **kw):
        return PoseInFrame(reference_frame=destination_frame, pose=Pose(**self.pose))

    async def move(self, component_name, destination, world_state=None, constraints=None, **kw):
        assert component_name == "gripper"          # move the GRIPPER, not the arm
        assert destination.reference_frame == "world"
        self.calls.append((destination.pose, constraints))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("planning failed")
        self.pose = pi.pose_dict(destination.pose)
        return True


class FakeGripper:
    def __init__(self, grab_ok=True):
        self.events = []
        self.grab_ok = grab_ok

    async def open(self):
        self.events.append("open")

    async def grab(self):
        self.events.append("grab")
        return self.grab_ok

    async def is_holding_something(self):
        return type("S", (), {"is_holding_something": self.grab_ok})()


class FakeArm:
    async def get_end_position(self):
        return Pose(**SCAN)


class FakeCam:
    def __init__(self, scenes):
        self.data = [to_pcd(*s) for s in scenes]
        self.i = 0

    async def get_point_cloud(self):
        d = self.data[min(self.i, len(self.data) - 1)]
        self.i += 1
        return d, "pointcloud/pcd"


@pytest.fixture(autouse=True)
def motion_cfg(monkeypatch):
    m = copy.deepcopy(pi.MOTION)
    m.update(STEP=0, I_AM_AT_THE_ESTOP=False, scan_pose=None, good_bin_pose=None, failed_bin_pose=None,
             extra_obstacles=None)
    monkeypatch.setattr(pi, "MOTION", m)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(pi.asyncio, "sleep", lambda s: real_sleep(0))
    return m


def cell(scenes, **kw):
    return pi.Cell(FakeRobot(kw.pop("plate_z", 0.0)), FakeMotion(kw.pop("fail_on", None)),
                   FakeGripper(kw.pop("grab_ok", True)), FakeArm(), FakeCam(scenes))


def run(c, step):
    return asyncio.run(pi.run_step(c, step))


def ready(m, step):
    m.update(I_AM_AT_THE_ESTOP=True, scan_pose=SCAN, extra_obstacles=[], good_bin_pose=GOOD_BIN,
             failed_bin_pose=FAILED_BIN)


def labels(c):
    return [lbl for lbl, _ in c.moves]


def test_step0_is_read_only():
    c = cell([scene("white_part")])
    assert run(c, 0) == "read-only" and c.motion.calls == []


def test_motion_refused_without_estop_flag():
    c = cell([scene("white_part")])
    with pytest.raises(pi.SafetyError, match="E-stop"):
        run(c, 1)
    assert c.motion.calls == []


def test_step1_nudges_up_in_a_straight_line(motion_cfg):
    motion_cfg["I_AM_AT_THE_ESTOP"] = True
    c = cell([scene("empty")])
    assert run(c, 1) == "nudged"
    (pose, cons), = c.motion.calls
    assert pose.z == pytest.approx(380.0) and pose.x == pytest.approx(SCAN["x"])
    assert cons is not None and cons.linear_constraint


def test_step2_requires_scan_pose(motion_cfg):
    motion_cfg["I_AM_AT_THE_ESTOP"] = True
    with pytest.raises(pi.SafetyError, match="scan_pose"):
        run(cell([scene("empty")]), 2)


def test_step3_requires_printer_obstacles(motion_cfg):
    motion_cfg.update(I_AM_AT_THE_ESTOP=True, scan_pose=SCAN)
    with pytest.raises(pi.SafetyError, match="extra_obstacles"):
        run(cell([scene("white_part")]), 3)


def test_step3_only_hovers(motion_cfg):
    ready(motion_cfg, 3)
    c = cell([scene("white_part")])
    assert run(c, 3) == "hovering"
    assert labels(c) == ["scan", "hover"]
    hover = c.moves[1][1]
    assert hover["x"] == pytest.approx(SCAN["x"], abs=3) and hover["z"] == pytest.approx(115, abs=3)
    assert c.gripper.events == []


def test_step4_white_goes_to_good_bin(motion_cfg):
    ready(motion_cfg, 4)
    c = cell([scene("white_part")] * 3 + [scene("empty")])
    assert run(c, 4) == "sorted"
    assert labels(c) == ["scan", "hover", "grasp", "lift", "good_bin_pose", "scan"]
    assert c.gripper.events == ["open", "grab", "open"]
    grasp = c.moves[2][1]
    assert grasp["z"] >= motion_cfg["workspace"]["min_z_mm"]


def test_step4_red_goes_to_failed_bin(motion_cfg):
    ready(motion_cfg, 4)
    c = cell([scene("red_part")] * 3 + [scene("empty")])
    run(c, 4)
    assert "failed_bin_pose" in labels(c) and "good_bin_pose" not in labels(c)


def test_step4_uncertain_colour_does_not_pick(motion_cfg):
    ready(motion_cfg, 4)
    c = cell([scene("pink_part")])
    assert run(c, 4) == "UNCERTAIN"
    assert labels(c) == ["scan"] and c.gripper.events == []


def test_failed_move_stops_all(motion_cfg):
    ready(motion_cfg, 4)
    c = cell([scene("white_part")], fail_on=2)
    with pytest.raises(RuntimeError, match="planning failed"):
        run(c, 4)
    assert c.robot.stopped == 1


def test_grasp_below_fence_refused_before_hover(motion_cfg):
    ready(motion_cfg, 4)
    c = cell([scene("white_part")], plate_z=-60.0)    # plate below table level -> grasp z < min_z
    with pytest.raises(pi.SafetyError, match="outside fence"):
        run(c, 4)
    assert labels(c) == ["scan"]


def test_failed_grab_backs_off(motion_cfg):
    ready(motion_cfg, 4)
    c = cell([scene("white_part")], grab_ok=False)
    assert run(c, 4) == "grasp failed"
    assert labels(c) == ["scan", "hover", "grasp", "retreat"]
    assert c.gripper.events == ["open", "grab", "open"]


HOME = {"x": 250.0, "y": 0.0, "z": 450.0, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": 0.0}


def test_step5_goes_home(motion_cfg):
    motion_cfg.update(I_AM_AT_THE_ESTOP=True, home_pose=HOME)
    c = cell([scene("empty")])
    assert run(c, 5) == "at home" and labels(c) == ["home"]


def test_step5_requires_home_pose(motion_cfg):
    motion_cfg["I_AM_AT_THE_ESTOP"] = True
    with pytest.raises(pi.SafetyError, match="home_pose"):
        run(cell([scene("empty")]), 5)


def test_step4_ends_at_home_when_set(motion_cfg):
    ready(motion_cfg, 4)
    motion_cfg["home_pose"] = HOME
    c = cell([scene("white_part")] * 3 + [scene("empty")])
    assert run(c, 4) == "sorted"
    assert labels(c)[-2:] == ["scan", "home"]


def test_unknown_step_refused(motion_cfg):
    motion_cfg["I_AM_AT_THE_ESTOP"] = True
    with pytest.raises(pi.SafetyError, match="unknown STEP"):
        run(cell([scene("empty")]), 9)
