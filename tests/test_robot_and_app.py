import asyncio

import numpy as np
import pytest

import synthetic as syn
from printbot.app.approval import ScriptedApprover
from printbot.app.events import EventLog
from printbot.app.state_machine import Orchestrator, State
from printbot.config import _deep_merge
from printbot.inspection.classifier import PrintClassifier
from printbot.robot.controller import MotionRefused, RobotController
from printbot.robot.grasp import plan_top_down_grasp
from printbot.robot.transforms import FixedTransformer

GRIPPER = {"max_opening_mm": 80, "finger_length_mm": 30, "jaw_clearance_mm": 10, "plate_margin_mm": 3}
POSES = {n: {"x": 300, "y": i * 100, "z": 300, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}
         for i, n in enumerate(("home", "success_bin", "failed_bin"))}


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ grasp

def test_grasp_refused_without_gripper_spec(settings):
    p = plan_top_down_grasp(np.zeros(3), np.array([1, 0, 0]), (30, 20), 15, settings["robot"]["gripper"])
    assert not p.feasible and "gripper spec unknown" in p.reasons[0]


def test_grasp_feasible_and_perpendicular_to_long_axis():
    p = plan_top_down_grasp(np.array([0, 0, 15.0]), np.array([1, 0, 0]), (30, 20), 15, GRIPPER)
    assert p.feasible and p.jaw_width_mm == 30
    assert p.grasp["theta"] == pytest.approx(90)
    assert p.grasp["z"] >= 3.0 - 1e-9       # never below plate margin
    assert p.pre_grasp["z"] > p.grasp["z"] and p.grasp["o_z"] == -1


def test_grasp_too_wide_refused():
    p = plan_top_down_grasp(np.zeros(3), np.array([1, 0, 0]), (120, 75), 20, GRIPPER)
    assert not p.feasible and "exceeds gripper opening" in p.reasons[0]


# -------------------------------------------------------------- controller

def test_controller_is_dry_run_without_env(monkeypatch, settings):
    monkeypatch.delenv("PRINTBOT_ALLOW_MOTION", raising=False)
    s = _deep_merge(settings, {"robot": {"motion_enabled": True}})
    assert RobotController(None, s, allow_motion=True).live is False


def test_controller_named_pose_missing(settings):
    with pytest.raises(MotionRefused):
        RobotController(None, settings).named_pose("home")


def test_transformer_camera_to_world():
    w = run(FixedTransformer(syn.T_WORLD_CAM).to_world(np.array([10.0, 20.0, 385.0])))
    assert np.allclose(w, [10, -20, 15])


# ------------------------------------------------------------ orchestrator

def make(settings, frames, answers, **robot_over):
    s = _deep_merge(settings, {"robot": {"gripper": GRIPPER, "named_poses": POSES, **robot_over}})
    ev = EventLog(None)
    ctl = RobotController(None, s, events=ev)
    appr = ScriptedApprover(answers)
    orch = Orchestrator(s, syn.SequenceSource(frames), PrintClassifier(s), FixedTransformer(syn.T_WORLD_CAM),
                        ctl, appr, ev)
    return orch, ev, appr


def test_success_cycle_completes_in_dry_run(settings):
    frames = [syn.part_frame(seed=i) for i in range(4)] + [syn.empty_frame(seed=10 + i) for i in range(3)]
    orch, ev, appr = make(settings, frames, [True, True])
    r = run(orch.run_cycle())
    assert r.final_state == State.COMPLETED, r.message
    assert State.PICK in r.history and State.PLACE in r.history
    moves = [e for e in ev.records if e["kind"] == "move"]
    assert all(e["dry_run"] for e in moves)
    assert [m["label"] for m in moves] == ["pre_grasp", "grasp", "lift", "success_bin", "home"]


def test_operator_denial_aborts_before_motion(settings):
    orch, ev, _ = make(settings, [syn.part_frame()], [False])
    r = run(orch.run_cycle())
    assert r.final_state == State.ABORTED
    assert not [e for e in ev.records if e["kind"] == "move"]


def test_failed_print_goes_to_human_by_default(settings):
    orch, ev, appr = make(settings, [syn.spaghetti_frame()], [True, True])
    r = run(orch.run_cycle())
    assert r.final_state == State.HUMAN_REVIEW and not appr.asked


def test_empty_plate_completes_without_motion(settings):
    orch, ev, _ = make(settings, [syn.empty_frame()], [])
    assert run(orch.run_cycle()).final_state == State.COMPLETED
    assert not [e for e in ev.records if e["kind"] == "move"]


def test_plate_not_clear_after_removal_goes_to_human(settings):
    orch, _, _ = make(settings, [syn.part_frame()], [True, True])
    r = run(orch.run_cycle())
    assert r.final_state == State.HUMAN_REVIEW and "not clear" in r.message


def test_camera_failure_is_error(settings):
    class Broken:
        async def get_frame(self):
            raise RuntimeError("no recent color frame")

    orch, _, _ = make(settings, [syn.part_frame()], [])
    orch.frames = Broken()
    r = run(orch.run_cycle())
    assert r.final_state == State.ERROR and "no recent color frame" in r.message


def test_wrist_camera_scan_pose_visited_before_inspection_and_verification(settings):
    frames = [syn.part_frame(seed=i) for i in range(4)] + [syn.empty_frame(seed=10 + i) for i in range(3)]
    poses = {**POSES, "scan": {"x": 300, "y": 0, "z": 450, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}}
    orch, ev, _ = make(settings, frames, [True, True], named_poses=poses)
    r = run(orch.run_cycle())
    labels = [e["label"] for e in ev.records if e["kind"] == "move"]
    assert r.final_state == State.COMPLETED
    assert labels[0] == "scan" and labels[-1] == "scan"
