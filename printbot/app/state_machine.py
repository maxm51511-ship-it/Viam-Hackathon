"""Task orchestration state machine.

    IDLE -> ENVIRONMENT_CHECK -> INSPECTION -> ASSESSMENT
      ASSESSMENT: EMPTY     -> COMPLETED
                  UNCERTAIN -> HUMAN_REVIEW
                  SUCCESS   -> GRASP_PLANNING (dest = success_bin)
                  FAILED    -> GRASP_PLANNING (dest = failed_bin) if failed_removal_mode == "grasp"
                               else HUMAN_REVIEW
    GRASP_PLANNING -> OPERATOR_APPROVAL -> RELEASE -> PICK -> PLACE
                   -> REMOVAL_VERIFICATION -> COMPLETED
    any infeasible plan / denied approval -> HUMAN_REVIEW or ABORTED
    any exception -> ERROR (robot stop_all issued)

Dependencies are injected (frame source, classifier, transformer, controller,
approver) so the full cycle runs offline in tests with fakes and the
controller in DRY-RUN.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from printbot.inspection.classifier import Assessor
from printbot.robot.grasp import GraspPlan, plan_top_down_grasp
from printbot.types import Assessment, Verdict

log = logging.getLogger(__name__)


class State(str, enum.Enum):
    IDLE = "IDLE"
    ENVIRONMENT_CHECK = "ENVIRONMENT_CHECK"
    INSPECTION = "INSPECTION"
    ASSESSMENT = "ASSESSMENT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    GRASP_PLANNING = "GRASP_PLANNING"
    OPERATOR_APPROVAL = "OPERATOR_APPROVAL"
    RELEASE = "RELEASE"
    PICK = "PICK"
    PLACE = "PLACE"
    REMOVAL_VERIFICATION = "REMOVAL_VERIFICATION"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


TERMINAL = {State.COMPLETED, State.HUMAN_REVIEW, State.ABORTED, State.ERROR}


@dataclass
class CycleResult:
    final_state: State
    history: list[State] = field(default_factory=list)
    assessment: Optional[Assessment] = None
    grasp: Optional[GraspPlan] = None
    message: str = ""


class Orchestrator:
    def __init__(self, settings: dict[str, Any], frames, classifier, transformer, controller, approver, events):
        self.s = settings
        self.frames = frames
        self.classifier = classifier
        self.transformer = transformer
        self.ctl = controller
        self.approver = approver
        self.events = events
        self.assessor = Assessor(settings["classifier"])
        self.state = State.IDLE
        self.history: list[State] = [State.IDLE]

    def _go(self, s: State, **info) -> None:
        self.events.record("state", frm=self.state.value, to=s.value, **info)
        log.info("state %s -> %s %s", self.state.value, s.value, info or "")
        self.state = s
        self.history.append(s)

    def _end(self, s: State, msg: str, **kw) -> CycleResult:
        self._go(s, message=msg)
        return CycleResult(s, list(self.history), message=msg, **kw)

    async def inspect(self, n: Optional[int] = None) -> Assessment:
        n = n or int(self.s["classifier"]["min_observations"])
        inspections = []
        for _ in range(n):
            frame = await self.frames.get_frame()
            insp = self.classifier.inspect(frame)
            self.events.record("inspection", verdict=insp.verdict, confidence=round(insp.confidence, 3),
                               reasons=insp.reasons, n_objects=len(insp.objects))
            inspections.append(insp)
        return self.assessor.assess(inspections)

    async def run_cycle(self) -> CycleResult:
        try:
            return await self._run()
        except Exception as e:
            log.exception("cycle failed")
            if self.ctl is not None:
                await self.ctl.stop()
            return self._end(State.ERROR, f"{type(e).__name__}: {e}")

    async def _run(self) -> CycleResult:
        self._go(State.ENVIRONMENT_CHECK)
        await self.frames.get_frame()  # camera reachable; raises otherwise

        # The RealSense is wrist-mounted (parented to the arm), so the arm must
        # look at the plate before inspecting. Skipped if no "scan" pose is set
        # (e.g. the operator positioned the arm by hand).
        if not await self._go_scan_pose():
            return self._end(State.ABORTED, "operator declined move to scan pose")

        self._go(State.INSPECTION)
        a = await self.inspect()
        self._go(State.ASSESSMENT, verdict=a.verdict.value, confidence=round(a.confidence, 3), reasons=a.reasons)

        if a.verdict == Verdict.EMPTY:
            return self._end(State.COMPLETED, "build plate empty - nothing to remove", assessment=a)
        if a.needs_human or a.verdict == Verdict.UNCERTAIN:
            return self._end(State.HUMAN_REVIEW, "classification uncertain - human inspection required", assessment=a)
        if a.verdict == Verdict.FAILED and self.s["robot"]["failed_removal_mode"] != "grasp":
            return self._end(State.HUMAN_REVIEW, "failed print detected - automatic failed-print removal disabled",
                             assessment=a)
        dest = "success_bin" if a.verdict == Verdict.SUCCESS else "failed_bin"

        self._go(State.GRASP_PLANNING, destination=dest)
        plan = await self._plan_grasp(a)
        if not plan.feasible:
            return self._end(State.HUMAN_REVIEW, "no safe grasp: " + "; ".join(plan.reasons), assessment=a, grasp=plan)
        self.ctl.named_pose(dest)  # fail early if destination is not configured
        self.ctl.named_pose("home")

        self._go(State.OPERATOR_APPROVAL)
        details = (f"Verdict: {a.verdict.value} (conf {a.confidence:.2f})\n" + "\n".join(a.reasons) +
                   f"\nGrasp: {plan.reasons[0]}\nDestination: {dest}\nController live: {self.ctl.live}")
        if not await self.approver.approve("Execute pick-and-place?", details):
            return self._end(State.ABORTED, "operator declined", assessment=a, grasp=plan)

        self._go(State.RELEASE, mode=self.s["robot"]["release_mode"])
        if not await self.approver.approve("Is the print released from the build plate (cooled / plate flexed)?"):
            return self._end(State.ABORTED, "print not confirmed released", assessment=a, grasp=plan)

        self._go(State.PICK)
        await self.ctl.open_gripper()
        await self.ctl.move_gripper_to(plan.pre_grasp, label="pre_grasp")
        await self.ctl.move_gripper_to(plan.grasp, label="grasp")
        grabbed = await self.ctl.grab()
        holding = await self.ctl.is_holding()
        if not grabbed or holding is False:
            await self.ctl.open_gripper()
            await self.ctl.move_gripper_to(plan.pre_grasp, label="retreat")
            return self._end(State.HUMAN_REVIEW, "grasp did not secure the object", assessment=a, grasp=plan)
        await self.ctl.move_gripper_to(plan.lift, label="lift")

        self._go(State.PLACE, destination=dest)
        await self.ctl.go_named(dest)
        await self.ctl.open_gripper()
        await self.ctl.go_named("home")

        self._go(State.REMOVAL_VERIFICATION)
        if not await self._go_scan_pose():
            return self._end(State.HUMAN_REVIEW, "removal not verified (scan move declined)", assessment=a, grasp=plan)
        after = await self.inspect()
        if after.verdict != Verdict.EMPTY:
            return self._end(State.HUMAN_REVIEW, f"plate not clear after removal ({after.verdict.value})",
                             assessment=a, grasp=plan)
        return self._end(State.COMPLETED, f"{a.verdict.value} print moved to {dest}", assessment=a, grasp=plan)

    async def _go_scan_pose(self) -> bool:
        if self.ctl is None or "scan" not in self.s["robot"].get("named_poses", {}):
            return True
        if self.ctl.live and not await self.approver.approve("Move arm to scan pose?"):
            return False
        await self.ctl.go_named("scan")
        return True

    async def _plan_grasp(self, a: Assessment) -> GraspPlan:
        obj = a.primary_object
        if obj is None or obj.top_center_cam_mm is None:
            return GraspPlan(False, ["no 3D object estimate (depth missing?)"])
        c_cam = np.asarray(obj.top_center_cam_mm, dtype=np.float64)
        yaw = math.radians(obj.yaw_deg)
        axis_cam = np.array([math.cos(yaw), math.sin(yaw), 0.0]) * 20.0
        c_w = await self.transformer.to_world(c_cam)
        axis_w = (await self.transformer.to_world(c_cam + axis_cam)) - c_w
        return plan_top_down_grasp(c_w, axis_w, obj.footprint_mm, obj.max_height_mm,
                                   self.s["robot"]["gripper"], self.s["robot"]["approach_offset_mm"])
