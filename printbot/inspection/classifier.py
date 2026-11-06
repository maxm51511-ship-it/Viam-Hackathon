"""Print-quality classification.

Two levels:

* ``PrintClassifier.inspect(frame)`` — one frame -> ``Inspection`` by fusing
  independent evidence sources: geometric heuristics (always), YOLO (if a
  trained model is loaded), reference comparison (if an STL/G-code is given).
* ``Assessor`` — aggregates several Inspections into an ``Assessment``. A
  defect is NEVER declared from a single observation, and anything short of a
  confident, consistent agreement becomes UNCERTAIN -> human inspection.

Safety bias: disagreements between evidence sources, missing depth, or
unexplained debris all reduce confidence below the action threshold.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Optional

from printbot.inspection.colour_rule import colour_vote
from printbot.inspection.reference import ExpectedPart, compare
from printbot.perception.objects import extract_objects
from printbot.perception.plate import PlateSegmenter
from printbot.types import (Assessment, Detection2D, Frame, Inspection, MaterialClass, ObjectEstimate,
                            Verdict)

log = logging.getLogger(__name__)

_DEFECT = {MaterialClass.SPAGHETTI, MaterialClass.LOOSE_FILAMENT, MaterialClass.OTHER_DEFECT}
NO_DEPTH_CONFIDENCE_CAP = 0.6


class PrintClassifier:
    def __init__(self, settings: dict[str, Any], yolo=None, expected: Optional[ExpectedPart] = None):
        self.cfg = settings["classifier"]
        self.obj_cfg = settings["objects"]
        self.segmenter = PlateSegmenter(settings["plate"])
        self.yolo = yolo
        self.expected = expected

    # -------------------------------------------------------------- heuristics
    def heuristic_vote(self, obj: ObjectEstimate) -> tuple[Verdict, float, str]:
        c = self.cfg
        if obj.n_fragments >= c["spaghetti_min_fragments"] or obj.solidity < c["spaghetti_max_solidity"]:
            return (Verdict.FAILED, 0.75,
                    f"stringy/fragmented material (solidity {obj.solidity:.2f}, {obj.n_fragments} fragments)")
        tall_enough = (obj.max_height_mm != obj.max_height_mm) or obj.max_height_mm >= c["success_min_height_mm"]
        if obj.solidity >= c["success_min_solidity"] and obj.n_fragments <= 2 and tall_enough:
            return Verdict.SUCCESS, 0.75, f"compact part (solidity {obj.solidity:.2f}, h {obj.max_height_mm:.1f} mm)"
        return Verdict.UNCERTAIN, 0.4, f"ambiguous shape (solidity {obj.solidity:.2f}, {obj.n_fragments} fragments)"

    def yolo_vote(self, dets: list[Detection2D]) -> Optional[tuple[Verdict, float, str]]:
        strong = [d for d in dets if d.confidence >= self.cfg["yolo_min_confidence"]]
        defects = [d for d in strong if d.label in _DEFECT]
        parts = [d for d in strong if d.label == MaterialClass.PRINTED_OBJECT]
        if defects:
            best = max(defects, key=lambda d: d.confidence)
            return Verdict.FAILED, best.confidence, f"YOLO: {best.label.value} ({best.confidence:.2f})"
        if parts:
            best = max(parts, key=lambda d: d.confidence)
            return Verdict.SUCCESS, best.confidence, f"YOLO: printed_object ({best.confidence:.2f})"
        return None

    # ------------------------------------------------------------------ main
    def inspect(self, frame: Frame) -> Inspection:
        seg = self.segmenter.segment(frame)
        reasons = list(seg.notes)
        if seg.plate_polygon is None:
            return Inspection(Verdict.UNCERTAIN, 0.0, reasons, [], [])

        objects = extract_objects(frame, seg, self.obj_cfg)
        dets: list[Detection2D] = []
        if self.yolo is not None:
            try:
                dets = self.yolo.predict(frame.bgr)
            except Exception as e:  # inference failure must not crash the cell
                reasons.append(f"YOLO inference failed: {e}")

        if not objects and not any(d.label in _DEFECT | {MaterialClass.PRINTED_OBJECT} for d in dets):
            conf = 0.9 if seg.used_depth else NO_DEPTH_CONFIDENCE_CAP
            return Inspection(Verdict.EMPTY, conf, reasons + ["no material above plate"], [], dets)

        if self.cfg.get("mode") == "colour_demo":
            return self._colour_demo(frame, seg, objects, dets, reasons)

        votes: list[tuple[Verdict, float, str]] = []
        if objects:
            primary = objects[0]
            votes.append(self.heuristic_vote(primary))
            debris = [o for o in objects[1:] if o.pixel_area > 0.05 * primary.pixel_area]
            if debris:
                votes.append((Verdict.FAILED, 0.6, f"{len(debris)} separate piece(s) of material besides main object"))
            if self.expected is not None:
                chk = compare(primary, self.expected)
                if chk is not None:
                    if chk.consistent:
                        votes.append((Verdict.SUCCESS, 0.7, f"dimensions match {self.expected.source}"))
                    else:
                        votes.append((Verdict.FAILED, 0.7, "; ".join(chk.reasons)))
        yv = self.yolo_vote(dets)
        if yv:
            votes.append(yv)

        verdict, conf = _fuse(votes)
        if not seg.used_depth:
            conf = min(conf, NO_DEPTH_CONFIDENCE_CAP)
        reasons += [v[2] for v in votes]
        return Inspection(verdict, conf, reasons, objects, dets)

    def _colour_demo(self, frame, seg, objects, dets, reasons) -> Inspection:
        """Demo rule: every object is labelled by colour. Any red object -> FAILED;
        any unclear object -> UNCERTAIN (never guess); all white -> SUCCESS."""
        votes = [colour_vote(frame.bgr, o, self.cfg["colour"]) for o in objects]
        reasons += [f"object {i}: {v[2]}" for i, v in enumerate(votes)]
        kinds = {v[0] for v in votes}
        if Verdict.UNCERTAIN in kinds:
            verdict, conf = Verdict.UNCERTAIN, 0.3
        elif Verdict.FAILED in kinds:
            verdict, conf = Verdict.FAILED, min(v[1] for v in votes if v[0] == Verdict.FAILED)
        else:
            verdict, conf = Verdict.SUCCESS, min(v[1] for v in votes)
        if not seg.used_depth:
            conf = min(conf, NO_DEPTH_CONFIDENCE_CAP)
        return Inspection(verdict, conf, reasons, objects, dets)


def _fuse(votes: list[tuple[Verdict, float, str]]) -> tuple[Verdict, float]:
    """Decisive votes must agree. FAILED evidence outranks SUCCESS only when
    nothing contradicts it; any contradiction -> UNCERTAIN."""
    decisive = [(v, c) for v, c, _ in votes if v in (Verdict.SUCCESS, Verdict.FAILED)]
    if not decisive:
        return Verdict.UNCERTAIN, 0.3
    kinds = {v for v, _ in decisive}
    if len(kinds) > 1:
        return Verdict.UNCERTAIN, 0.3
    v = kinds.pop()
    confs = [c for _, c in decisive]
    # independent agreeing sources raise confidence (noisy-OR), capped
    p_wrong = 1.0
    for c in confs:
        p_wrong *= (1 - c)
    return v, min(0.95, 1 - p_wrong)


class Assessor:
    """Aggregate per-frame inspections into one actionable decision."""

    def __init__(self, cfg: dict[str, Any]):
        self.min_obs = int(cfg["min_observations"])
        self.agreement = float(cfg["agreement_frac"])
        self.min_conf = float(cfg["min_confidence_to_act"])

    def assess(self, inspections: list[Inspection]) -> Assessment:
        n = len(inspections)
        if n < self.min_obs:
            return Assessment(Verdict.UNCERTAIN, 0.0, [f"only {n}/{self.min_obs} observations"], n, True)
        counts = Counter(i.verdict for i in inspections)
        verdict, k = counts.most_common(1)[0]
        frac = k / n
        agreeing = [i for i in inspections if i.verdict == verdict]
        conf = sum(i.confidence for i in agreeing) / len(agreeing) * frac
        reasons = [f"{k}/{n} observations agree on {verdict.value}"]
        reasons += sorted({r for i in agreeing for r in i.reasons})[:8]
        primary = next((i.objects[0] for i in reversed(agreeing) if i.objects), None)

        if frac < self.agreement:
            return Assessment(Verdict.UNCERTAIN, conf, reasons + ["observations inconsistent"], n, True, primary)
        needs_human = verdict == Verdict.UNCERTAIN or conf < self.min_conf
        if needs_human and verdict != Verdict.UNCERTAIN:
            reasons.append(f"confidence {conf:.2f} below action threshold {self.min_conf:.2f}")
        final = Verdict.UNCERTAIN if needs_human else verdict
        return Assessment(final, conf, reasons, n, needs_human, primary)
