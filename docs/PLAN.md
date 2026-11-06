# Technical assessment and implementation plan

_Last updated 2026-11-06. Labels: **[C]** confirmed · **[A]** assumption · **[P]** proposal · **[U]** unknown (needs someone to look at the hardware)._

## 1. Repository state at start

- **[C]** The remote `main` branch had a single commit (`5f0b558`, "Initial commit") containing only a 2-line README. There was no code, no config and no Viam integration, so nothing needed extending or preserving.
- **[C]** All foundation work is on the branch `feat/project-foundation`. Nothing is committed yet.

## 2. Hardware and Viam: feasibility

_Source: the "[Code That Moves] Viam Overview" PDF handed out at the event, plus the installed SDK. The machine lives in the **hackathons** org → **Arm Farm** location, and it is already configured._

| Item | Status | Notes |
|---|---|---|
| Arm `arm` | **[C]** `viam:ufactory:xArm6` (confirmed by the team) | Frame: parent `world`, at the origin. Reach is about 700 mm; our fence stops at 650 mm. |
| Gripper `gripper` | **[C]** `viam:ufactory:gripper`, names its arm | Frame: parent `arm`, z +105 mm (the PDF says "probably"). For motion planning, move the **gripper**. Jaw opening is **[U]**: measure it. |
| Camera `cam` | **[C]** `viam:camera:realsense`, D435 RGB + stereo depth | **Wrist-mounted.** Machine's calibrated frame (read from CONFIGURE, 2026-11-06): parent `arm`, translation (−83, 14, 18), ov (0.030, 0.004, 1.000) θ −97.7°. This is close to Viam's tutorial (−73, 40, 18) θ 270°. The PDF's (73, −40, 18) θ 90° is **wrong for this rig**. Good enough for pick and place; use `viam:camera-calibration:handeye` for more precision. Because the camera moves with the arm, inspection needs a **scan pose**. |
| Motion service | **[C]** builtin, plans around the obstacles | `Motion.move(component_name, destination, world_state, constraints)` (SDK 0.85.2). |
| Obstacles | **[C]** `table`, `wall-front`, `wall-side` and `ceiling` are preconfigured (`erh:vmodutils:obstacle`) | Direct `MoveToPosition` / `MoveToJointPositions` go straight through them, so we use the motion service only. |
| E-stop | **[C]** red button on the UFactory control box | Must be within reach for every motion test. |
| Sandbox | **[C]** CONTROL tab, Python, `VIAM_MACHINE_ADDRESS` / `VIAM_API_KEY` / `VIAM_API_KEY_ID` injected | We can't confirm numpy, OpenCV or Pillow are available there, so `sandbox/print_inspector.py` uses only the standard library. |
| Vision modules offered | **[C]** shape-finder + detections-to-segments, `viam-labs:vision:yolov8`, EfficientDet-COCO, hand-gesture, AprilTag | None of these knows "failed print" out of the box. |
| Existing services on our machine | **[C]** from the Sandbox template | Vision: `Classification`, `color-detector`, `color-cdbeb1`, `color-cca68c`. ML model: `Shape-Infeed`. Generic: `code-1`, `code-2`, owned by teammates and not touched by us. |
| Helpers | **[C]** manual mode, `erh:vmodutils:arm-position-saver`, 3D SCENE tab | Use these to record the scan, bin and home poses. |
| Data manager | **[C]** | Use it to collect training images later. |
| xArm force/torque | **[U]** | Not in the PDF. Don't assume it. |
| Printer | **[C]** none at the event | Demo stand-in: a textured gold PEI sheet lying on a box. Props are red and white domes and a cube that is red on top and white below. Bins are a black wire basket and a white paper plate. |

### Demo decisions (team, 2026-11-06)

- **Demo colour rule:** red material = FAILED, white material = GOOD, anything else = UNCERTAIN (a human checks it).
- **No real failed prints are available.** Red prints and red filament tangles stand in for failures.
- **Not using YOLO for the demo.** There's no training data for failures, and a pretrained model doesn't know spaghetti. Depth segmentation plus the colour rule is deterministic and needs no training. The YOLO path (`printbot/perception/yolo.py`, or the `viam-labs:vision:yolov8` module with a custom model) stays available for later.

## 3. Architecture (as built)

```
 RealSense ─┐                             ┌─ xArm + gripper
            │   viam-server (hardware PC) │
            └── camera ── frame system ── arm/gripper, motion, data_manager
                              ▲  gRPC (viam-sdk, local network or cloud)
   printbot (Python, runs ON the hardware PC so safety logic stays local)
   perception/  sources → plate (HSV + depth plane) → objects → yolo (optional)
   inspection/  classifier (heuristics ⊕ YOLO ⊕ STL/G-code) → Assessor (multi-frame)
   robot/       transforms (cam→world) · grasp · controller (motion.move, DRY-RUN default)
   app/         state machine · operator approval · JSONL event log
```

- **Decoupling:** modules talk only through the dataclasses in `printbot/types.py`. Perception has no robot dependency, and the robot code doesn't depend on OpenCV.
- **Safety defaults:**
  - Physical motion needs three interlocks: the `motion_enabled` setting, the `PRINTBOT_ALLOW_MOTION=1` environment variable, and the `--allow-motion` flag.
  - An operator approves every pick.
  - The operator also confirms that the print has been released from the plate.
  - Failed prints go to a human by default.
- **Cloud vs local [P]:**
  - Time-critical control and the safety logic run on the PC connected to the hardware.
  - The Viam cloud handles monitoring: the CONTROL tab, logs, and data sync.
  - Packaging `printbot` as a Viam module with a Generic service and `DoCommand` (`{"inspect":…}`, `{"run_cycle":…}`) is a later step. It lets the cell be triggered from the Viam app or a TypeScript dashboard without SSH. We haven't done it yet because a client script is faster to debug.

## 4. Vision approach

1. **Plate segmentation [C, implemented]:**
   - Configurable HSV ranges give the first estimate of the plate.
   - The largest region becomes the plate boundary.
   - A **RANSAC plane** is fitted to the exposed plate pixels.
   - Any pixel more than `object_min_height_mm` above that plane counts as material.
   - Colour only seeds the plane; depth decides what is material. Filament that is the same colour as the plate is still detected (there's a test for this).
   - Without depth, the pipeline falls back to colour only and caps confidence at 0.6. That is below the 0.7 action threshold, so a human is always asked.
2. **Heuristic classification [C, implemented]:**
   - Each blob gets shape descriptors: solidity, fragment count, height variation, and separate debris pieces.
   - Compact blobs vote SUCCESS; stringy or fragmented ones vote FAILED.
   - These rules have only been validated on **synthetic** scenes, so the thresholds need tuning on real captures.
3. **YOLO [P, wrapper implemented, no model yet]:**
   - **Recommendation: Ultralytics YOLO11n-seg**, fine-tuned on our dataset. It does instance masks, runs on CPU at an acceptable speed for a few frames per cycle, and installs with a single pip package.
   - A pretrained COCO model does **not** know spaghetti, so YOLO votes are ignored until we have trained weights.
   - Viam's hosted `yolo-onnx-training` script trains detection only, so segmentation training happens locally (`tools/train_yolo.py`).
4. **STL / G-code reference [partial]:**
   - Implemented: gross-dimension checks (footprint and height) against the STL or G-code.
   - Next: Open3D ICP of the observed point cloud against a sampled STL surface, using visibility-aware deviation (only faces the camera can see), aggregated over several frames.
   - A single fixed D435 sees only the top and front faces. Missing geometry must never be read as a defect from one view.
5. **Multi-frame assessment [C, implemented]:**
   - Needs at least 3 observations, at least 80% agreement, and confidence of at least 0.7.
   - Anything else becomes UNCERTAIN and goes to HUMAN_REVIEW.

## 5. Mechanical release and sensing (Part Two)

- **Vertical lifting alone will not reliably detach prints.** Adhesion to PEI, glass or textured plates is strongest while the plate is hot.
- **Lowest-risk strategy [P]:**
  1. Wait for the bed to cool to about 30 °C or below.
  2. If the plate is a removable flexible sheet, flex it. Flexing releases the print through shear and peel. A straight upward pull doesn't do that.
  3. Only then pick the part up.
  - `release_mode: "operator"` means a human confirms the release step until a mechanism has been validated.
- **"Raise or move the plate partway":** this is only feasible if the printer's bed can tilt, or if the flexible sheet can be gripped at one edge.
  - Lifting one edge of a magnetic spring-steel sheet with the gripper is a plausible way to peel the print off.
  - It is **[U]** until we have the printer in hand.
- **Force control:**
  - Use a probe-and-check loop with a hard force cap, a hard displacement cap, a time limit, and abort on anomaly.
  - Never apply force that keeps increasing until something gives.
  - The actual sensing method is **[U]**: xArm joint current/torque estimates, an optional F/T sensor, or gripper position feedback. None of it is implemented, by design.
- **Failed-print removal:**
  - Gripper grasping (`failed_removal_mode: "grasp"`) is reasonable for one compact clump.
  - Diffuse spaghetti needs a compliant sweeper end effector, which is hardware we don't have.
  - Sweeping is **not implemented**; default is `"human"`.
- **Grasp mechanics (Part Three B):**
  - FDM parts are anisotropic: weakest across layers and at thin walls.
  - An isotropic FEA model in Fusion overestimates strength.
  - In practice: grip across the short side, low in the part, on the solid perimeter, with the gripper's force or current limit set low.
  - FEA is optional. If we do it, it only produces a ranking of grip regions, using derated (lower-than-nominal) material properties.

## 6. Roadmap

### Smallest working demo (target)

1. The live D435 feed gives a correct SUCCESS / FAILED / EMPTY verdict on real prints.
2. The operator approves.
3. A human confirms the part is loose (the bed has cooled).
4. The gripper makes a top-down pick of a compact part and places it in the success bin.
5. A re-inspection confirms the plate is empty.
6. Failed prints are detected and flagged for a human.

### Part One: vision and inspection

| # | Task | Files | Status | Test / done when |
|---|---|---|---|---|
| 1.1 | Camera acquisition | `perception/sources.py`, `tools/check_machine.py`, `tools/capture_frames.py` | **Code done, needs hardware** | Colour and depth come back aligned. Intrinsics are not zero. |
| 1.2 | Plate segmentation | `perception/plate.py` | **Done (synthetic)** | Correct boundaries on 20 real frames under 2 lighting conditions. |
| 1.3 | Object measurement | `perception/objects.py` | **Done (synthetic)** | Measured height and footprint within ±2 mm of calipers on 5 parts. |
| 1.4 | Heuristic classifier and multi-frame assessor | `inspection/classifier.py` | **Done (synthetic)** | Fewer than 5% wrong automatic actions on a labelled real set. Everything else goes to UNCERTAIN. |
| 1.5 | Dataset and YOLO training | `datasets/`, `perception/yolo.py`, `tools/train_yolo.py` | **Scaffolded** | 150+ images. Mask mAP50 above 0.6 on spaghetti in the validation split. |
| 1.6 | STL/G-code comparison | `inspection/reference.py` | **Dimensions done; ICP next** | A truncated print is flagged on 3 of 3 sessions. |
| 1.7 | Point-cloud path | `perception/geometry.parse_pcd` | **Parser done** | Verify the PCD units on hardware and pin `pcd_units`. |

### Part Two: release and failed-print removal

| # | Task | Status | Blocked on |
|---|---|---|---|
| 2.1 | Identify the printer, plate type and bed temperature access | Not started | Hardware **[U]** |
| 2.2 | Pick a release strategy (cool + flex / peel by edge lift) | Not started | 2.1 |
| 2.3 | Characterise the force/torque feedback available | Not started | Arm model and options |
| 2.4 | Separation check (re-inspect after nudging; part moved = released) | Not started | 2.2 |
| 2.5 | Failed-print removal by grasp | Code path exists, disabled | Real failure captures |

### Part Three: grasp, pick and sort

| # | Task | Files | Status | Done when |
|---|---|---|---|---|
| 3.1 | Machine config with frames | `config/machine.example.json` | Draft | `check_machine.py` lists all resources and frames. |
| 3.2 | Camera extrinsics calibration | `tools/` (next) | Not started | Touching 4 plate corners with the gripper tip gives error under 5 mm. |
| 3.3 | Named poses and obstacles (printer box, bins) | `config/settings.json` | Not started | Plans succeed against a **fake arm** first. |
| 3.4 | Top-down grasp planner | `robot/grasp.py` | **Done (unit tests)** | Dry-run poses look correct when viewed in the Viam app. |
| 3.5 | Orchestrated pick and place | `app/state_machine.py`, `tools/run_cycle.py` | **Done (dry-run)** | Live run with the E-stop held, slow speed, a sacrificial print. |
| 3.6 | Package as a Viam module (Generic + DoCommand) | `module/` (later) | Not started | Cycle can be triggered from the Viam app. |

## 7. Next concrete task

**Now:** run `sandbox/print_inspector.py` in the Viam Sandbox (see `sandbox/README.md`) on an empty plate, a white print and a red print. Then record the arm pose it prints as `robot.named_poses.scan`.

**Later:**

Bring up `viam-server` with the RealSense and run `tools/check_machine.py`. It tells us whether depth is aligned to colour, the image source names, and whether the intrinsics are valid. Then run `tools/capture_frames.py` on an empty plate, a good print and a failed print. Those captures go straight into tuning `plate.hsv_ranges` and the classifier thresholds with `tools/inspect_print.py --frames`.

## 8. Cross-check against the Viam pick-and-place tutorial (2026-11-06)

Source: docs.viam.com/tutorials/pick-and-place (same hardware: xArm6, wrist D435, UFactory gripper).

| Topic | Tutorial | Ours | Status |
|---|---|---|---|
| Component passed to `motion.move` | `gripper-1` | `gripper` | Same |
| Obstacles | Configured `vmodutils/obstacle` components, no WorldState in code | Same; `extra_obstacles = []` | Same |
| Detect from a fixed observation pose | Always returns to home before detecting | Scan pose before every inspection | Same |
| Gripper frame to grasp point | Gripper frame is 60 mm above where the fingers close (`GRIPPER_LENGTH_MM`) | **Was missing.** Grasp treated the gripper frame as the fingertips, about 60 mm too low. | **Fixed:** `fingertip_offset_mm = 60`, and the fence now checks the fingertips |
| Settle after open/grab | 0.3 s sleeps | Was missing | **Fixed:** `settle_s = 0.3` |
| Holding check | Not used | Aborted unless both `grab()` and `is_holding_something` were true | **Relaxed:** aborts only if both say nothing was grabbed |
| Camera `align_color_depth: true` | Required | Not checked | **Check on the machine** (`sandbox/TESTING.md`) |
| Camera frame | (-73, 40, 18) θ270 | PDF says (73, -40, 18) θ90 | **Mismatch, check on the machine.** Our code uses whatever the machine config says, through `transform_pose`. |
| Object 3D position | `detections-to-segments` vision service, `geometry.center` in the cam frame | Our own point-cloud segmentation; top-surface centre converted to world | Different by design (no model or training needed); equivalent result |
| Approach and descend | Move in the `cam-1` frame, then a relative descend in the gripper frame | Absolute world-frame poses computed once at scan time | Equivalent; ours doesn't depend on the camera having moved |
| Saved poses | `arm-position-saver` joint replay (no planner) | `motion.move` to recorded poses (collision-aware) | Ours is safer; joint replay is a fallback if planning is slow |
| Descend / lift | Unconstrained | Linear constraint | Stricter. If planning fails, try with `linear=False`. |
| Deployment | Local script (uv), then an inline module with `do_command({"action": "pick_cycle"})` | Sandbox script | The inline module is the documented next step. |

### Companion repo check (viam-devrel/pick-and-place: reference-solution.py, module-reference.py, machine-fragment.json)

- **Imports:** all at the top of the file in every reference script. Our lazy imports caused the Sandbox `ResourceNotFoundError('motion', 'builtin')` (fixed in `89bb141`).
- **Constants:** `GRIPPER_LENGTH_MM = -60`, `APPROACH_MM = -100`, `SETTLE_S = 0.3`. These match our `fingertip_offset_mm`, `approach_above_mm` and `settle_s`.
- **Grasp height:** the reference places the fingertips at `geometry.center`, the object's centre. We now close at **mid-height** (`grasp_depth_frac = 0.5`) instead of 10 mm below the top.
- **`motion.move` calls:** two per pick, with `component_name` set to the gripper, and no `world_state` or constraints. Ours has the same, plus an optional straight-line constraint.
- **The reference has none of these:** a holding check, try/except, or a workspace fence. Ours adds all three as extra safety.
- **`cam-1` in `machine-fragment.json`:** `align_color_depth: true`; frame `(-73, 40, 18)` θ270, parent `arm-1`. Check what our machine has (`sandbox/TESTING.md`).
- **`arm-1` in `machine-fragment.json`:** `speed_degs_per_sec: 30`.
