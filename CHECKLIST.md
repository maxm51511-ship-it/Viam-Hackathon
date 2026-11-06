# Development checklist

_Validation: (S) = synthetic or unit tests only · (H) = validated on hardware. Nothing has been validated on hardware yet._

## Completed

- [x] Repository audit and plan (`docs/PLAN.md`)
- [x] Python environment and pinned core dependencies (`viam-sdk` 0.85.2 installs on Windows / Python 3.14)
- [x] Shared data contracts (`printbot/types.py`)
- [x] Frame acquisition: live Viam camera and file replay (code only; live path not run)
- [x] Plate segmentation, HSV + depth plane (S)
- [x] Object measurement: height, footprint, shape descriptors (S)
- [x] PCD parser, ASCII + binary (S)
- [x] Heuristic classifier and multi-frame assessor with human fallback (S)
- [x] STL and G-code expected-dimension check (S)
- [x] YOLO-seg inference wrapper and dataset scaffold (no model trained)
- [x] Top-down grasp planner with constraint checks (S)
- [x] Robot controller over `motion.move` on the gripper, DRY-RUN by default (S)
- [x] Orchestrator state machine, operator approval and event log (S, dry-run)
- [x] Sandbox vision script, standard library only, red/white demo rule (S)
- [x] `colour_demo` classifier mode in printbot (S)
- [x] Scan pose for the wrist camera in the orchestrator (S, dry-run)
- [x] Hardware and frames confirmed from the Viam overview PDF
- [x] Tools: `check_machine`, `capture_frames`, `inspect_print`, `run_cycle`, `train_yolo`

## Blocked on hardware information

- [ ] Which arm model is on our machine (xArm6 or xArm850): check the CONFIGURE tab
- [ ] Gripper jaw opening and finger length → `robot.gripper` in settings
- [ ] Printer model, plate type (flexible sheet?), bed temperature access
- [ ] Force/torque feedback available on our xArm

## Awaiting hardware validation

- [ ] **Run `sandbox/print_inspector.py` in the Sandbox: empty, white and red prints** (next step)
- [ ] Record poses: scan, home, good bin, failed bin (arm-position-saver or manual mode)

- [ ] Live camera: colour and depth aligned, source names, intrinsics, PCD units
- [ ] HSV profile and classifier thresholds tuned on real captures
- [ ] Camera extrinsics calibration (frame system)
- [ ] Named poses (home, bins) and obstacles; plans validated against a fake arm first
- [ ] First live pick, slow speed, E-stop in hand

## Remaining implementation

- [ ] Calibration tool (touch-off of the plate corners)
- [ ] YOLO dataset collection, annotation and training
- [ ] ICP registration against the STL (Open3D)
- [ ] Release mechanism and separation detection (after the printer is identified)
- [ ] Failed-print removal: grasp mode validation; sweeper is hardware-dependent
- [ ] Optional: package as a Viam module (Generic + DoCommand), remote dashboard
- [ ] Stretch: gesture control (non-motion commands only)
