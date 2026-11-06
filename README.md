# Viam-Hackathon

Viam arm + 3D printer: autonomous print inspection, classification and removal.

A UFactory xArm and an Intel RealSense D435 run on [Viam](https://www.viam.com). The cell:

1. Inspects the finished print with colour and depth.
2. Classifies it as **success**, **failed** (spaghetti or loose filament), or **uncertain**.
3. After operator approval, picks the part up and sorts it into a bin.

When it's uncertain, it hands the print to a human instead of acting.

> **Status:** the software foundation runs and its tests pass on synthetic data. **Nothing has been run on hardware yet.** Robot motion is dry-run by default.

## Try the vision in the Viam Sandbox

Follow [`sandbox/README.md`](sandbox/README.md): paste `sandbox/print_inspector.py` into CONTROL → Sandbox and press Run. Demo rule: **white = good, red = failed**, anything else is sent to a human.

## Quick start

```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python tools\inspect_print.py --demo     # writes annotated images to logs/inspect
```

To connect to a machine:

1. Copy `.env.example` to `.env` and fill in the machine address and API key.
2. Run these:

```
.venv\Scripts\python tools\check_machine.py            # read-only: resources, frames, camera
.venv\Scripts\python tools\capture_frames.py --n 10 --tag empty_plate
.venv\Scripts\python tools\inspect_print.py --frames data\captures
.venv\Scripts\python tools\run_cycle.py                # live camera, DRY-RUN robot, asks for approval
```

## Layout

| Path | Contents |
|---|---|
| `printbot/perception` | Camera acquisition, plate segmentation, depth geometry, YOLO |
| `printbot/inspection` | Classifier, multi-frame assessor, STL/G-code comparison |
| `printbot/robot` | Viam connection, camera→world transforms, grasp planning, motion controller |
| `printbot/app` | State machine, operator approval, event log |
| `config/` | `settings.json` (tuning) and `machine.example.json` (Viam config draft) |
| `datasets/print_defects` | YOLO segmentation dataset layout and labelling guide |
| `docs/PLAN.md` | Feasibility, architecture, roadmap |
| `CHECKLIST.md` | What's done, what's blocked, what's left |

## Safety

Physical motion needs **all** of these:

1. `robot.motion_enabled: true` in the settings
2. `PRINTBOT_ALLOW_MOTION=1` set in the environment
3. The `--allow-motion` flag
4. Operator approval at the prompt

Validate plans against a simulated (fake) arm first, and keep the physical E-stop within reach.
