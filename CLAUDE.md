# CLAUDE.md: Viam 3D-print inspection and removal cell

## Objective

1. Inspect the completed 3D print on the plate.
2. Classify it as SUCCESS, FAILED or UNCERTAIN.
3. For a successful print: release it, pick it up and drop it in the success bin.
4. For a failed print: send it to the failed bin, or hand it to a human.
5. Whenever the system is uncertain, a human inspects. The system never guesses.

## Hardware

Source: the Viam overview PDF. Full detail in `docs/PLAN.md` §2.

- **Machine:** in the hackathons org, Arm Farm location, already configured. Resource names are `arm`, `gripper`, `cam`, plus the builtin motion service and table/wall obstacles.
- **Arm:** `viam:ufactory:xArm6` (confirmed), at the world origin.
- **Gripper:** `viam:ufactory:gripper`. Frame is parent `arm`, z +105 mm. Motion planning moves the **gripper**.
- **Camera:** `cam`, `viam:camera:realsense` D435. **Wrist-mounted:** parent `arm`. The machine's calibrated frame is translation (−83, 14, 18), θ −97.7 (the PDF's (73, −40, 18) θ 90 is wrong for this rig). Inspection needs a scan pose. Inspection therefore needs a scan pose.
- **E-stop:** red button on the UFactory control box.
- **No printer at the event:** the demo plate is a PEI sheet on a box. Props are red and white domes and a red-top/white cube. Bins are a wire basket and a paper plate.
- **Unknown [U]:** gripper jaw opening, force/torque sensing.
- **Demo rule:** red = failed, white = good, other = uncertain. No real failed prints; red items stand in for them.
- **Dev machine:** Windows 11, Python 3.14.2. No `viam` CLI or `viam-server` installed (we don't need them; the machine is hosted).

## Software

- **Language:** Python. `viam-sdk` 0.85.2, `numpy`, `opencv-python`, `pytest`. Optional `ultralytics` (`requirements-ml.txt`, not yet installed).
- **Code layout** (in `printbot/`):
  - `types.py`: the contracts between modules
  - `perception/`: camera acquisition, segmentation, measurement
  - `inspection/`: classification
  - `robot/`: arm and gripper control
  - `app/`: orchestration
- **Command-line tools** are in `tools/`.
- **Settings:** `config/settings.json` holds settings, merged over `printbot/config.py:DEFAULTS`.
- **Secrets:** environment variables only, from `.env` (see `.env.example`).

## Conventions

- Units are mm and degrees, which matches Viam's Python protobuf API (joint positions are in degrees in Python).
- Camera-frame vs world-frame conversion happens **only** in `printbot/robot/transforms.py`.
- Move the **gripper** through `motion.move`. Never call `arm.move_to_position` directly, because it ignores obstacles.
- Motion is DRY-RUN unless all three are set: `robot.motion_enabled`, `PRINTBOT_ALLOW_MOTION=1`, and `--allow-motion`.
- Don't classify a print as defective from one frame. That is the `Assessor` `min_observations` rule.
- Verify Viam API calls against the installed SDK (`inspect.signature`) or the viam skills. Never invent model names or attributes. Mark unknowns `VERIFY` / **[U]**.
- Workstream branches: `ws1/vision`, `ws2/release`, `ws3/manipulation`, plus `feat/*` for shared work. Never commit or push without the team's approval.

## Relevant Viam skills

Install with:

```
/plugin marketplace add viam-devrel/agent-skills
/plugin install viam-skills@viam-agent-skills
```

| Skill | Use it for |
|---|---|
| `viam-python` | The Python SDK |
| `viam-go-motion-vision` | Frame system, motion and WorldState concepts |
| `viam-modules-fleet` | Config schema, CLI, modules |
| `viam-ml` | Data capture and YOLO training |
| `local-viam-server` | Bringing up a machine and `viam-server` |
| `viam-machine-config` | Pushing a config. It **replaces** the whole config. |

## Viam Sandbox

`sandbox/print_inspector.py` is a standalone, standard-library-only copy of the vision pipeline. It analyses the point cloud and the red/white rule, and never moves the arm. Paste it into CONTROL → Sandbox. Keep it in sync with `printbot` by hand; `tests/test_sandbox.py` covers it.

## Install and test

```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest                   # 51 tests, synthetic data, no hardware
.venv\Scripts\python tools\inspect_print.py --demo
copy .env.example .env                           # fill in machine address + API key
.venv\Scripts\python tools\check_machine.py      # read-only
```

## Milestones and open issues

See `CHECKLIST.md` and `docs/PLAN.md` §6.
