# Testing vision in the Viam Sandbox

The script `print_inspector.py` runs **vision only**. It reads the arm's position and the camera, and it **never sends a motion command**.

## Steps

1. Go to app.viam.com → **hackathons** org → **Arm Farm** location → your machine → **CONTROL** tab → **Sandbox** → **Python**.
2. Point the camera at the plate. `cam` is mounted on the arm's wrist, so first move the arm so the camera looks **straight down at the build plate from 25–45 cm** above it. Use either:
   - the arm control card, or
   - manual mode: arm **Configure** panel → *Enter manual mode* → move it by hand → *Exit manual mode*.

   Keep the E-stop on the UFactory control box in reach.
3. Replace everything in `main.py` with the contents of `print_inspector.py`, then press **Run**.
   - Our `connect()` reads `VIAM_MACHINE_ADDRESS`, `VIAM_API_KEY` and `VIAM_API_KEY_ID`, the same variables as the Sandbox template, so you don't need to change anything.
4. Read the console:

```
arm end position (record as scan pose): x=... y=... z=... o=(...) theta=...
--- capture 1/3 (...)
    plate plane: 62% of view, 352 mm from camera
    object 0: GOOD      red=0.00 white=0.81 h=14.8mm extent=(31.2, 20.4)mm cam=(...)
...
===== VERDICT: GOOD  (1 object(s): GOOD)
    object 0 top centre in world frame: (412.3, -35.0, 18.9) mm
```

At the end, the script also prints what the machine's existing vision services (`color-detector`, `color-cdbeb1`, `color-cca68c`, `Classification`) report on the same camera. This is read-only, so you can compare them with our verdict.

Copy the `arm end position` line. It becomes the `scan` pose in `config/settings.json`.

## Demo colour rule

| Material on the plate | Verdict |
|---|---|
| White print | **GOOD**, goes to the good bin |
| Red print, or red filament tangle | **FAILED**, goes to the failed bin |
| Any other colour, a mix, or changing between captures | **UNCERTAIN**, a human checks it |
| Nothing above the plate | **EMPTY** |

We don't have real failed prints, so stage these instead:
- a red printed part
- a loose ball of red filament scrap pulled into "spaghetti"

**Depth decides what counts as material, and colour only labels it.** That's why the plate colour doesn't matter, as long as the plate is the closest large flat surface the camera sees.

## Tuning (top of the script, `CONFIG`)

| Symptom | Change |
|---|---|
| "too few points in view" | Move the camera closer, or raise `max_range_mm`. |
| "no large plane (plate) found" | Make the plate fill more of the view, or lower `min_plane_frac`. |
| The gripper fingers are detected as an object | Raise `min_range_mm`. |
| White print says UNCERTAIN | Lower `white_v_min` (dim light) or raise `white_s_max`. |
| Red print says UNCERTAIN | Lower `red_s_min` (but no lower than 0.4, or pink and orange start counting as red). |
| Thin prints are not detected | Lower `min_height_mm` (about 2 mm is the floor because of D435 depth noise). |

## Photos that will help

Please send these if you can:
1. An empty plate, a white print and a red print, each taken from the scan pose.
   - **Best:** copy the console output for each one.
   - **Also useful:** a screenshot of the `cam` feed in the CONTROL tab.
2. A phone photo of the printer and arm, showing where the plate is relative to the arm base.
3. If a laptop is connected (`.env` set up), run `python tools/capture_frames.py --tag white_print --n 5`. That saves colour + depth so we can tune the full pipeline offline.

## Movement, one step at a time

Movement is controlled by the `MOTION` block near the bottom of the script. Each run does exactly one step. Only move on to the next step once the previous one behaved correctly. For every step from 1 up, someone stands at the E-stop and you set `"I_AM_AT_THE_ESTOP": True`.

| `STEP` | What it does | What it needs | Check before moving on |
|---|---|---|---|
| 0 (default) | Vision plus a read-only report. Prints `gripper pose (world): {...}`. **No motion.** | nothing | The printed pose matches where the arm really is. |
| 1 | Moves the gripper straight **up 30 mm** | E-stop flag | It moved up by about 30 mm and nothing else happened. |
| 2 | Moves to `scan_pose` | `scan_pose` | The camera ends up looking straight down at the plate. |
| 3 | Scan, inspect, then **hover** 100 mm above the print. Nothing touches the print. | `extra_obstacles` (the printer; `[]` if none) | The gripper is centred over the print. |
| 4 | Full cycle: open, hover, descend, grab, lift, drop in the good or failed bin, return to scan, check the plate is empty | `good_bin_pose`, `failed_bin_pose` | |

**Recording a pose** (scan pose, good bin, failed bin):
1. Move the arm to the position, using the control card or manual mode.
2. Run `STEP 0`.
3. Copy the printed `gripper pose (world)` dict into `MOTION`. For example:
   `"scan_pose": {"x": 300.0, "y": 0.0, "z": 400.0, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": 0.0},`

**Safety built in:**
- Every move goes through the motion service, which plans around the table, wall and ceiling obstacles. The script never makes a direct arm move.
- A software fence limits the gripper to z 5–700 mm and a reach of 150–650 mm.
- All targets for a pick are checked against the fence **before** the first move of that pick.
- Descending and lifting follow a straight line.
- Any failed move triggers `stop_all()`.
- UNCERTAIN or EMPTY means no pick.
- A failed grab opens the gripper and backs off.

**Not done yet:** the gripper isn't rotated to match the print's orientation. Place prints so their short side is across the jaws.
