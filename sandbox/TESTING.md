# Real-world test plan (xArm6 + wrist RealSense)

**Only one file goes to Viam:** `sandbox/print_inspector.py`. Paste its full contents into CONTROL → Sandbox → `main.py`, replacing everything that's there. Nothing else gets copied or uploaded. The machine config (arm, gripper, cam, obstacles) is already set up, so don't push anything from `config/` to it.

Run **one step per run** and don't skip ahead. After each run, set `"STEP": 0` again.

## Before you start (every session)

- [ ] Someone is standing at the **E-stop** (red button on the UFactory control box) and watching the arm, not the screen.
- [ ] Nothing is in the arm's path. Hands, laptops and cables are off the gold sheet and away from the bins.
- [ ] The gold sheet is on its box. The good bin and the failed bin are where they were when their positions were recorded.
- [ ] The Sandbox `main.py` contains the **latest** `print_inspector.py` from the `feat/project-foundation` branch.
- [ ] Manual mode is **off**: click *Exit manual mode* in the arm panel.

## Test 1: STEP 0, vision only (no motion)

1. Move the arm to the scan position (jog it, or put it there in manual mode). Put the **white dome** in the middle of the gold sheet.
2. In `MOTION`, check that `"STEP": 0` is set, then press **Run**.
3. **Pass if:**
   - `gripper pose (world)` shows about `x 103, y -600, z 367`. That confirms the 105 mm gripper offset. If it shows `z ≈ 471`, **stop and tell us**: the saved positions are then 105 mm off.
   - The `plate plane:` line covers 40% or more of the view, at about the sheet's distance from the camera, for example: `plate plane: 55% of view, 340 mm from camera`.
   - The console shows `object 0: GOOD`, `white=` above 0.35, a height close to the dome's real height, and `VERDICT: GOOD`.
   - All four positions are reported as `OK`.
4. Repeat with the **red dome**: expect `VERDICT: FAILED`.
5. Repeat with the **red-top/white cube**: expect FAILED or UNCERTAIN. Either is acceptable.
6. Repeat with an **empty sheet**: expect `VERDICT: EMPTY`.
7. **Copy all four console outputs** and send them to the team chat so we can tune the settings.

| Problem in the output | Fix in `CONFIG` |
|---|---|
| `no large plane (plate) found in view` | Make the sheet fill more of the view (lower the scan position), or lower `min_plane_frac`. |
| `too few points in view` | Lower the scan position (closer to the sheet), or raise `max_range_mm`. |
| White dome shows `UNCERTAIN` | Lower `white_v_min` (try 0.5) or raise `white_s_max` (try 0.3). |
| Red dome shows `UNCERTAIN` | Lower `red_s_min` (try 0.45). |
| The gripper fingers show up as an object | Raise `min_range_mm`. |

## Test 2: STEP 1, nudge up (first motion)

1. Set `"STEP": 1` and `"I_AM_AT_THE_ESTOP": True`, then press **Run**.
2. **Pass if:** the gripper rises about 30 mm **straight up**, nothing else moves, and `dz = +30` is printed (±3 mm is fine).
3. **Stop and tell us** if it twists, swings sideways, or moves more than a few centimetres.

## Test 3: STEP 5, home, then STEP 2, scan

1. Move the arm a little away from the scan position.
2. Set `"STEP": 5` and press **Run**. **Pass if:** it returns to the scan position (home is the same as scan).
3. Move it away again. Set `"STEP": 2` and press **Run**. **Pass if:** same result.

## Test 4: STEP 3, hover (no contact)

1. Put the white dome on the sheet and set `"STEP": 3`.
2. **Pass if:** the arm goes to scan, inspects (`VERDICT: GOOD`), then moves to about **10 cm above the dome**, centred over it.
3. Measure how far off-centre it is. If it's more than about 1 cm, **don't run STEP 4**. Send us the offset and its direction instead, because the camera position needs calibrating.

## Test 5: STEP 4, full pick and sort

1. White dome on the sheet. Set `"STEP": 4`. Keep a hand on the E-stop.
2. **Expected sequence:**
   - scan → inspect → gripper opens
   - hover → descends in a straight line → grabs
   - lifts → moves to the **good bin** → releases
   - back to scan → re-inspects → `RESULT: sorted`
3. Repeat with the **red dome**. It should go to the **failed bin**.
4. **Pass if:** both domes end up in the correct bin and the plate is reported `EMPTY` afterwards.
5. If the grab misses, the script prints `grasp failed`, opens the gripper and backs off. Note how far off it was, so we can adjust `grasp_depth_mm` or the position.

## Demo rehearsal

- Run Test 5 twice in a row for each colour.
- Also show the **UNCERTAIN** case: the cube, or something pink or orange. The arm stays still and asks for a human, which is a good safety point to demonstrate.
- Fallback if picking is unreliable: STEP 0 (vision) plus STEP 3 (hover) still shows detection, classification and targeting.

## What to send back after testing

- The console output for each test, or a screenshot.
- For any failure: the step number, what the arm did, and the `REFUSED:` or error message.
- If a position was re-recorded: the new values from the arm card.
