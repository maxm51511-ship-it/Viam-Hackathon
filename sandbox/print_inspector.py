"""Viam CONTROL-tab Sandbox: 3D-print inspector + step-by-step pick and sort.

Paste this whole file into the Sandbox editor (Python) and press Run.

MOTION IS OFF BY DEFAULT. Set MOTION["STEP"] (further down) to choose what a run does:
    STEP 0  vision + read-only report (gripper pose, pose checks).  NO MOTION.
    STEP 1  nudge the gripper straight UP 30 mm (first motion test).
    STEP 2  move to the recorded scan pose.
    STEP 3  scan, inspect, then HOVER above the detected print (no contact).
    STEP 4  full cycle: scan, inspect, pick, drop in good/failed bin, back to scan,
            verify plate is empty, then go to home_pose (if set).
    STEP 5  go to home_pose (the arm's safe resting position).
Poses are absolute positions in the WORLD frame, whose origin (0,0,0) is the
base of the arm, so the arm can always return to a recorded pose.
Every STEP >= 1 also needs MOTION["I_AM_AT_THE_ESTOP"] = True. Only advance a
step after the previous one behaved correctly. All moves go through the motion
service, which plans around the table/wall/ceiling obstacles - never direct arm
moves (those go straight through obstacles).

Demo colour rule:
    RED   material on the plate -> FAILED print  (stand-in for spaghetti/defects)
    WHITE material on the plate -> GOOD print
    anything else / mixed       -> UNCERTAIN -> ask a human, no motion

Vision (pure standard library - no numpy/OpenCV/Pillow needed):
  1. camera.get_point_cloud() -> PCD bytes with x, y, z AND rgb per point.
  2. Find the plate = the closest large plane in the central view (RANSAC).
  3. Points more than min_height_mm above that plane = material on the plate.
     (Depth, not colour, decides what is an object - works with any plate colour.)
  4. Group those points into objects on a grid; classify each by colour.
  5. Repeat for N captures and require agreement before reporting a verdict.
  6. Convert object centres camera -> world with robot.transform_pose.
"""

import asyncio
import colorsys
import math
import os
import random
import struct
import time

# ----------------------------------------------------------------- settings
CONFIG = {
    "camera": "cam",              # names from the hackathon machine config
    "arm": "arm",
    "world_frame": "world",
    "n_captures": 3,              # observations required for a verdict
    "min_range_mm": 120,          # ignore points closer than this (gripper fingers in view)
    "max_range_mm": 800,          # ignore points farther than this
    "roi_tan": 0.45,              # only use the central view: |x/z|,|y/z| < roi_tan
    "min_height_mm": 3.0,         # height above plate that counts as material
    "max_height_mm": 200.0,
    "ransac_iters": 150,
    "ransac_inlier_mm": 2.0,
    "min_plane_frac": 0.20,       # a plane must cover this much of the view to be the plate
    "grid_mm": 4.0,               # clustering cell size
    "min_object_points": 60,
    "point_stride": 2,            # use every Nth point (speed)
    # colour rule (HSV, h in degrees 0-360, s/v 0-1)
    "red_hue_max": 20, "red_hue_min": 330, "red_s_min": 0.50, "red_v_min": 0.20,
    "white_s_max": 0.25, "white_v_min": 0.60,
    "colour_frac": 0.35,          # a class needs >= this share of the object's points
    "colour_dominance": 2.0,      # ...and this many times more points than the other class
    # Read-only cross-check: vision services already on the machine (from the
    # Sandbox template's resource list). Missing ones are skipped.
    "compare_vision_services": ["color-detector", "color-cdbeb1", "color-cca68c", "Classification"],
}


# ----------------------------------------------------------------- PCD parse
_PCD_FMT = {("F", 4): "f", ("F", 8): "d", ("I", 1): "b", ("I", 2): "h", ("I", 4): "i",
            ("U", 1): "B", ("U", 2): "H", ("U", 4): "I"}


def parse_pcd(data, stride=1):
    """PCD bytes -> list of (x, y, z, r, g, b) with xyz in mm. ASCII or binary."""
    pos, header = 0, {}
    while True:
        nl = data.index(b"\n", pos)
        line = data[pos:nl].decode("ascii", "replace").strip()
        pos = nl + 1
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        header[key.upper()] = vals
        if key.upper() == "DATA":
            break
    fields = header["FIELDS"]
    n = int(header["POINTS"][0])
    enc = header["DATA"][0].lower()
    ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
    irgb = fields.index("rgb") if "rgb" in fields else None

    rows = []
    if enc == "binary":
        fmt = "<" + "".join(_PCD_FMT[(t, int(s))] for t, s in zip(header["TYPE"], header["SIZE"]))
        rec = struct.calcsize(fmt)
        body = data[pos:pos + rec * n]
        for i, row in enumerate(struct.iter_unpack(fmt, body)):
            if i % stride == 0:
                rows.append(row)
        rgb_types = header["TYPE"][irgb] if irgb is not None else None
    elif enc == "ascii":
        for i, line in enumerate(data[pos:].decode("ascii").splitlines()):
            if i % stride == 0 and line.strip():
                rows.append([float(v) for v in line.split()])
        rgb_types = "A"
    else:
        raise ValueError("unsupported PCD encoding: " + enc)

    zs = sorted(abs(r[iz]) for r in rows[:: max(1, len(rows) // 500)] if r[iz] == r[iz])
    scale = 1000.0 if zs and zs[len(zs) // 2] < 10 else 1.0   # metres -> mm if needed
    pts = []
    for r in rows:
        x, y, z = r[ix] * scale, r[iy] * scale, r[iz] * scale
        if not (z == z) or z == 0:        # NaN / invalid
            continue
        if irgb is None:
            c = 0
        elif rgb_types == "F":            # packed into a float32
            c = struct.unpack("<I", struct.pack("<f", r[irgb]))[0]
        else:
            c = int(r[irgb])
        pts.append((x, y, z, (c >> 16) & 255, (c >> 8) & 255, c & 255))
    return pts


# ----------------------------------------------------------------- geometry
def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    m = math.sqrt(_dot(a, a))
    return (a[0] / m, a[1] / m, a[2] / m) if m > 1e-9 else None


def fit_plane(points, iters, inlier_mm, seed=0):
    """RANSAC plane (n, d) with n . p + d = 0, normal oriented toward the camera,
    so signed distance > 0 means 'above the plate'."""
    rng = random.Random(seed)
    sample = points if len(points) <= 3000 else rng.sample(points, 3000)
    best, best_count = None, -1
    for _ in range(iters):
        a, b, c = rng.sample(sample, 3)
        n = _norm(_cross(_sub(b, a), _sub(c, a)))
        if n is None:
            continue
        d = -_dot(n, a)
        count = sum(1 for p in sample if abs(_dot(n, p) + d) < inlier_mm)
        if count > best_count:
            best, best_count = (n, d), count
    if best is None:
        return None
    n, d = best
    if d < 0:                                  # camera origin must be on the + side
        n, d = (-n[0], -n[1], -n[2]), -d
    return n, d, best_count / len(sample)


def find_plate_plane(points, cfg):
    """The table/floor is usually visible BEHIND the plate, and the biggest plane
    is not necessarily the plate. Extract up to 3 planes and return the one
    CLOSEST to the camera that still covers >= min_plane_frac of the view."""
    remaining, planes, total = list(points), [], len(points)
    for k in range(3):
        if len(remaining) < 300:
            break
        fit = fit_plane(remaining, cfg["ransac_iters"], cfg["ransac_inlier_mm"], seed=k)
        if fit is None:
            break
        n, d, _ = fit
        inl = [abs(_dot(n, p) + d) < cfg["ransac_inlier_mm"] for p in remaining]
        frac = sum(inl) / total
        if frac < cfg["min_plane_frac"]:
            break
        planes.append((n, d, frac))
        remaining = [p for p, i in zip(remaining, inl) if not i]
    return min(planes, key=lambda pl: pl[1]) if planes else None


def colour_class(r, g, b, cfg):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h *= 360
    if s >= cfg["red_s_min"] and v >= cfg["red_v_min"] and (h <= cfg["red_hue_max"] or h >= cfg["red_hue_min"]):
        return "red"
    if s <= cfg["white_s_max"] and v >= cfg["white_v_min"]:
        return "white"
    return "other"


def analyze(pcd_bytes, cfg=CONFIG):
    """One observation. Returns dict(plate_found, objects=[...], notes=[...])."""
    pts = parse_pcd(pcd_bytes, cfg["point_stride"])
    notes = [f"{len(pts)} valid points"]
    roi = [p for p in pts if cfg["min_range_mm"] <= p[2] <= cfg["max_range_mm"]
           and abs(p[0] / p[2]) < cfg["roi_tan"] and abs(p[1] / p[2]) < cfg["roi_tan"]]
    if len(roi) < 500:
        return {"plate_found": False, "objects": [], "notes": notes + ["too few points in view (range/ROI)"]}

    fit = find_plate_plane([p[:3] for p in roi], cfg)
    if fit is None:
        return {"plate_found": False, "objects": [], "notes": notes + ["no large plane (plate) found in view"]}
    n, d, inlier_frac = fit
    notes.append(f"plate plane: {inlier_frac:.0%} of view, {d:.0f} mm from camera")

    # in-plane basis for clustering
    ref = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = _norm(_cross(n, ref))
    v = _cross(n, u)

    cells = {}
    for p in roi:
        h = _dot(n, p[:3]) + d
        if cfg["min_height_mm"] < h < cfg["max_height_mm"]:
            key = (int(math.floor(_dot(u, p[:3]) / cfg["grid_mm"])), int(math.floor(_dot(v, p[:3]) / cfg["grid_mm"])))
            cells.setdefault(key, []).append((p, h))

    # connected components over occupied grid cells (8-neighbourhood)
    seen, objects = set(), []
    for start in cells:
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            c = stack.pop()
            members.extend(cells[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) < cfg["min_object_points"]:
            continue
        counts = {"red": 0, "white": 0, "other": 0}
        for p, _ in members:
            counts[colour_class(p[3], p[4], p[5], cfg)] += 1
        total = len(members)
        red, white = counts["red"] / total, counts["white"] / total
        if red >= cfg["colour_frac"] and counts["red"] >= cfg["colour_dominance"] * counts["white"]:
            verdict = "FAILED"
        elif white >= cfg["colour_frac"] and counts["white"] >= cfg["colour_dominance"] * counts["red"]:
            verdict = "GOOD"
        else:
            verdict = "UNCERTAIN"
        heights = sorted(h for _, h in members)
        top = [p for p, h in members if h >= 0.8 * heights[-1]]
        cx = sum(p[0] for p in top) / len(top)
        cy = sum(p[1] for p in top) / len(top)
        cz = sum(p[2] for p in top) / len(top)
        us = [_dot(u, p[:3]) for p, _ in members]
        vs = [_dot(v, p[:3]) for p, _ in members]
        objects.append({
            "verdict": verdict,
            "points": total,
            "red_frac": round(red, 2), "white_frac": round(white, 2),
            "height_mm": round(heights[int(0.98 * (total - 1))], 1),
            "extent_mm": (round(max(us) - min(us), 1), round(max(vs) - min(vs), 1)),
            "top_center_cam_mm": (round(cx, 1), round(cy, 1), round(cz, 1)),
        })
    objects.sort(key=lambda o: o["points"], reverse=True)
    return {"plate_found": True, "objects": objects, "notes": notes}


def decide(observations, n_required):
    """Combine several observations: verdict per object rank must agree in all."""
    if len(observations) < n_required or not all(o["plate_found"] for o in observations):
        return "UNCERTAIN", "plate not found in every capture"
    counts = {len(o["objects"]) for o in observations}
    if counts == {0}:
        return "EMPTY", "no material on the plate"
    if len(counts) > 1:
        return "UNCERTAIN", f"object count changed between captures {sorted(counts)}"
    verdicts = {tuple(obj["verdict"] for obj in o["objects"]) for o in observations}
    if len(verdicts) > 1:
        return "UNCERTAIN", "captures disagree"
    v = verdicts.pop()
    if "UNCERTAIN" in v:
        return "UNCERTAIN", "an object is neither clearly red nor clearly white"
    return ("FAILED" if "FAILED" in v else "GOOD"), f"{len(v)} object(s): {', '.join(v)}"


# ----------------------------------------------------------------- Viam I/O
async def connect():
    """Same environment variables as the Sandbox template."""
    from viam.robot.client import RobotClient

    addr = next((os.environ[k] for k in ("VIAM_MACHINE_ADDRESS", "VIAM_ADDRESS", "VIAM_ROBOT_ADDRESS",
                                         "MACHINE_ADDRESS") if os.environ.get(k)), None)
    if addr is None:
        raise RuntimeError("machine address env var not found; paste the Sandbox template's connect() here. "
                           "VIAM_* vars present: " + ", ".join(k for k in os.environ if k.startswith("VIAM")))
    opts = RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"],
                                            api_key_id=os.environ["VIAM_API_KEY_ID"])
    return await RobotClient.at_address(addr, opts)


async def to_world(robot, xyz):
    from viam.proto.common import Pose, PoseInFrame

    q = PoseInFrame(reference_frame=CONFIG["camera"], pose=Pose(x=xyz[0], y=xyz[1], z=xyz[2], o_z=1))
    r = await robot.transform_pose(q, CONFIG["world_frame"])
    return round(r.pose.x, 1), round(r.pose.y, 1), round(r.pose.z, 1)


async def compare_vision_services(robot):
    """Print what the machine's existing vision services report (read-only)."""
    from viam.services.vision import VisionClient

    present = {r.name for r in robot.resource_names}
    for name in CONFIG["compare_vision_services"]:
        if name not in present:
            continue
        try:
            svc = VisionClient.from_robot(robot, name)
            props = await svc.get_properties()
            if props.detections_supported:
                dets = await svc.get_detections_from_camera(CONFIG["camera"])
                summary = ", ".join(f"{d.class_name}:{d.confidence:.2f}@({d.x_min},{d.y_min})-({d.x_max},{d.y_max})"
                                    for d in dets) or "no detections"
            elif props.classifications_supported:
                cls = await svc.get_classifications_from_camera(CONFIG["camera"], 3)
                summary = ", ".join(f"{c.class_name}:{c.confidence:.2f}" for c in cls) or "no classifications"
            else:
                summary = "no detections/classifications supported"
            print(f"    [{name}] {summary}")
        except Exception as e:
            print(f"    [{name}] error: {e}")


async def inspect_plate(cell):
    """Run N captures and return (verdict, why, last_observation)."""
    observations = []
    for i in range(CONFIG["n_captures"]):
        t0 = time.time()
        data, mime = await cell.cam.get_point_cloud()
        obs = analyze(data)
        observations.append(obs)
        print(f"\n--- capture {i + 1}/{CONFIG['n_captures']} ({len(data) / 1e6:.1f} MB, {mime}, "
              f"{time.time() - t0:.1f}s)")
        for note in obs["notes"]:
            print("   ", note)
        for k, o in enumerate(obs["objects"]):
            print(f"    object {k}: {o['verdict']:<9} red={o['red_frac']:.2f} white={o['white_frac']:.2f} "
                  f"h={o['height_mm']}mm extent={o['extent_mm']}mm cam={o['top_center_cam_mm']}")
        await asyncio.sleep(0.3)
    verdict, why = decide(observations, CONFIG["n_captures"])
    print(f"\n===== VERDICT: {verdict}  ({why})")
    return verdict, why, observations[-1]


# ================================================================== MOTION
MOTION = {
    "STEP": 0,                    # 0 = no motion. See the top of this file.
    "I_AM_AT_THE_ESTOP": False,   # must be True for any STEP >= 1
    "gripper": "gripper",
    "motion": "builtin",
    # Gripper poses in the WORLD frame (mm, orientation vector + theta in degrees).
    # To record one: put the arm there (control card or manual mode), run STEP 0,
    # and copy the printed "gripper pose (world)" dict here.
    # Scan pose recorded 2026-11-06 from the ARM control card (arm end flange):
    #   x=98.98 y=-601.08 z=471.52 o=(0.04, 0.01, -1) theta=100.79
    # converted to the GRIPPER frame (105 mm further along the tool axis, per the
    # machine's frame config). Verify: at this pose STEP 0 should print a
    # "gripper pose (world)" matching these values.
    "scan_pose": {"x": 103.18, "y": -600.03, "z": 366.61,
                  "o_x": 0.04, "o_y": 0.01, "o_z": -1.0, "theta": 100.79},
    "good_bin_pose": None,        # above the WHITE / good-print bin
    "failed_bin_pose": None,      # above the RED / failed-print bin
    "home_pose": None,            # safe resting position, clear of printer and bins
    # Extra obstacles for the planner, world frame. The PRINTER is not one of the
    # machine's preconfigured obstacles (table, wall-front, wall-side, ceiling):
    #   [{"label": "printer_frame", "center": {"x": .., "y": .., "z": ..},
    #     "dims_mm": {"x": .., "y": .., "z": ..}}]
    # Keep the boxes AROUND the bed, not over it. Set [] to say "none needed".
    # STEP 3 and 4 refuse to run while this is None.
    # Demo setup (2026-11-06): NO printer - the "build plate" is a PEI sheet lying
    # on a box, with nothing above it, and the arm approaches from above. The
    # box sits BELOW the plate surface, and the fingertips stop
    # plate_clearance_mm above the plate, so no extra obstacle is needed.
    "extra_obstacles": [],
    "nudge_up_mm": 30.0,
    "approach_above_mm": 100.0,   # hover height above the print top
    "grasp_depth_mm": 10.0,       # how far below the print top the fingertips go
    "plate_clearance_mm": 4.0,    # fingertips never closer than this to the plate
    "lift_mm": 120.0,
    "linear_tolerance_mm": 5.0,   # straight-line descend / lift
    # Software workspace fence (world frame). Every target is checked before moving.
    "workspace": {"min_z_mm": 5.0, "max_z_mm": 700.0, "min_reach_mm": 150.0, "max_reach_mm": 650.0},
}


class SafetyError(RuntimeError):
    pass


class Cell:
    """Handles to the machine: built from a RobotClient in the Sandbox, or from
    fakes in the offline tests."""

    def __init__(self, robot, motion, gripper, arm, cam):
        self.robot, self.motion, self.gripper, self.arm, self.cam = robot, motion, gripper, arm, cam
        self.moves = []           # executed moves: (label, pose)

    @classmethod
    def from_robot(cls, robot):
        from viam.components.arm import Arm
        from viam.components.camera import Camera
        from viam.components.gripper import Gripper
        from viam.services.motion import Motion

        return cls(robot, Motion.from_robot(robot, MOTION["motion"]), Gripper.from_robot(robot, MOTION["gripper"]),
                   Arm.from_robot(robot, CONFIG["arm"]), Camera.from_robot(robot, CONFIG["camera"]))


POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


def pose_dict(p):
    return {k: round(float(getattr(p, k)), 2) for k in POSE_KEYS}


def check_pose(label, pose):
    """Reasons this target is outside the software fence ([] = OK)."""
    ws = MOTION["workspace"]
    if pose is None or any(k not in pose for k in POSE_KEYS):
        return [f"{label}: pose not set / incomplete"]
    problems = []
    reach = math.hypot(pose["x"], pose["y"])
    if not ws["min_z_mm"] <= pose["z"] <= ws["max_z_mm"]:
        problems.append(f"{label}: z={pose['z']:.0f} outside [{ws['min_z_mm']}, {ws['max_z_mm']}] mm")
    if not ws["min_reach_mm"] <= reach <= ws["max_reach_mm"]:
        problems.append(f"{label}: reach {reach:.0f} mm outside [{ws['min_reach_mm']}, {ws['max_reach_mm']}] mm")
    if math.sqrt(pose["o_x"] ** 2 + pose["o_y"] ** 2 + pose["o_z"] ** 2) < 0.5:
        problems.append(f"{label}: orientation vector is ~zero")
    return problems


def world_state():
    from viam.proto.common import Geometry, GeometriesInFrame, Pose, RectangularPrism, Vector3, WorldState

    boxes = MOTION["extra_obstacles"] or []
    if not boxes:
        return None
    geoms = [Geometry(center=Pose(x=b["center"]["x"], y=b["center"]["y"], z=b["center"]["z"], o_z=1),
                      box=RectangularPrism(dims_mm=Vector3(x=b["dims_mm"]["x"], y=b["dims_mm"]["y"],
                                                           z=b["dims_mm"]["z"])),
                      label=b.get("label", "obstacle")) for b in boxes]
    return WorldState(obstacles=[GeometriesInFrame(reference_frame=CONFIG["world_frame"], geometries=geoms)])


async def move_gripper(cell, label, pose, linear=False):
    """Collision-aware gripper move via the motion service. Checks the fence
    first; on ANY failure stops all motion and re-raises."""
    from viam.proto.common import Pose, PoseInFrame
    from viam.proto.service.motion import Constraints, LinearConstraint

    problems = check_pose(label, pose)
    if problems:
        raise SafetyError("; ".join(problems))
    constraints = None
    if linear:
        constraints = Constraints(linear_constraint=[LinearConstraint(
            line_tolerance_mm=MOTION["linear_tolerance_mm"])])
    print(f"  -> move {label}{' (straight line)' if linear else ''}: {pose}")
    try:
        ok = await cell.motion.move(component_name=MOTION["gripper"],
                                    destination=PoseInFrame(reference_frame=CONFIG["world_frame"], pose=Pose(**pose)),
                                    world_state=world_state(), constraints=constraints)
        if not ok:
            raise RuntimeError(f"motion.move returned False for {label}")
    except Exception:
        print("  !! move failed - stopping all motion")
        await cell.robot.stop_all()
        raise
    cell.moves.append((label, pose))


async def current_gripper_pose(cell):
    pif = await cell.motion.get_pose(component_name=MOTION["gripper"], destination_frame=CONFIG["world_frame"])
    return pose_dict(pif.pose)


def plan_pick(top_world, height_mm, scan_pose):
    """Hover / grasp / lift poses above the print.

    Keeps the scan orientation: the wrist camera and gripper share the arm's z
    axis, so 'camera looking down' means 'gripper pointing down'. The jaw
    rotation is NOT aligned to the part yet - place prints with their short
    side across the jaws."""
    x, y, top_z = top_world
    usable = height_mm - MOTION["plate_clearance_mm"]
    if usable < 2.0:
        raise SafetyError(f"print too flat to pinch ({height_mm:.1f} mm)")
    depth = min(MOTION["grasp_depth_mm"], usable)
    orient = {k: scan_pose[k] for k in ("o_x", "o_y", "o_z", "theta")}
    return {
        "hover": {"x": x, "y": y, "z": round(top_z + MOTION["approach_above_mm"], 1), **orient},
        "grasp": {"x": x, "y": y, "z": round(top_z - depth, 1), **orient},
        "lift": {"x": x, "y": y, "z": round(top_z + MOTION["lift_mm"], 1), **orient},
    }


def require(cond, msg):
    if not cond:
        raise SafetyError(msg)


async def run_step(cell, step):
    m = MOTION
    if step >= 1:
        require(m["I_AM_AT_THE_ESTOP"], "set MOTION['I_AM_AT_THE_ESTOP'] = True (someone at the E-stop) to move")
    if step not in (0, 1, 2, 3, 4, 5):
        raise SafetyError(f"unknown STEP {step}")
    if 2 <= step <= 4:
        probs = check_pose("scan_pose", m["scan_pose"])
        require(not probs, "; ".join(probs))
    if 3 <= step <= 4:
        require(m["extra_obstacles"] is not None,
                "measure the printer and set MOTION['extra_obstacles'] (or [] if none) before STEP 3/4")
    if step == 4:
        for k in ("good_bin_pose", "failed_bin_pose"):
            probs = check_pose(k, m[k])
            require(not probs, "; ".join(probs))
    if step == 5 or (step == 4 and m["home_pose"] is not None):
        probs = check_pose("home_pose", m["home_pose"])
        require(not probs, "; ".join(probs))

    here = await current_gripper_pose(cell)
    print(f"gripper pose (world): {here}")

    if step == 0:
        try:
            p = await cell.arm.get_end_position()
            print(f"arm end position: x={p.x:.1f} y={p.y:.1f} z={p.z:.1f} "
                  f"o=({p.o_x:.2f},{p.o_y:.2f},{p.o_z:.2f}) theta={p.theta:.1f}")
        except Exception as e:
            print("could not read arm pose:", e)
        for k in ("scan_pose", "good_bin_pose", "failed_bin_pose", "home_pose"):
            probs = check_pose(k, m[k])
            print(f"  {k}: {'OK' if not probs else '; '.join(probs)}")
        verdict, _, obs = await inspect_plate(cell)
        if verdict in ("GOOD", "FAILED"):
            for k, o in enumerate(obs["objects"]):
                w = await to_world(cell.robot, o["top_center_cam_mm"])
                print(f"    object {k} top centre in world frame: {w} mm")
        print("\nSTEP 0 done - no motion was sent.")
        return "read-only"

    if step == 1:
        target = dict(here, z=round(here["z"] + m["nudge_up_mm"], 1))
        await move_gripper(cell, "nudge_up", target, linear=True)
        after = await current_gripper_pose(cell)
        print(f"gripper pose after: {after}  (dz = {after['z'] - here['z']:+.1f} mm, "
              f"expected +{m['nudge_up_mm']})")
        return "nudged"

    if step == 5:
        await move_gripper(cell, "home", m["home_pose"])
        return "at home"

    await move_gripper(cell, "scan", m["scan_pose"])
    if step == 2:
        return "at scan pose"

    verdict, why, obs = await inspect_plate(cell)
    if verdict not in ("GOOD", "FAILED"):
        print(f"no automatic action for {verdict}: {why}")
        return verdict
    if len(obs["objects"]) > 1:
        print(f"note: {len(obs['objects'])} objects - picking the largest only")
    target = obs["objects"][0]
    top_world = await to_world(cell.robot, target["top_center_cam_mm"])
    poses = plan_pick(top_world, target["height_mm"], m["scan_pose"])
    bin_name = "good_bin_pose" if verdict == "GOOD" else "failed_bin_pose"
    to_check = dict(poses)
    if step >= 4:
        to_check["bin"] = m[bin_name]
    problems = [p for k, v in to_check.items() for p in check_pose(k, v)]
    require(not problems, "planned poses outside fence: " + "; ".join(problems))  # before any further motion
    print(f"print top (world): {top_world}, height {target['height_mm']} mm -> {verdict}, bin: {bin_name}")

    if step == 3:
        await move_gripper(cell, "hover", poses["hover"])
        print("STEP 3 done - hovering above the print. Check it is centred before trying STEP 4.")
        return "hovering"

    await cell.gripper.open()
    await move_gripper(cell, "hover", poses["hover"])
    await move_gripper(cell, "grasp", poses["grasp"], linear=True)
    grabbed = await cell.gripper.grab()
    status = await cell.gripper.is_holding_something()
    holding = getattr(status, "is_holding_something", grabbed)
    if not (grabbed and holding):
        print("!! grasp failed - opening and backing off")
        await cell.gripper.open()
        await move_gripper(cell, "retreat", poses["hover"], linear=True)
        return "grasp failed"
    await move_gripper(cell, "lift", poses["lift"], linear=True)
    await move_gripper(cell, bin_name, m[bin_name])
    await cell.gripper.open()
    await move_gripper(cell, "scan", m["scan_pose"])
    after, why, _ = await inspect_plate(cell)
    print(f"plate after removal: {after} ({why})")
    if m["home_pose"] is not None:
        await move_gripper(cell, "home", m["home_pose"])
    return "sorted" if after == "EMPTY" else "plate not clear"


async def main():
    robot = await connect()
    try:
        print("Resources:", ", ".join(sorted(r.name for r in robot.resource_names)))
        cell = Cell.from_robot(robot)
        try:
            result = await run_step(cell, MOTION["STEP"])
            print(f"\nRESULT (STEP {MOTION['STEP']}): {result}")
        except SafetyError as e:
            print(f"\nREFUSED (STEP {MOTION['STEP']}): {e}")
        if MOTION["STEP"] == 0:
            print("\n--- cross-check: existing vision services on this machine")
            await compare_vision_services(robot)
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
