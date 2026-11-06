"""Parse a Bambu Studio .gcode file and report where each printed object sits on the bed.

Reads the per-object blocks that Bambu Studio emits:

    ; start printing object, unique label id: 540
    ...
    ; stop printing object, unique label id: 540

and the feature comments inside them ("; FEATURE: Outer wall" etc). Only
extruding moves count toward an object's geometry, and Support features are
excluded by default so the numbers describe the part itself, not the scaffolding
it was printed on.

Handles relative extrusion (M83), absolute extrusion (M82), G90/G91 positioning,
G92 resets, G0/G1 linear moves and G2/G3 arcs (I/J center-offset or R form,
including P>1 multi-turn arcs). Arcs are flattened to short chords.

X/Y in Bambu P2S gcode are already bed millimetres with the origin at the
front-left corner of the 256 x 256 mm bed, so no transform is applied. Z is
height above the bed surface.

Usage:
    python gcode_parser.py "path/to/print.gcode"
    python gcode_parser.py print.gcode --names project.3mf --json parts.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass, field

# Feature names (lowercased prefixes) treated as scaffolding, not part geometry.
# Covers "Support", "Support interface" and "Support transition".
SUPPORT_FEATURES = ("support",)

# Arcs are flattened into chords no longer than this (mm).
DEFAULT_ARC_STEP = 0.4

INF = float("inf")


def is_support(feature: str | None) -> bool:
    """True if this "; FEATURE:" name is support scaffolding, not part geometry."""
    if not feature:
        return False
    low = feature.strip().lower()
    return any(low.startswith(p) for p in SUPPORT_FEATURES)


class Extent:
    """Bounding box plus length-weighted centroid of a set of extruded segments."""

    def __init__(self) -> None:
        self.lo = [INF, INF, INF]
        self.hi = [-INF, -INF, -INF]
        self._wsum = 0.0
        self._csum = [0.0, 0.0, 0.0]
        self.length = 0.0
        self.segments = 0

    @property
    def empty(self) -> bool:
        return self.segments == 0

    def add(self, p0: list[float], p1: list[float]) -> None:
        seg = math.dist(p0, p1)
        for k in range(3):
            a, b = p0[k], p1[k]
            if b < a:
                a, b = b, a
            if a < self.lo[k]:
                self.lo[k] = a
            if b > self.hi[k]:
                self.hi[k] = b
        if seg > 0.0:
            self._wsum += seg
            for k in range(3):
                self._csum[k] += seg * 0.5 * (p0[k] + p1[k])
            self.length += seg
        self.segments += 1

    @property
    def size(self) -> list[float]:
        if self.empty:
            return [0.0, 0.0, 0.0]
        return [self.hi[k] - self.lo[k] for k in range(3)]

    @property
    def bbox_center(self) -> list[float]:
        if self.empty:
            return [float("nan")] * 3
        return [0.5 * (self.lo[k] + self.hi[k]) for k in range(3)]

    @property
    def centroid(self) -> list[float]:
        """Centroid of deposited material, weighted by extruded path length."""
        if self._wsum <= 0.0:
            return self.bbox_center
        return [self._csum[k] / self._wsum for k in range(3)]


@dataclass
class PrintedObject:
    label_id: int
    name: str | None = None
    model: Extent = field(default_factory=Extent)
    support: Extent = field(default_factory=Extent)
    features: dict[str, float] = field(default_factory=dict)

    def record_feature(self, feature: str | None, length: float) -> None:
        key = feature or "(unlabelled)"
        self.features[key] = self.features.get(key, 0.0) + length


def _arc_sweep(a0: float, a1: float, clockwise: bool, turns: int) -> float:
    """Signed angular sweep from a0 to a1 in the given direction."""
    sweep = a1 - a0
    if clockwise:
        while sweep > 1e-12:
            sweep -= 2 * math.pi
    else:
        while sweep < -1e-12:
            sweep += 2 * math.pi
    full = -2 * math.pi if clockwise else 2 * math.pi
    if abs(sweep) < 1e-12:
        sweep = full  # endpoint == start point: a complete circle
    sweep += (max(turns, 1) - 1) * full
    return sweep


def _arc_points(
    start: list[float],
    end: list[float],
    i: float | None,
    j: float | None,
    radius: float | None,
    clockwise: bool,
    turns: int,
    arc_step: float,
):
    """Yield points along a G2/G3 arc in the XY plane (Z interpolated linearly)."""
    x0, y0, z0 = start
    x1, y1, z1 = end
    if i is not None or j is not None:
        cx, cy = x0 + (i or 0.0), y0 + (j or 0.0)
    elif radius is not None:
        # R form: the center lies on the perpendicular bisector of the chord.
        dx, dy = x1 - x0, y1 - y0
        chord = math.hypot(dx, dy)
        if chord == 0.0 or abs(radius) < chord / 2:
            yield end
            return
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        h = math.sqrt(max(radius * radius - (chord / 2) ** 2, 0.0))
        # Positive R takes the short way round, negative R the long way.
        sign = 1.0 if (radius > 0) == clockwise else -1.0
        cx = mx + sign * h * (-dy / chord)
        cy = my + sign * h * (dx / chord)
    else:
        yield end
        return

    r = math.hypot(x0 - cx, y0 - cy)
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    sweep = _arc_sweep(a0, a1, clockwise, turns)

    steps = max(1, int(math.ceil(abs(sweep) * r / arc_step)))
    for s in range(1, steps + 1):
        t = s / steps
        a = a0 + sweep * t
        yield [cx + r * math.cos(a), cy + r * math.sin(a), z0 + (z1 - z0) * t]


_START_RE = re.compile(r"start printing object.*?id:\s*(\d+)", re.I)
_STOP_RE = re.compile(r"stop printing object.*?id:\s*(\d+)", re.I)


def parse_gcode(
    path: str,
    exclude_support: bool = True,
    arc_step: float = DEFAULT_ARC_STEP,
) -> dict[int, PrintedObject]:
    """Return {label_id: PrintedObject} for every object block in the file."""
    objects: dict[int, PrintedObject] = {}
    current: PrintedObject | None = None
    feature: str | None = None
    wiping = False

    pos = [0.0, 0.0, 0.0]
    e_pos = 0.0
    rel_e = True      # Bambu emits M83 up front; M82 flips this
    rel_xyz = False   # G90 default

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(";"):
                c = line[1:].strip()
                if c.startswith("FEATURE:"):
                    feature = c[len("FEATURE:"):].strip()
                elif c == "WIPE_START":
                    wiping = True
                elif c == "WIPE_END":
                    wiping = False
                elif "printing object" in c:
                    # The feature deliberately carries across the object boundary:
                    # Bambu emits "; FEATURE:" only when the feature changes, so the
                    # first paths of a block can belong to the still-current feature.
                    m = _START_RE.search(c)
                    if m:
                        label = int(m.group(1))
                        current = objects.setdefault(label, PrintedObject(label))
                    elif _STOP_RE.search(c):
                        current = None
                continue

            if ";" in line:
                line = line.split(";", 1)[0].strip()
                if not line:
                    continue

            parts = line.split()
            cmd = parts[0].upper()

            if cmd in ("G0", "G1", "G2", "G3"):
                words: dict[str, float] = {}
                for tok in parts[1:]:
                    key = tok[0].upper()
                    if key in "XYZEIJRP":
                        try:
                            words[key] = float(tok[1:])
                        except ValueError:
                            pass

                target = list(pos)
                for key, axis in (("X", 0), ("Y", 1), ("Z", 2)):
                    if key in words:
                        target[axis] = pos[axis] + words[key] if rel_xyz else words[key]

                if "E" in words:
                    if rel_e:
                        de = words["E"]
                        e_pos += words["E"]
                    else:
                        de = words["E"] - e_pos
                        e_pos = words["E"]
                else:
                    de = 0.0

                # Only deposited material defines geometry: skip travels, retracts
                # and wipe moves, and anything outside an object block.
                if de > 0.0 and not wiping and current is not None:
                    # Support always lands in its own extent so it can be reported;
                    # it counts as part geometry only when the caller asks for it.
                    if is_support(feature):
                        targets = [current.support]
                        if not exclude_support:
                            targets.append(current.model)
                    else:
                        targets = [current.model]
                    before = targets[0].length

                    prev = list(pos)
                    if cmd in ("G2", "G3"):
                        pts = _arc_points(
                            pos,
                            target,
                            words.get("I"),
                            words.get("J"),
                            words.get("R"),
                            clockwise=(cmd == "G2"),
                            turns=int(words.get("P", 1)),
                            arc_step=arc_step,
                        )
                    else:
                        pts = iter([target])
                    for pt in pts:
                        for track in targets:
                            track.add(prev, pt)
                        prev = pt
                    current.record_feature(feature, targets[0].length - before)

                pos = target

            elif cmd == "G92":
                for tok in parts[1:]:
                    key = tok[0].upper()
                    try:
                        val = float(tok[1:])
                    except ValueError:
                        continue
                    if key == "E":
                        e_pos = val
                    elif key in "XYZ":
                        pos["XYZ".index(key)] = val

            elif cmd == "M82":
                rel_e = False
            elif cmd == "M83":
                rel_e = True
            elif cmd == "G90":
                rel_xyz = False
            elif cmd == "G91":
                rel_xyz = True

    return objects


# ---------------------------------------------------------------------------
# Optional: pull real part names out of the Bambu .3mf project
# ---------------------------------------------------------------------------

def read_3mf_parts(path: str) -> list[tuple[str, float, float]]:
    """Return [(name, x, y)] for each placed instance in a Bambu .3mf project."""
    with zipfile.ZipFile(path) as z:
        names: dict[str, str] = {}
        settings = z.read("Metadata/model_settings.config").decode("utf-8", "replace")
        for oid, body in re.findall(
            r'<object id="(\d+)"(.*?)</object>', settings, re.S
        ):
            m = re.search(r'key="name" value="([^"]*)"', body)
            if m:
                names[oid] = m.group(1)

        model = z.read("3D/3dmodel.model").decode("utf-8", "replace")
        out: list[tuple[str, float, float]] = []
        for m in re.finditer(
            r'<item objectid="(\d+)"[^>]*transform="([^"]+)"', model
        ):
            oid, xf = m.group(1), m.group(2).split()
            if len(xf) < 12:
                continue
            # 3MF row-major 4x3 matrix: the translation is the last three values.
            out.append((names.get(oid, f"object {oid}"), float(xf[9]), float(xf[10])))
    return out


def assign_names(
    objects: dict[int, PrintedObject],
    parts: list[tuple[str, float, float]],
    tol: float = 30.0,
) -> None:
    """Name each parsed object by the nearest .3mf instance position in XY."""
    pool = list(parts)
    for obj in sorted(objects.values(), key=lambda o: o.label_id):
        if obj.model.empty or not pool:
            continue
        cx, cy, _ = obj.model.bbox_center
        best = min(range(len(pool)), key=lambda k: math.dist((cx, cy), pool[k][1:3]))
        name, px, py = pool.pop(best)
        d = math.dist((cx, cy), (px, py))
        obj.name = name if d <= tol else f"{name}? ({d:.0f} mm from 3mf position)"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(objects: dict[int, PrintedObject], exclude_support: bool) -> None:
    print(
        f"{len(objects)} printed object(s); Support features "
        f"{'excluded' if exclude_support else 'included'}."
    )
    print("Coordinates are bed mm (origin front-left); Z is height above the bed.\n")

    for obj in sorted(objects.values(), key=lambda o: o.label_id):
        head = f"label id {obj.label_id}"
        print(f"{obj.name}  [{head}]" if obj.name else head)

        m = obj.model
        if m.empty:
            print("  no part extrusion found\n")
            continue

        lo, hi, size = m.lo, m.hi, m.size
        print(f"  bbox      X {lo[0]:8.2f} .. {hi[0]:8.2f}  ({size[0]:6.2f} mm)")
        print(f"            Y {lo[1]:8.2f} .. {hi[1]:8.2f}  ({size[1]:6.2f} mm)")
        print(f"            Z {lo[2]:8.2f} .. {hi[2]:8.2f}  ({size[2]:6.2f} mm)")
        print(f"  height    {size[2]:.2f} mm tall, top of part at Z {hi[2]:.2f}")
        c, bc = m.centroid, m.bbox_center
        print(f"  centroid  ({c[0]:7.2f}, {c[1]:7.2f}, {c[2]:6.2f})  extrusion-weighted")
        print(f"  bbox ctr  ({bc[0]:7.2f}, {bc[1]:7.2f}, {bc[2]:6.2f})")

        s = obj.support
        if not s.empty:
            print(
                f"  support   Z {s.lo[2]:.2f} .. {s.hi[2]:.2f}, footprint "
                f"X {s.lo[0]:.1f}..{s.hi[0]:.1f} Y {s.lo[1]:.1f}..{s.hi[1]:.1f}"
            )
        feats = ", ".join(
            f"{k} {v:.0f}mm"
            for k, v in sorted(obj.features.items(), key=lambda kv: -kv[1])
        )
        print(f"  features  {feats}")
        print(f"  extruded  {m.length:.0f} mm of path in {m.segments} segments\n")


def to_json(objects: dict[int, PrintedObject]) -> list[dict]:
    out = []
    for obj in sorted(objects.values(), key=lambda o: o.label_id):
        m = obj.model
        if m.empty:
            continue
        out.append(
            {
                "label_id": obj.label_id,
                "name": obj.name,
                "bbox_min_mm": [round(v, 3) for v in m.lo],
                "bbox_max_mm": [round(v, 3) for v in m.hi],
                "size_mm": [round(v, 3) for v in m.size],
                "height_mm": round(m.size[2], 3),
                "top_z_mm": round(m.hi[2], 3),
                "centroid_mm": [round(v, 3) for v in m.centroid],
                "bbox_center_mm": [round(v, 3) for v in m.bbox_center],
                "support_z_mm": None
                if obj.support.empty
                else [round(obj.support.lo[2], 3), round(obj.support.hi[2], 3)],
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gcode", help="Bambu Studio .gcode file")
    ap.add_argument(
        "--names",
        metavar="PROJECT.3MF",
        help="Bambu .3mf project to take part names from",
    )
    ap.add_argument("--json", metavar="OUT.JSON", help="also write results as JSON")
    ap.add_argument(
        "--include-support",
        action="store_true",
        help="count Support features as part geometry",
    )
    ap.add_argument(
        "--arc-step",
        type=float,
        default=DEFAULT_ARC_STEP,
        help=f"arc flattening resolution in mm (default {DEFAULT_ARC_STEP})",
    )
    args = ap.parse_args(argv)

    exclude_support = not args.include_support
    objects = parse_gcode(
        args.gcode, exclude_support=exclude_support, arc_step=args.arc_step
    )
    if not objects:
        print(
            "No '; start printing object' blocks found - is this a Bambu gcode "
            "sliced with label_objects enabled?",
            file=sys.stderr,
        )
        return 1

    if args.names:
        assign_names(objects, read_3mf_parts(args.names))

    report(objects, exclude_support)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(to_json(objects), fh, indent=2)
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
