# red_white dataset (YOLO detection)

```
datasets/red_white/
  data.yaml
  bootstrap_report.json          # per-box red/white fractions + label for every image
  images/train/*.jpg   images/val/*.jpg
  labels/train/*.txt   labels/val/*.txt      # same stem as the image, YOLO box format
```

Images and labels are git-ignored (size); only the structure, docs and report are versioned.
Regenerate with:

```
viam dataset export --destination=raw_exports/<name> --dataset-id=<id>   # pulls raw frames
python tools/build_red_white_dataset.py                                 # (re)writes this folder
```

## What this is

Two classes for the hackathon demo rule (red filament = failed print, white
filament = good print): `0: failed`, `1: good`. This is a **different**
dataset/model from `datasets/print_defects` (5-class segmentation for real
print defects - spaghetti, loose filament, etc). The two are not
interchangeable; this one only tells you "red part" vs "white part", not
whether a print actually succeeded.

## Sources

| Source | Dir | Images | How boxes are found | How labelled |
|---|---|---|---|---|
| Original mixed shapes | `data/` | 5 | hand-verified (see below) | `colour_rule` on box pixels |
| Good-only capture | `raw_exports/good_only/` | 4 | automatic (HSV + plate-silhouette mask) | forced `good` (dataset identity) |
| Mixed capture | `raw_exports/mixed/` | 4 | automatic | `colour_rule` on box pixels |
| Empty plate | `raw_exports/empty_plate/` | 7 | none - pure negatives | n/a (zero boxes) |

The "empty plate" negatives exist specifically because the plate's own
embossed text and engraved logo graphic look enough like white material to
fool naive detection (see below) - explicit zero-object examples teach the
model that pattern is the plate, not a part.

## Provenance (important - read before trusting this data)

Automatic HSV-based object segmentation was tried first **on the original
5-image set** and failed: the plate's embossed "Bambu Textured PEI Plate"
text and its engraved logo graphic sit in the same brightness/saturation
range as the real white plastic parts, with no HSV threshold found that
cleanly separated real objects from plate markings there. The repo's own
`printbot.perception.plate` avoids this by segmenting with **depth** (flat
engraving vs. raised plastic) - that export has no matching depth frame, so
that path wasn't available. **Those 5 images' object boxes are therefore
hand-verified** (read directly off each image), not auto-segmented.

The later three exports (good_only, mixed, empty_plate) have objects
positioned away from the engraving, so automatic detection works there -
`tools/build_red_white_dataset.py`'s `detect_boxes()`:
1. restricts search to the plate's own (possibly tilted) silhouette, not its
   rectangular bbox, which would catch real desk background in the corners;
2. applies a morphological opening to kill thin engraved strokes;
3. rejects boxes by bounding-box area, min side length, and **fill ratio**
   (actual "on" pixel count vs. bbox area) - a real solid object fills much
   more of its own bbox than a thin snaking line does, even though both can
   fragment into multi-component or low-density shapes (a shaded sphere's
   highlight-only mask has fill ~0.2-0.3; the plate-engraving bridging blob
   sits at ~0.03-0.06 - tuned empirically, see git history for the sweep);
4. merges any two boxes that substantially overlap - a part with a strong
   colour split (e.g. red top, white base) can fragment into two components
   after the opening; they're one physical object.

For all sources, the failed/good **label** per box comes from the repo's
official `printbot.inspection.colour_rule` thresholds (`printbot/config.py`
`classifier.colour`) run over that box's actual red/white pixels - the same
logic the live system uses, not a one-off guess. The one exception is
good_only, where the label is forced to `good` from the dataset's own
identity (more reliable than re-deriving it from a fragile pixel heuristic);
the script still computes and prints the red/white fractions there as a
sanity check, flagging a warning if red_frac is implausibly high.

One box in the original set (`...MfZd.../635,615`) is a near-even split
(red=0.49/white=0.51) - a genuinely ambiguous object by the colour rule's own
dominance test (neither side reaches the 2x-dominance bar
`printbot.inspection.colour_rule.colour_vote` requires; this would be
`UNCERTAIN` there). The bootstrap script breaks the tie toward whichever
fraction is (barely) larger. Worth a second look before leaning on this
example.

## Known limitations

- **~20 images total is still very little data.** Expect limited
  generalization to new lighting, part positions, or plate framing not
  represented here (e.g. the original 5-image set's tilted/rotated plate).
- Train/val split is one image held out per source, not a real split - don't
  trust val metrics as a generalization estimate.
- `good_only` and `mixed` capture the same 3 spheres in near-identical
  positions across all their frames (seconds apart) - low positional
  diversity for the `good` class specifically.
