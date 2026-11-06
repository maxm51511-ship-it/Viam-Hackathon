# print_defects dataset (YOLO segmentation)

```
datasets/print_defects/
  data.yaml
  images/train/*.png     images/val/*.png
  labels/train/*.txt     labels/val/*.txt      # same stem as the image
```

Images and labels are git-ignored (size); only the structure and docs are versioned.

## Label format

One line per instance, polygon normalised to 0–1 (Ultralytics segmentation format):

```
<class_id> x1 y1 x2 y2 x3 y3 ...
```

| id | class | Annotate | Notes |
|---|---|---|---|
| 0 | `build_plate` | exposed plate surface | One polygon. It helps the model learn the plate's appearance under different lighting. |
| 1 | `printed_object` | each intact part | Includes the brim if the brim is attached. |
| 2 | `spaghetti` | tangled, unstructured extrusion | Draw one polygon around each connected tangle. |
| 3 | `loose_filament` | single strands, blobs, purge lines | Covers loose material that isn't a tangle. |
| 4 | `other_defect` | warped or detached part, layer shift, missing section | Use sparingly. Each defect type needs about 50 examples before it's worth adding. |

## Collecting images

1. Run `python tools/capture_frames.py --tag <scenario> --n 20 --out data/captures/<session>`. Each capture saves colour + depth + intrinsics.
   - You can also enable `GetImages` capture on the camera (see `config/machine.example.json`). The Viam data manager then syncs images to the Viam cloud. You can tag and export them from there.
2. Vary the conditions on purpose:
   - filament colours, including one that matches the plate
   - plate types
   - room lights on and off
   - parts at different positions on the plate
   - partial occlusion by the print head
3. Aim for at least 150 labelled images. Make sure spaghetti and loose filament each appear in at least 50 images, then grow the set from real failures.
4. Annotate with any tool that exports "YOLO segmentation" polygons, for example CVAT, Label Studio or Roboflow.
5. Split train/val by **print session**, not by frame. Consecutive frames of the same scene leak between the splits and inflate the scores.

## Training

```
pip install -r requirements-ml.txt
python tools/train_yolo.py --model yolo11n-seg.pt --epochs 100
```

Viam's hosted `yolo-onnx-training` script trains **detection** (bounding boxes) from Viam datasets, not segmentation. We train segmentation locally, then run it in-process in `printbot.perception.yolo`.
