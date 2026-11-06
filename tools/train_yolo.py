"""Fine-tune a YOLO segmentation model on datasets/print_defects.

    pip install -r requirements-ml.txt
    python tools/train_yolo.py --model yolo11n-seg.pt --epochs 100 --imgsz 640

Copies best weights to models/print_defects-seg.pt. Check the validation mAP
per class (especially spaghetti / loose_filament) before trusting the model;
the classifier only uses detections above classifier.yolo_min_confidence and
still requires agreement with depth heuristics.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(a) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed: pip install -r requirements-ml.txt", file=sys.stderr)
        return 1
    data = ROOT / "datasets" / "print_defects" / "data.yaml"
    model = YOLO(a.model)
    res = model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                      project=str(ROOT / "runs"), name="print_defects", exist_ok=True)
    best = Path(res.save_dir) / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "print_defects-seg.pt")
    print(f"copied {best} -> models/print_defects-seg.pt")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    raise SystemExit(main(p.parse_args()))
