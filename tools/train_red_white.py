"""Fine-tune a YOLO detection model on datasets/red_white (classes: failed=red,
good=white - the hackathon demo colour rule, not the real print_defects model).

    pip install -r requirements-ml.txt
    python tools/train_red_white.py --model yolov8n.pt --epochs 80 --imgsz 640

Copies best weights to models/red_white.pt. See datasets/red_white/README.md
before trusting this model: ~20 images total, low positional diversity for
the good class, and a token (not real) train/val split - expect this to
overfit and generalize narrowly. Treat it as a starting point.
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
    data = ROOT / "datasets" / "red_white" / "data.yaml"
    model = YOLO(a.model)
    res = model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                      project=str(ROOT / "runs"), name="red_white", exist_ok=True)
    best = Path(res.save_dir) / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "red_white.pt")
    print(f"copied {best} -> models/red_white.pt")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default=None)
    raise SystemExit(main(p.parse_args()))
