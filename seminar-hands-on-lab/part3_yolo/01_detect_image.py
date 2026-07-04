"""
Part 3.1 - Object detection with YOLO11 on a single image.

The modern deep-learning detector, in contrast to Part 2's hand-crafted Haar
cascades. YOLO11 detects 80 COCO object classes (person, car, dog, laptop, ...)
in one pass - not just faces, and not one class per model.

First run auto-downloads the tiny 'yolo11n.pt' weights (~5 MB).

Run:
    python 01_detect_image.py path/to/photo.jpg
    python 01_detect_image.py path/to/photo.jpg --model yolo11s.pt --conf 0.4
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to an image (or a URL)")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n/s/m/l/x .pt")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--out", default="yolo_detected.jpg")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)               # downloads weights on first use
    results = model(args.image, conf=args.conf)
    r = results[0]

    # summarise detections by class name
    counts = {}
    for c in r.boxes.cls.tolist():
        name = model.names[int(c)]
        counts[name] = counts.get(name, 0) + 1

    print(f"Detected {len(r.boxes)} object(s) in {os.path.basename(args.image)}:")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3} x {name}")

    r.save(filename=args.out)              # writes an annotated copy
    print(f"\nSaved annotated image -> {args.out}")
    print("Compare with Part 2: one model, 80 classes, boxes + labels + confidence "
          "- no separate cascade per object.")


if __name__ == "__main__":
    main()
