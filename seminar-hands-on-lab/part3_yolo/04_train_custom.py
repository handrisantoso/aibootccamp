"""
Part 3.4 - Train YOLO11 on your OWN objects (transfer learning).

The detection analogue of Part 4's VLM fine-tuning: start from pretrained
yolo11n weights and adapt them to custom classes with a small labelled dataset.

DATA FORMAT (Ultralytics/YOLO):
    my_dataset/
      images/train/*.jpg   images/val/*.jpg
      labels/train/*.txt   labels/val/*.txt   # one .txt per image
    data.yaml                                 # points to the folders + class names

Each label .txt line:  <class_id> <x_center> <y_center> <width> <height>
(all normalised 0-1). Tools like Roboflow, CVAT or LabelImg export this directly.

A minimal data.yaml:
    path: ./my_dataset
    train: images/train
    val: images/val
    names:
      0: helmet
      1: no_helmet

Run:
    python 04_train_custom.py --data my_dataset/data.yaml --epochs 50
    python 04_train_custom.py --data data.yaml --epochs 50 --model yolo11s.pt --imgsz 640
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="custom_yolo11")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"data.yaml not found: {args.data}\nSee this file's header for the layout.")

    from ultralytics import YOLO

    # start from pretrained weights = transfer learning (best practice for small data)
    model = YOLO(args.model)

    print(f"Training {args.model} on {args.data} for {args.epochs} epochs...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=10,        # early stop if val stops improving (avoid overfitting)
    )

    # evaluate on the val split
    metrics = model.val()
    print(f"\nDone. Best weights: runs/detect/{args.name}/weights/best.pt")
    print(f"val mAP50-95: {metrics.box.map:.3f}   mAP50: {metrics.box.map50:.3f}")
    print("\nBest practices (same spirit as VLM fine-tuning in Part 4):")
    print("  - start from pretrained weights, don't train from scratch")
    print("  - clean train/val split, balanced classes, cover edge cases")
    print("  - watch val mAP; use --patience early stopping to avoid overfitting")
    print("  - predict with your model:  YOLO('runs/detect/%s/weights/best.pt')" % args.name)


if __name__ == "__main__":
    main()
