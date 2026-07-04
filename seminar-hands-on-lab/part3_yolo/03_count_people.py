"""
Part 3.3 - Count people (and compare with Part 2).

Filters YOLO detections to the 'person' class (COCO id 0) and counts them - the
same task you tried with Haar/HOG in Part 2, now far more robust to pose,
lighting and crowding.

Run:
    python 03_count_people.py street.jpg
    python 03_count_people.py street.jpg --conf 0.4 --out counted.jpg
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default="people_yolo.jpg")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    # classes=[0] restricts detection to 'person'
    results = model(args.image, conf=args.conf, classes=[0])
    r = results[0]

    n = len(r.boxes)
    print(f"YOLO11 counted {n} person/people in {os.path.basename(args.image)} "
          f"(conf >= {args.conf})")
    r.save(filename=args.out)
    print(f"Saved -> {args.out}")
    print("\nRun Part 2's 02_detect_people.py on the SAME image to compare: YOLO "
          "generally finds more, with tighter boxes and fewer false positives.")


if __name__ == "__main__":
    main()
