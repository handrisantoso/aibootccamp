"""
Part 2.2 - Detect PEOPLE (full body).

Two classic detectors, so you can compare:
  --method haar : OpenCV Haar full-body cascade (fast, weaker)
  --method hog  : HOG + linear SVM pedestrian detector (the other landmark
                  hand-crafted-feature approach; usually better for people)

Both are hand-crafted-feature detectors from the pre-deep-learning era - the
perfect contrast to the VLMs in Part 3.

Run:
    python 02_detect_people.py street.jpg
    python 02_detect_people.py street.jpg --method hog --out people.jpg
"""
import argparse
import os

import cv2


def detect_haar(gray, img):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 80))
    return [(x, y, w, h) for (x, y, w, h) in boxes]


def detect_hog(gray, img):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return [(x, y, w, h) for (x, y, w, h) in rects]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--method", choices=["haar", "hog"], default="hog")
    ap.add_argument("--out", default="people_detected.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = detect_haar(gray, img) if args.method == "haar" else detect_hog(gray, img)
    print(f"[{args.method}] detected {len(boxes)} person/people in {os.path.basename(args.image)}")

    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (85, 102, 224), 2)

    cv2.imwrite(args.out, img)
    print(f"Saved -> {args.out}")
    print("\nTry both methods on the same street photo. Notice HOG usually wins on "
          "upright pedestrians - and that BOTH are far less flexible than a VLM you "
          "can simply ask 'how many people are in this image?'")


if __name__ == "__main__":
    main()
