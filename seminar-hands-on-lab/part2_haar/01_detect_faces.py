"""
Part 2.1 - Haar cascade FACE detection (Viola-Jones).

The classic real-time face detector from the seminar. OpenCV ships the
pre-trained Haar cascades, so there is nothing to download or train.

Draws a box around each detected face and (optionally) eyes, then saves an
annotated image. Also prints how many faces were found.

Run:
    python 01_detect_faces.py path/to/photo.jpg
    python 01_detect_faces.py path/to/photo.jpg --eyes --out annotated.jpg
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--eyes", action="store_true", help="also detect eyes inside each face")
    ap.add_argument("--out", default="faces_detected.jpg")
    ap.add_argument("--scale", type=float, default=1.1, help="scaleFactor: smaller=more thorough, slower")
    ap.add_argument("--neighbors", type=int, default=5, help="minNeighbors: higher=fewer false positives")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_dir = cv2.data.haarcascades  # bundled with opencv-python
    face_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=args.scale, minNeighbors=args.neighbors, minSize=(30, 30))
    print(f"Detected {len(faces)} face(s) in {os.path.basename(args.image)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 120), 2)
        if args.eyes:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            for (ex, ey, ew, eh) in eye_cascade.detectMultiScale(roi_gray, 1.1, 6):
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (245, 166, 35), 2)

    cv2.imwrite(args.out, img)
    print(f"Saved annotated image -> {args.out}")
    print("\nTune scaleFactor/minNeighbors if you get misses or false boxes. "
          "This is the same idea as the seminar's 'cascade rejects empty windows early'.")


if __name__ == "__main__":
    main()
