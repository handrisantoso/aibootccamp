"""
Part 2.3 - Bridge Haar -> Eigenfaces.

This connects the two classical halves of the seminar: use the Haar detector to
FIND and CROP faces out of ordinary photos, producing the aligned, same-size
crops that Eigenfaces (Part 1) needs as input.

Point it at a folder of photos (optionally one subfolder per person) and it
writes cropped 64x64 grayscale faces in the person-folder layout that
part1_eigenfaces/01_eigenfaces_train.py --dataset kaggle expects.

Run:
    python 03_crop_faces_for_eigenfaces.py --in ./raw_photos --out ./faces
    # then:
    python ../part1_eigenfaces/01_eigenfaces_train.py --dataset kaggle --data-dir ./faces
"""
import argparse
import os

import cv2


def iter_images(root):
    """Yield (person_label, image_path). If root has subfolders, use them as labels."""
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if subdirs:
        for d in sorted(subdirs):
            for fn in os.listdir(os.path.join(root, d)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    yield d, os.path.join(root, d, fn)
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                yield "unknown", os.path.join(root, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="folder of photos")
    ap.add_argument("--out", required=True, help="output folder for cropped faces")
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    n_faces = 0
    for label, path in iter_images(args.inp):
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        out_dir = os.path.join(args.out, label)
        os.makedirs(out_dir, exist_ok=True)
        for i, (x, y, w, h) in enumerate(faces):
            crop = cv2.resize(gray[y:y + h, x:x + w], (args.size, args.size))
            base = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(os.path.join(out_dir, f"{base}_{i}.png"), crop)
            n_faces += 1

    print(f"Cropped {n_faces} face(s) -> {args.out}")
    print("These aligned crops are exactly what Eigenfaces wants. "
          "This is the classic 'detect, then recognize' pipeline.")


if __name__ == "__main__":
    main()
