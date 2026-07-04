"""
Part 1.2 - Eigenfaces: recognize a face by nearest neighbour in eigenface space.

Uses the model saved by 01_eigenfaces_train.py. This is the classic pipeline:
  1. project every training face into the low-dim eigenface space (the "weights")
  2. project a test/query face the same way
  3. classify by the nearest training face (Euclidean distance in that space)

Reports accuracy on the held-out test split, and can identify one image.

Run:
    python 02_recognize.py                         # evaluate on the test split
    python 02_recognize.py --image path/to/face.jpg   # identify one face
"""
import argparse
import pickle
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/eigenfaces.pkl")
    ap.add_argument("--image", help="optional: identify a single face image")
    ap.add_argument("--k", type=int, default=1, help="k for k-NN")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}. Run 01_eigenfaces_train.py first.")
    with open(args.model, "rb") as f:
        m = pickle.load(f)
    pca, shape, names = m["pca"], m["shape"], m["names"]

    from sklearn.neighbors import KNeighborsClassifier

    # project training faces into eigenface space -> these are the "weight vectors"
    Ztr = pca.transform(m["Xtr"])
    clf = KNeighborsClassifier(n_neighbors=args.k).fit(Ztr, m["ytr"])

    # ---- evaluate on held-out test faces ----
    Zte = pca.transform(m["Xte"])
    acc = clf.score(Zte, m["yte"])
    print(f"Recognition accuracy on held-out faces: {acc*100:.1f}%  "
          f"({len(m['yte'])} test images, {len(names)} people, {pca.n_components_} eigenfaces)")

    # ---- identify a single supplied image ----
    if args.image:
        from PIL import Image
        h, w = shape
        im = Image.open(args.image).convert("L").resize((w, h))
        x = (np.asarray(im, dtype=np.float32) / 255.0).reshape(1, -1)
        z = pca.transform(x)
        pred = clf.predict(z)[0]
        # distance to nearest neighbour = confidence proxy
        dist, _ = clf.kneighbors(z, n_neighbors=1)
        print(f"\nImage: {args.image}")
        print(f"Predicted identity: {names[pred]}  (nearest-neighbour distance {dist[0][0]:.2f})")
        print("Tip: a large distance means 'probably not anyone I was trained on'. "
              "Set a threshold to reject unknown faces in production.")


if __name__ == "__main__":
    main()
