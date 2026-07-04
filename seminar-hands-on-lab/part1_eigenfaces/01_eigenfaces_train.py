"""
Part 1.1 - Eigenfaces: learn the face basis with PCA.

Mirrors the seminar's Eigenfaces slide. We:
  1. load a face dataset (Olivetti by default - auto-downloads via scikit-learn)
  2. compute the MEAN face
  3. run PCA to get the top "eigenfaces" (principal components)
  4. save the fitted PCA + labels so 02_recognize.py can use them
  5. plot the mean face + first eigenfaces + a cumulative-variance curve

Datasets (pick with --dataset):
  olivetti  - 400 images, 40 people (default, tiny & instant)
  lfw       - Labeled Faces in the Wild, real photos (larger download)
  kaggle    - your own Kaggle dataset folder (see --data-dir and README)

Run:
    python 01_eigenfaces_train.py                    # Olivetti
    python 01_eigenfaces_train.py --dataset lfw --components 150
    python 01_eigenfaces_train.py --dataset kaggle --data-dir ./faces
"""
import argparse
import os
import pickle

import numpy as np


def load_olivetti():
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces(shuffle=True, random_state=42)
    return d.images, d.target, d.images.shape[1:], [str(t) for t in sorted(set(d.target))]


def load_lfw(min_faces=30):
    from sklearn.datasets import fetch_lfw_people
    d = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.5)
    return d.images, d.target, d.images.shape[1:], list(d.target_names)


def load_kaggle(data_dir):
    """Expect data_dir/<person_name>/*.jpg. Works for most Kaggle face datasets."""
    from PIL import Image
    imgs, labels, names = [], [], []
    people = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    if not people:
        raise SystemExit(f"No person subfolders in {data_dir}. Expected {data_dir}/<name>/*.jpg")
    for idx, person in enumerate(people):
        names.append(person)
        for fn in os.listdir(os.path.join(data_dir, person)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".pgm")):
                im = Image.open(os.path.join(data_dir, person, fn)).convert("L").resize((64, 64))
                imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
                labels.append(idx)
    if not imgs:
        raise SystemExit("No images found under the person folders.")
    return np.array(imgs), np.array(labels), (64, 64), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["olivetti", "lfw", "kaggle"], default="olivetti")
    ap.add_argument("--data-dir", help="folder for --dataset kaggle")
    ap.add_argument("--components", type=int, default=100, help="number of eigenfaces to keep")
    ap.add_argument("--outdir", default="model")
    args = ap.parse_args()

    if args.dataset == "olivetti":
        images, y, shape, names = load_olivetti()
    elif args.dataset == "lfw":
        images, y, shape, names = load_lfw()
    else:
        if not args.data_dir:
            raise SystemExit("--dataset kaggle needs --data-dir")
        images, y, shape, names = load_kaggle(args.data_dir)

    n = images.shape[0]
    X = images.reshape(n, -1)  # flatten each image to a vector
    print(f"Dataset: {args.dataset}  |  {n} images  |  {len(names)} people  |  image {shape}")

    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    k = min(args.components, Xtr.shape[0], Xtr.shape[1])
    print(f"Fitting PCA with {k} components on {Xtr.shape[0]} training faces...")
    pca = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=42).fit(Xtr)

    var = float(pca.explained_variance_ratio_.sum())
    print(f"Top {k} eigenfaces explain {var*100:.1f}% of variance "
          f"({X.shape[1]} pixels -> {k} numbers, a {X.shape[1] // k}x compression)")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "eigenfaces.pkl"), "wb") as f:
        pickle.dump({"pca": pca, "shape": shape, "names": names,
                     "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}, f)
    print(f"Saved model -> {os.path.join(args.outdir, 'eigenfaces.pkl')}")

    # ---- visualise ----
    try:
        import matplotlib.pyplot as plt
        h, w = shape
        fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
        axes[0, 0].imshow(pca.mean_.reshape(h, w), cmap="gray")
        axes[0, 0].set_title("mean face", fontsize=9)
        for i in range(1, 6):
            axes[0, i].imshow(pca.components_[i - 1].reshape(h, w), cmap="gray")
            axes[0, i].set_title(f"eigenface {i}", fontsize=9)
        for i in range(6):
            axes[0, i].axis("off")
        # cumulative variance curve spanning the bottom row
        gs = axes[1, 0].get_gridspec()
        for a in axes[1]:
            a.remove()
        axbig = fig.add_subplot(gs[1, :])
        axbig.plot(np.cumsum(pca.explained_variance_ratio_))
        axbig.set_title("cumulative variance explained", fontsize=9)
        axbig.set_xlabel("# components"); axbig.set_ylabel("variance"); axbig.grid(alpha=0.3)
        fig.tight_layout()
        out_png = os.path.join(args.outdir, "eigenfaces.png")
        fig.savefig(out_png, dpi=120)
        print(f"Saved figure -> {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"(skipped plot: {e})")

    print("\nNext: python 02_recognize.py")


if __name__ == "__main__":
    main()
