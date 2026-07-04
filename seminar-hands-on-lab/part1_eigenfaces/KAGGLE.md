# Using a Kaggle dataset for Eigenfaces (optional)

The scripts default to the **Olivetti** faces dataset, which auto-downloads via
scikit-learn — zero setup. Use this section only if you want real Kaggle data.

## Option A — download in the browser (simplest)

1. Grab any face dataset with one folder per person, e.g.
   - "Face Recognition Dataset" (folders per celebrity)
   - "LFW - People (Face Recognition)"
   - "5 Celebrity Faces Dataset"
2. Unzip so the layout is:
   ```
   faces/
     person_a/  img1.jpg  img2.jpg ...
     person_b/  img1.jpg  img2.jpg ...
   ```
3. Train on it:
   ```bash
   python 01_eigenfaces_train.py --dataset kaggle --data-dir ./faces --components 120
   python 02_recognize.py
   ```

## Option B — Kaggle API (scriptable)

```bash
pip install kaggle
# put your kaggle.json token in ~/.kaggle/ (Account -> Create New API Token)
kaggle datasets download -d vasukipatel/face-recognition-dataset
unzip face-recognition-dataset.zip -d faces
python 01_eigenfaces_train.py --dataset kaggle --data-dir ./faces/"Original Images"/"Original Images"
```

## Tips

- Eigenfaces expects **aligned, cropped, similar-sized** faces. If your dataset
  has full scenes, run Part 2's Haar detector first to crop faces, then feed the
  crops here — that is exactly how the two halves of the seminar connect.
- Grayscale, consistent resolution (the loader resizes to 64x64) matters more
  than sheer image count. Quality over quantity.
