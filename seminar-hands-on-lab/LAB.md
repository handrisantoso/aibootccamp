# Hands-On Lab — From Pixels to Prompts

**The full seminar, runnable end to end: Eigenfaces → Haar detection → Vision-Language Models.**

This lab walks the same arc as the talk. You will recognise faces with **PCA/Eigenfaces**, detect faces and people with **Haar cascades**, then use a modern **VLM** (via Ollama) for captioning, structured extraction, RAG, and finally fine-tune one with LoRA. Each part stands alone, but together they show *why* the field moved from hand-crafted features to promptable multimodal models.

---

## Map of the lab

| Part | Topic | Folder | Needs |
|------|-------|--------|-------|
| 1 | Eigenfaces — face **recognition** with PCA | `part1_eigenfaces/` | CPU only |
| 2 | Haar cascades — face & people **detection** | `part2_haar/` | CPU only |
| 3 | YOLO11 — modern **object detection** | `part3_yolo/` | CPU (fast) / GPU (video, training) |
| 4 | VLMs — captioning, RAG, extraction, **fine-tuning** | `part4_vlm/` | CPU (use) / GPU (tune) |

The through-line: **Part 2 detects and crops faces → Part 1 recognises them → Part 3 detects 80 object classes with one deep model → Part 4 shows how one VLM replaces the whole zoo of task-specific models.**

---

## Setup

Each part uses its own dependencies. **Best practice: create a fresh virtual
environment** so heavy libraries (PyTorch, OpenCV) don't collide with your other
projects. A full venv + conda walkthrough is in `part3_yolo/SETUP.md` — the same
pattern works for every part.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate

# Parts 1-2 (classical CV)
pip install -r requirements-classical.txt

# Part 3 (YOLO object detection)
pip install -r requirements-yolo.txt

# Part 4 (VLM use) - also install Ollama from https://ollama.com
pip install -r requirements.txt

# Part 4 fine-tuning (GPU) - only when you get there
pip install -r requirements-train.txt
```

---

# Part 1 — Eigenfaces (face recognition with PCA)

The 1991 Turk & Pentland approach from the seminar: represent any face as a
weighted sum of a few dozen "eigenfaces," then recognise by nearest neighbour in
that compact space.

**Dataset:** defaults to **Olivetti** (400 faces, auto-downloads via
scikit-learn — zero setup). Swap in **LFW** or a **Kaggle** dataset any time —
see `part1_eigenfaces/KAGGLE.md`.

```bash
cd part1_eigenfaces

# 1. learn the eigenface basis + visualise mean face / eigenfaces / variance curve
python 01_eigenfaces_train.py                     # Olivetti
# python 01_eigenfaces_train.py --dataset lfw --components 150
# python 01_eigenfaces_train.py --dataset kaggle --data-dir ./faces

# 2. recognise: accuracy on held-out faces, or identify one image
python 02_recognize.py
python 02_recognize.py --image some_face.jpg
```

**What to notice:** thousands of pixels collapse to ~100 numbers, yet recognition
still works well. That "compress to a few numbers" idea is the ancestor of every
modern embedding. **Limit:** it needs aligned, cropped, similar-lit faces — which
is exactly what Part 2 gives us.

---

# Part 2 — Haar cascades (face & people detection)

Viola-Jones (2001): the first real-time face detector, built on Haar contrast
features. OpenCV bundles the pre-trained cascades — nothing to download.

```bash
cd part2_haar

# faces (add --eyes to also box eyes)
python 01_detect_faces.py photo.jpg --eyes --out faces_detected.jpg

# people: compare the Haar full-body cascade vs. the HOG+SVM pedestrian detector
python 02_detect_people.py street.jpg --method hog --out people.jpg
python 02_detect_people.py street.jpg --method haar

# bridge to Part 1: detect + crop faces into the Eigenfaces input layout
python 03_crop_faces_for_eigenfaces.py --in ./raw_photos --out ./faces
python ../part1_eigenfaces/01_eigenfaces_train.py --dataset kaggle --data-dir ./faces
```

**What to notice:** `scaleFactor` and `minNeighbors` trade misses against false
positives — the practical face of the "cascade rejects empty windows early" idea.
**Limit:** each cascade does exactly one thing (frontal faces, or upright bodies)
and breaks on pose/lighting it never saw.

---

# Part 3 — YOLO11 (modern object detection)

Where Haar detects one class per cascade, **YOLO11** detects 80 object classes
(person, car, dog, laptop, ...) in a single deep-network pass — the leap from
hand-crafted features to learned ones.

> **New to Python environments? Read `part3_yolo/SETUP.md` first** — it walks
> through creating a fresh `venv` or `conda` environment and installing YOLO,
> step by step. This is the recommended way to keep heavy ML libraries isolated.

```bash
cd part3_yolo
# after activating your env and: pip install -r ../requirements-yolo.txt
yolo checks                                  # confirm install + CPU/GPU

# 1. detect everything in an image (weights auto-download on first run)
python 01_detect_image.py photo.jpg --out yolo_detected.jpg

# 2. real-time on webcam or a video file
python 02_detect_video_webcam.py --source 0
python 02_detect_video_webcam.py --source clip.mp4 --classes 0   # people only

# 3. count people — compare directly with Part 2's Haar/HOG
python 03_count_people.py street.jpg

# 4. TRAIN on your own objects (transfer learning from pretrained weights)
python 04_train_custom.py --data my_dataset/data.yaml --epochs 50
```

**What to notice:** one model, 80 classes, boxes + labels + confidence, and you
fine-tune it on custom objects the same way you fine-tune a VLM in Part 4 — start
from pretrained weights, keep a clean train/val split, watch mAP, early-stop.
**Still a limit:** YOLO detects *objects from a fixed vocabulary*; it can't answer
"what's unusual here?" or read a document. That's the gap Part 4 closes.

---

# Part 4 — Vision-Language Models

One model, many tasks, steered by language. We start by just *using* a VLM, then
climb the ladder to fine-tuning. Full detail for each step is in the script
headers; the ladder is:

```
Use it → Prompt well → Structured output → Ground (RAG) → Fine-tune → Evaluate → Deploy
```

### 4.0 Setup Ollama

```bash
# install Ollama (https://ollama.com), then:
ollama pull llava:7b            # ~4.7 GB general VLM
ollama pull nomic-embed-text    # embeddings, for the RAG step
# smaller option: ollama pull moondream
```

### 4.1 Use it — captioning

```bash
cd part4_vlm
python 00_image_captioning.py photo.jpg     # 3 caption styles, one model
python 01_ollama_basic.py photo.jpg         # client vs raw HTTP
```

### 4.2 Prompt it well + structured output

```bash
python 02_prompting.py receipt.jpg          # vague vs specific vs role vs few-shot
python 03_structured_output.py receipt.jpg  # force + validate JSON, retry on fail
```

### 4.3 Ground it — RAG over images + a knowledge base

```bash
python 04_rag_grounding.py product.jpg "Is this compliant with our packaging policy?"
```

Describes the image, retrieves matching policy from `../data/knowledge_base.json`,
then re-asks the VLM with that context so the answer is grounded and cited.

### 4.4 Package a prompt (no training)

```bash
ollama create receipt-reader -f Modelfile
ollama run receipt-reader ./../data/images/receipt.jpg
```

### 4.5 Fine-tune with LoRA (GPU)

> Needs a GPU (~16 GB for Qwen2-VL-2B via QLoRA). No local GPU? Run on Colab /
> RunPod / Lambda — upload `part4_vlm/` and `data/`.

```bash
pip install -r ../requirements-train.txt

python 05_prepare_dataset.py --in ../data/sample_dataset.json --outdir ../data/prepared
python 06_finetune_qwen2vl_lora.py --model Qwen/Qwen2-VL-2B-Instruct \
       --data ../data/prepared --output ../out/lora --epochs 2 --batch-size 1 --grad-accum 8
python 07_evaluate.py --base Qwen/Qwen2-VL-2B-Instruct \
       --adapter ../out/lora --test ../data/prepared/test.jsonl
python 08_export_to_ollama.py --base Qwen/Qwen2-VL-2B-Instruct \
       --adapter ../out/lora --outdir ../out/merged
```

**Best practices enforced by the scripts:** clean train/val/test split with no
image leakage, freeze the base model and train only LoRA adapters, small LR / few
epochs, and evaluate base-vs-tuned (accuracy, valid-JSON rate, a "don't-know"
rate, and a regression check for catastrophic forgetting) before you ship.

---

## The big picture

| | Eigenfaces | Haar | YOLO11 | VLM |
|---|---|---|---|---|
| Era | 1991 | 2001 | 2020s | 2023+ |
| Features | PCA (linear) | hand-designed contrast | learned, deep | learned, deep, multimodal |
| Tasks | recognise faces | detect faces/people | detect 80 object classes | *anything you can describe* |
| New task means | re-train on new faces | a different cascade | re-train on new classes | **change the prompt** |

Each column is more general than the last. That final column — solve a new task
by *changing the prompt* — is the punchline of the whole seminar.

---

## Datasets & assets

- **Eigenfaces:** Olivetti / LFW auto-download; Kaggle route in `part1_eigenfaces/KAGGLE.md`.
- **Haar:** cascades ship with `opencv-python`; bring your own photos.
- **YOLO11:** `yolo11n.pt` weights auto-download on first run; trains on COCO or
  your own dataset (see `part3_yolo/04_train_custom.py`).
- **VLM:** models pulled via Ollama; sample RAG knowledge base and fine-tuning
  dataset are in `data/`. Drop your own images in `data/images/`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Olivetti/LFW download fails | Check network; scikit-learn caches to `~/scikit_learn_data` |
| Haar finds no faces | Lower `--neighbors`, ensure frontal faces, try a clearer photo |
| YOLO `torch.cuda.is_available()` False | CPU still works; for GPU install a CUDA torch build (see `part3_yolo/SETUP.md`) |
| `yolo: command not found` | Activate your env; `pip install -r requirements-yolo.txt` |
| `connection refused` :11434 | Start Ollama: `ollama serve` |
| CUDA OOM in fine-tuning | Use the 2B model, `--batch-size 1`, raise `--grad-accum`, keep 4-bit |
