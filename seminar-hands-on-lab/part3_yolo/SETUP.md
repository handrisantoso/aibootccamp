# Setting up a fresh Python environment for YOLO

Object-detection libraries pull in heavy dependencies (PyTorch, CUDA, OpenCV).
**Always use a dedicated virtual environment** so they don't collide with your
other projects. Two ways — pick whichever you already have.

> Check your Python first: `python --version` (need 3.9–3.12 for YOLO11).

---

## Option A — `venv` (built into Python, nothing to install)

### macOS / Linux
```bash
# 1. create an isolated environment in a folder called .yolo-env
python -m venv .yolo-env

# 2. activate it (your prompt will show (.yolo-env))
source .yolo-env/bin/activate

# 3. upgrade pip, then install YOLO
python -m pip install --upgrade pip
pip install -r ../requirements-yolo.txt

# 4. when you're done for the day
deactivate
```

### Windows (PowerShell)
```powershell
python -m venv .yolo-env
.yolo-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ..\requirements-yolo.txt
# later:  deactivate
```

---

## Option B — `conda` (if you use Anaconda/Miniconda)

```bash
# 1. create a named environment with a specific Python version
conda create -n yolo python=3.11 -y

# 2. activate it
conda activate yolo

# 3. install YOLO (pip works fine inside a conda env)
pip install -r ../requirements-yolo.txt

# later:  conda deactivate
```

---

## Verify the install

```bash
# CLI check - prints versions and your device (CPU vs CUDA GPU)
yolo checks
```

Or from Python:
```bash
python -c "import torch, ultralytics; \
print('ultralytics', ultralytics.__version__); \
print('torch', torch.__version__); \
print('GPU available:', torch.cuda.is_available())"
```

- **`GPU available: True`** → detection and training will use your NVIDIA GPU.
- **`False`** → everything still works on CPU. Image detection is fast; video and
  training are slower but fine for the lab.

---

## GPU notes (optional)

`pip install ultralytics` gives you a CPU/GPU build of PyTorch that works out of
the box. If `torch.cuda.is_available()` is `False` but you *have* an NVIDIA GPU,
install a CUDA-matched PyTorch build:

```bash
# example for CUDA 12.1 - see https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Common issues

| Symptom | Fix |
|---------|-----|
| `command not found: python` | Try `python3` instead |
| `Activate.ps1 cannot be loaded` (Windows) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| Model weights download slowly | `yolo11n.pt` (~5 MB) auto-downloads on first run; just wait once |
| `torch.cuda.is_available()` False on a GPU box | Install a CUDA-matched torch build (above) |

Once `yolo checks` looks good, head back to the YOLO steps in `../LAB.md`.
