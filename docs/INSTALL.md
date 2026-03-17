

# Getting Started

## 1. Create Conda Environment

```bash
conda create -n recondrive python=3.10
conda activate recondrive
```

If you do not want to touch your existing Python installation, a local `venv`
works as well:

```bash
cd ReconDrive
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2. Install Dependencies

### Step 1: Automatic Installation

```bash
python -m pip install -r requirements.txt
```

Before running inference, verify that the external assets are present:

```bash
python scripts/check_prereqs.py
```

The repository does not ship usable checkpoints or datasets by default:

- `checkpoints/*.ckpt` and `checkpoints/vggt.pt` may be placeholder symlinks.
- Inference requires a full nuScenes root, not only the bundled
  `samples/DEPTH_MAP` cache.

---

### Step 2: Manual Installation

#### (1) Install PyTorch3D

```bash
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2
conda install ./pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2
```

`torch`, `torchvision`, `torchaudio`, `xformers`, and `pytorch3d` must be
installed as a compatible set. The versions listed in `requirements.txt` and
the sample PyTorch3D package above are not interchangeable across arbitrary CUDA
or Torch builds.

---

#### (2) Install Gaussian Splatting Dependencies

```bash
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

---

#### (3) Install SAM2

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# Download SAM2 checkpoint
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

After downloading your assets, prefer environment-variable overrides instead of
editing scripts:

```bash
CHECKPOINT_PATH=/path/to/recondrive_stage2.ckpt \
VGGT_CHECKPOINT=/path/to/vggt_model.pt \
DATA_PATH=/path/to/nuscenes \
bash scripts/inference.sh
```

For a repo-local smoke test that stays inside `.venv` and uses `nuScenes mini`,
this is the smallest confirmed command:

```bash
cd ReconDrive
source .venv/bin/activate
CHECKPOINT_PATH=$PWD/downloaded_checkpoints/recondrive_stage2.ckpt \
VGGT_CHECKPOINT=$PWD/downloaded_checkpoints/vggt_model.pt \
DATA_PATH=$PWD/data/nuscenes-mini \
MAX_SAMPLES_PER_SCENE=1 \
SAVE_NOVEL_RENDERS=false \
bash scripts/inference.sh
```

