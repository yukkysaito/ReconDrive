# ReconDrive MCAP Runbook

## Overview

This repository now supports a repo-local smoke workflow that converts a ROS 2
`mcap` bag into a ReconDrive-friendly dataset and runs inference from it.

Current assumptions:

- Python dependencies for ReconDrive live in `./.venv`
- ROS 2 bag preprocessing uses the system ROS Python environment
- The MCAP workflow below uses `camera0` to `camera5`
- The current smoke config disables vehicle flow: `use_vehicle_flow: false`

---

## 1. Repo-Local Setup

### Create and activate `.venv`

```bash
cd /path/to/ReconDrive
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python dependencies into `.venv`

```bash
./.venv/bin/pip install -r requirements.txt
```

### Check prerequisites

```bash
./.venv/bin/python scripts/check_prereqs.py
```

---

## 2. Checkpoints

Place the model files in repo-local paths:

- `downloaded_checkpoints/recondrive_stage2.ckpt`
- `downloaded_checkpoints/vggt_model.pt`

The smoke configs in this repo already point at those paths.

---

## 3. Optional NuScenes Mini Smoke

If you want to confirm the base environment before touching MCAP, use the
NuScenes mini config:

```bash
cd /path/to/ReconDrive
source .venv/bin/activate

CHECKPOINT_PATH=$PWD/downloaded_checkpoints/recondrive_stage2.ckpt \
VGGT_CHECKPOINT=$PWD/downloaded_checkpoints/vggt_model.pt \
DATA_PATH=$PWD/data/nuscenes-mini \
MAX_SAMPLES_PER_SCENE=1 \
SAVE_NOVEL_RENDERS=false \
bash scripts/inference.sh
```

---

## 4. MCAP Preprocessing

### What the preprocessor reads

`scripts/prepare_mcap_dataset.py` reads:

- `/sensing/camera/camera0..5/image_raw/compressed`
- `/sensing/camera/camera0..5/camera_info`
- `/tf_static`
- `/tf` with `map -> base_link`

It builds:

- synchronized image frames
- per-camera intrinsics `K`
- camera-to-ego extrinsics `c2e_extr`
- ego poses from `map -> base_link`

### Example command

```bash
cd /path/to/ReconDrive
source /opt/ros/humble/setup.bash

python3 scripts/prepare_mcap_dataset.py \
  --mcap /path/to/input.mcap \
  --output ./data/mcap_camera0_5_smoke \
  --cameras camera0,camera1,camera2,camera3,camera4,camera5 \
  --max_sync_frames 12 \
  --context_span 6 \
  --sync_tolerance_ms 40
```

### Output

The preprocessor writes:

- `data/mcap_camera0_5_smoke/metadata.json`
- `data/mcap_camera0_5_smoke/images/camera*/<timestamp>.jpg`

The current smoke extraction produces:

- `12` synchronized frames
- `5` valid ReconDrive temporal windows

To process a longer slice of the bag, increase `--max_sync_frames`.

---

## 5. Run ReconDrive on the MCAP-Derived Dataset

Use the dedicated config:

- `configs/mcap/recondrive_camera0_5_smoke.yaml`

### Example command

```bash
cd /path/to/ReconDrive
source .venv/bin/activate

./.venv/bin/python -m scripts.inference \
  --cfg_path configs/mcap/recondrive_camera0_5_smoke.yaml \
  --restore_ckpt downloaded_checkpoints/recondrive_stage2.ckpt \
  --vggt_checkpoint downloaded_checkpoints/vggt_model.pt \
  --data_path data/mcap_camera0_5_smoke \
  --output_dir work_dirs/recondrive_mcap_camera0_5_smoke1 \
  --device 0 \
  --max_scenes 1 \
  --max_samples_per_scene 1 \
  --frame_skip 1 \
  --no_renders
```

### Notes

- `--no_renders` disables extra translated novel-view rendering
- GT/pred image pairs are still saved under `gt_views`
- The MCAP config drops full-resolution originals before GPU transfer to avoid
  running out of VRAM on the current GPU

---

## 6. How To Visualize Results

### A. Metrics summary

The main numeric outputs are:

- `work_dirs/recondrive_mcap_camera0_5_smoke1/inference_summary.json`
- `work_dirs/recondrive_mcap_camera0_5_smoke1/inference_detailed.json`
- `work_dirs/recondrive_mcap_camera0_5_smoke1/scene_<scene>_evaluation.json`

### B. Predicted vs GT images

Even with `--no_renders`, ReconDrive saves image pairs here:

```text
work_dirs/recondrive_mcap_camera0_5_smoke1/
  <scene_name>/
    sample_0000/
      gt_views/
        original_cam_0_pred.png
        original_cam_0_gt.png
        ...
```

For the current run, an example directory is:

```text
work_dirs/recondrive_mcap_camera0_5_smoke1/<scene_name>/sample_0000/gt_views
```

You can inspect those `*_pred.png` and `*_gt.png` files directly with any image
viewer.

### C. Novel-view render outputs

If you want translated novel-view images like `left_1.0m` / `right_1.0m`, rerun
without `--no_renders`:

```bash
cd /path/to/ReconDrive
source .venv/bin/activate

./.venv/bin/python -m scripts.inference \
  --cfg_path configs/mcap/recondrive_camera0_5_smoke.yaml \
  --restore_ckpt downloaded_checkpoints/recondrive_stage2.ckpt \
  --vggt_checkpoint downloaded_checkpoints/vggt_model.pt \
  --data_path data/mcap_camera0_5_smoke \
  --output_dir work_dirs/recondrive_mcap_camera0_5_renders \
  --device 0 \
  --max_scenes 1 \
  --max_samples_per_scene 1 \
  --frame_skip 1
```

Those images are saved under paths like:

```text
work_dirs/recondrive_mcap_camera0_5_renders/<scene_name>/sample_0000/left_1.0m/
work_dirs/recondrive_mcap_camera0_5_renders/<scene_name>/sample_0000/right_1.0m/
```

### D. Fast sanity check from the terminal

To list the generated PNGs:

```bash
find work_dirs/recondrive_mcap_camera0_5_smoke1 -type f | sort
```

---

## 7. Files Added For MCAP Support

- `scripts/prepare_mcap_dataset.py`
- `dataset/mcap_scene_dataset.py`
- `configs/mcap/recondrive_camera0_5_smoke.yaml`

The scene data module now switches dataset type via:

- `data_cfg.dataset_type: mcap_preprocessed`

---

## 8. Current Status

The following path has been validated as a working smoke flow:

1. preprocess the provided MCAP into `data/mcap_camera0_5_smoke`
2. run GPU inference with `configs/mcap/recondrive_camera0_5_smoke.yaml`
3. inspect the JSON metrics and the `gt_views/*.png` image pairs
