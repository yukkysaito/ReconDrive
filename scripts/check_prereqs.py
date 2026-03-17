#!/usr/bin/env python3

"""Lightweight environment checker for ReconDrive inference."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "yaml",
    "numpy",
    "torch",
    "torchvision",
    "pytorch_lightning",
    "gsplat",
    "nuscenes",
    "lpips",
    "cv2",
    "pyquaternion",
    "einops",
    "huggingface_hub",
    "kornia",
    "skimage",
    "trimesh",
]

OPTIONAL_MODULES = [
    "sam2",
]


def check_module(name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, "missing"
    return True, "ok"


def describe_path(path: Path) -> str:
    exists = path.exists()
    if path.is_symlink():
        target = os.path.realpath(path)
        return f"symlink -> {target} ({'ok' if exists else 'broken'})"
    if exists:
        return "ok"
    return "missing"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Check local ReconDrive prerequisites")
    parser.add_argument(
        "--data-path",
        default="/data/datasets/nuscenes",
        help="Expected nuScenes dataset root",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(repo_root / "checkpoints" / "recondrive_stage2.ckpt"),
        help="ReconDrive stage-2 checkpoint path",
    )
    parser.add_argument(
        "--vggt-checkpoint",
        default=str(repo_root / "checkpoints" / "vggt.pt"),
        help="VGGT checkpoint path",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default=str(repo_root / "checkpoints" / "sam2.1_hiera_small.pt"),
        help="SAM2 checkpoint path",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print()

    print("Module checks:")
    missing_modules = []
    for name in REQUIRED_MODULES:
        ok, status = check_module(name)
        print(f"  - {name}: {status}")
        if not ok:
            missing_modules.append(name)

    if OPTIONAL_MODULES:
        print()
        print("Optional module checks:")
        for name in OPTIONAL_MODULES:
            _, status = check_module(name)
            print(f"  - {name}: {status}")

    print()
    print("Asset checks:")
    required_asset_paths = [
        ("data_path", Path(args.data_path)),
        ("recondrive_stage2", Path(args.checkpoint)),
        ("vggt", Path(args.vggt_checkpoint)),
    ]
    optional_asset_paths = [
        ("sam2", Path(args.sam2_checkpoint)),
    ]
    missing_assets = []
    for label, path in required_asset_paths:
        status = describe_path(path)
        print(f"  - {label}: {path} [{status}]")
        if "missing" in status or "broken" in status:
            missing_assets.append(label)

    if optional_asset_paths:
        print()
        print("Optional asset checks:")
        for label, path in optional_asset_paths:
            status = describe_path(path)
            print(f"  - {label}: {path} [{status}]")

    print()
    if missing_modules or missing_assets:
        print("Result: prerequisites are incomplete.")
        if missing_modules:
            print("Missing modules:", ", ".join(missing_modules))
        if missing_assets:
            print("Missing assets:", ", ".join(missing_assets))
        return 1

    print("Result: basic prerequisites look present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
