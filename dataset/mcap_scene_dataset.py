#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import json
import os
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from dataset.data_util import img_loader, align_dataset, stack_sample


class McapSceneDataset(Dataset):
    def __init__(
        self,
        path,
        stage,
        cameras=None,
        back_context=0,
        forward_context=1,
        data_transform=None,
        depth_type=None,
        with_pose=None,
        with_ego_pose=None,
        with_mask=None,
        min_context_num=1,
        num_context_timesteps=4,
        num_target_timesteps=4,
        cache_dir="",
        nuscenes_version="mcap_preprocessed",
        context_span=6,
    ):
        super().__init__()
        self.path = path
        self.stage = stage
        self.cameras = cameras or []
        self.context_span = context_span
        self.data_transform = data_transform
        self.with_depth = depth_type is not None
        self.with_pose = with_pose
        self.with_ego_pose = with_ego_pose
        self.with_mask = with_mask

        if self.with_depth:
            raise ValueError("McapSceneDataset does not provide gt_depth")
        if self.with_mask:
            raise ValueError("McapSceneDataset does not provide masks")

        metadata_path = os.path.join(self.path, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        meta_cameras = metadata.get("cameras", [])
        if self.cameras:
            missing_cameras = [camera for camera in self.cameras if camera not in meta_cameras]
            if missing_cameras:
                raise ValueError(f"Cameras not found in metadata: {missing_cameras}")
        else:
            self.cameras = meta_cameras

        self.scene_frames = []
        self.scene_names = []
        self.scene_tokens = []
        self.scenes_data = []

        for scene in metadata.get("scenes", []):
            frames = scene.get("frames", [])
            valid_windows = len(frames) - self.context_span - 1
            if valid_windows <= 0:
                continue

            self.scene_frames.append(frames)
            self.scene_names.append(scene["scene_name"])
            self.scene_tokens.append(scene["scene_token"])
            self.scenes_data.append(list(range(valid_windows)))

        print(f"Number of scenes: {len(self.scene_frames)}")
        print(f"Total samples: {sum(len(scene) for scene in self.scenes_data)}")

    def __len__(self):
        return sum(len(scene) for scene in self.scenes_data)

    def get_num_scenes(self):
        return len(self.scene_frames)

    def get_scene_length(self, scene_idx):
        return len(self.scenes_data[scene_idx])

    def get_scene_name(self, scene_idx):
        return self.scene_names[scene_idx]

    def get_scene_token(self, scene_idx):
        return self.scene_tokens[scene_idx]

    @staticmethod
    def _to_numpy(value):
        return value.cpu().numpy() if isinstance(value, torch.Tensor) else value

    def compute_frame_transforms(self, ego_pose_0, ego_pose_n, c2e_extr):
        ego_pose_0 = self._to_numpy(ego_pose_0)
        ego_pose_n = self._to_numpy(ego_pose_n)
        c2e_extr = self._to_numpy(c2e_extr)

        world_to_ego_0 = np.linalg.inv(ego_pose_0)
        world_to_ego_n = np.linalg.inv(ego_pose_n)
        e2c_extr = np.linalg.inv(c2e_extr)
        cam_t_cam = e2c_extr @ world_to_ego_n @ ego_pose_0 @ c2e_extr
        ego_t_ego = c2e_extr @ cam_t_cam @ e2c_extr

        ego_t_ego[3, :] = [0, 0, 0, 1]
        cam_t_cam[3, :] = [0, 0, 0, 1]

        return ego_t_ego.astype(np.float64), cam_t_cam.astype(np.float64)

    def _frame_camera_entry(self, scene_idx, frame_idx, camera):
        frame = self.scene_frames[scene_idx][frame_idx]
        return frame["cameras"][camera], frame

    def get_frame(self, idx, frame_idx, scene_idx, scene_token="", scene_name=""):
        sample = []
        contexts = []

        for camera in self.cameras:
            camera_entry, frame = self._frame_camera_entry(scene_idx, frame_idx, camera)
            image_path = os.path.join(self.path, camera_entry["image_path"])
            data = {
                "idx": idx,
                "token": str(frame["timestamp"]),
                "scene_token": scene_token,
                "scene_name": scene_name,
                "scene_idx": scene_idx,
                "sensor_name": camera,
                "contexts": contexts,
                "filename": image_path,
                "rgb": img_loader(image_path),
                "intrinsics": np.array(camera_entry["K"], dtype=np.float32),
                "timestamp": np.array(camera_entry["image_timestamp"], dtype=np.int64),
            }

            if self.with_pose:
                data["extrinsics"] = np.array(camera_entry["c2e_extr"], dtype=np.float32)
            if self.with_ego_pose:
                data["ego_pose"] = np.array(frame["ego_pose"], dtype=np.float64)

            sample.append(data)

        if self.data_transform:
            sample = [self.data_transform(entry) for entry in sample]

        sample = stack_sample(sample)
        sample = align_dataset(sample, contexts)
        sample.pop(("color_org", 0), None)
        return sample

    def __getitem__(self, sample_idx: int, scene_idx: int = -1) -> Dict[str, Any]:
        if scene_idx < 0:
            if len(self.scene_frames) != 1:
                raise ValueError("scene_idx must be provided when multiple scenes exist")
            scene_idx = 0

        start_frame_idx = self.scenes_data[scene_idx][sample_idx]
        scene_token = self.scene_tokens[scene_idx]
        scene_name = self.scene_names[scene_idx]

        cur_sample = self.get_frame(
            idx=f"{scene_idx}_{sample_idx}",
            frame_idx=start_frame_idx,
            scene_idx=scene_idx,
            scene_token=scene_token,
            scene_name=scene_name,
        )

        target_dict_list = []
        for offset in range(1, self.context_span):
            target_dict_list.append(
                self.get_frame(
                    idx=f"{scene_idx}_{sample_idx}_{offset}",
                    frame_idx=start_frame_idx + offset,
                    scene_idx=scene_idx,
                    scene_token=scene_token,
                    scene_name=scene_name,
                )
            )

        context_dict_list = [
            self.get_frame(
                idx=f"{scene_idx}_{sample_idx}_ctx",
                frame_idx=start_frame_idx + self.context_span,
                scene_idx=scene_idx,
                scene_token=scene_token,
                scene_name=scene_name,
            )
        ]

        if target_dict_list:
            target_dict = default_collate(target_dict_list)
            for key, value in target_dict.items():
                if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                    target_dict[key] = torch.cat([entry for entry in value], dim=0)
        else:
            target_dict = {}

        if self.with_ego_pose and "ego_pose" in cur_sample and "c2e_extr" in cur_sample:
            num_cameras = cur_sample["ego_pose"].shape[0]

            for offset, target_dict in enumerate(target_dict_list, start=1):
                ego_t_ego_list = []
                cam_t_cam_list = []
                for cam_idx in range(num_cameras):
                    ego_pose_0 = cur_sample["ego_pose"][cam_idx]
                    ego_pose_n = target_dict["ego_pose"][cam_idx]
                    c2e_extr = cur_sample["c2e_extr"][cam_idx]
                    ego_t_ego, cam_t_cam = self.compute_frame_transforms(
                        ego_pose_0,
                        ego_pose_n,
                        c2e_extr,
                    )
                    ego_t_ego_list.append(torch.from_numpy(ego_t_ego).float())
                    cam_t_cam_list.append(torch.from_numpy(cam_t_cam).float())
                target_dict[("ego_T_ego", 0, offset)] = torch.stack(ego_t_ego_list, dim=0)
                target_dict[("cam_T_cam", 0, offset)] = torch.stack(cam_t_cam_list, dim=0)

            for context_dict in context_dict_list:
                ego_t_ego_list = []
                cam_t_cam_list = []
                for cam_idx in range(num_cameras):
                    ego_pose_0 = cur_sample["ego_pose"][cam_idx]
                    ego_pose_n = context_dict["ego_pose"][cam_idx]
                    c2e_extr = cur_sample["c2e_extr"][cam_idx]
                    ego_t_ego, cam_t_cam = self.compute_frame_transforms(
                        ego_pose_0,
                        ego_pose_n,
                        c2e_extr,
                    )
                    ego_t_ego_list.append(torch.from_numpy(ego_t_ego).float())
                    cam_t_cam_list.append(torch.from_numpy(cam_t_cam).float())
                context_dict[("ego_T_ego", 0, self.context_span)] = torch.stack(ego_t_ego_list, dim=0)
                context_dict[("cam_T_cam", 0, self.context_span)] = torch.stack(cam_t_cam_list, dim=0)

            all_context_dict = default_collate([cur_sample] + context_dict_list)
            all_dict = default_collate([cur_sample] + target_dict_list + context_dict_list)

            for key, value in all_dict.items():
                if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                    all_dict[key] = torch.cat([entry for entry in value], dim=0)
            for key, value in all_context_dict.items():
                if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                    all_context_dict[key] = torch.cat([entry for entry in value], dim=0)

            for offset, target_dict in enumerate(target_dict_list, start=1):
                ego_key = ("ego_T_ego", 0, offset)
                cam_key = ("cam_T_cam", 0, offset)
                all_dict[ego_key] = target_dict[ego_key]
                all_dict[cam_key] = target_dict[cam_key]

            ego_key = ("ego_T_ego", 0, self.context_span)
            cam_key = ("cam_T_cam", 0, self.context_span)
            all_context_dict[ego_key] = all_dict[ego_key] = context_dict_list[0][ego_key]
            all_context_dict[cam_key] = all_dict[cam_key] = context_dict_list[0][cam_key]
        else:
            all_context_dict = {}
            all_dict = {}
            for key, value in cur_sample.items():
                if isinstance(value, torch.Tensor):
                    all_dict[key] = value.unsqueeze(0)
                    all_context_dict[key] = value.unsqueeze(0)
                elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                    all_dict[key] = [item.unsqueeze(0) for item in value]
                    all_context_dict[key] = [item.unsqueeze(0) for item in value]
                else:
                    all_dict[key] = value
                    all_context_dict[key] = value

        return {
            "cur_sample": cur_sample,
            "context_frames": all_context_dict,
            "target_frames": target_dict,
            "all_dict": all_dict,
        }
