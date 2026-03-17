#!/usr/bin/env python3

import argparse
import bisect
import json
import os
from pathlib import Path

import numpy as np


def quaternion_to_rotation_matrix_xyzw(x, y, z, w):
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(translation, rotation_xyzw):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation_matrix_xyzw(*rotation_xyzw)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def nearest_index_after(stamps, target, min_index):
    start = min_index + 1
    if start >= len(stamps):
        return None

    pos = bisect.bisect_left(stamps, target, lo=start)
    candidates = []
    if pos < len(stamps):
        candidates.append(pos)
    if pos - 1 >= start:
        candidates.append(pos - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda idx: abs(stamps[idx] - target))


def maybe_stop_early(camera_images, camera_infos, pose_entries, needed_count, static_ready):
    if len(camera_infos) != len(camera_images):
        return False
    if len(pose_entries) < needed_count:
        return False
    if not static_ready:
        return False
    return all(len(entries) >= needed_count for entries in camera_images.values())


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a small ReconDrive dataset from an MCAP bag")
    parser.add_argument("--mcap", required=True, help="Path to the input MCAP file")
    parser.add_argument("--output", required=True, help="Output directory for the preprocessed dataset")
    parser.add_argument(
        "--cameras",
        default="camera0,camera1,camera2,camera3,camera4,camera5",
        help="Comma-separated camera names to extract",
    )
    parser.add_argument(
        "--max_sync_frames",
        type=int,
        default=12,
        help="Maximum number of synchronized frames to keep in the prepared dataset",
    )
    parser.add_argument(
        "--context_span",
        type=int,
        default=6,
        help="Temporal context span expected by ReconDrive",
    )
    parser.add_argument(
        "--sync_tolerance_ms",
        type=float,
        default=40.0,
        help="Maximum timestamp mismatch allowed when synchronizing sensors",
    )
    parser.add_argument(
        "--scene_name",
        default=None,
        help="Optional scene name override",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python APIs are required. Run this script with the ROS environment sourced."
        ) from exc

    bag_path = Path(args.mcap).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cameras = [cam.strip() for cam in args.cameras.split(",") if cam.strip()]
    if not cameras:
        raise ValueError("At least one camera must be provided")

    images_root = output_root / "images"
    for camera in cameras:
        (images_root / camera).mkdir(parents=True, exist_ok=True)

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="mcap"),
        ConverterOptions("", ""),
    )

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    wanted_camera_info_topics = {
        f"/sensing/camera/{camera}/camera_info": camera for camera in cameras
    }
    wanted_image_topics = {
        f"/sensing/camera/{camera}/image_raw/compressed": camera for camera in cameras
    }

    camera_info_type_cache = {}
    image_type_cache = {}
    tf_type = get_message(topic_types["/tf"])
    tf_static_type = get_message(topic_types["/tf_static"])

    static_transforms = {}
    camera_infos = {}
    camera_images = {camera: [] for camera in cameras}
    pose_entries = []

    needed_count = args.max_sync_frames + args.context_span + 2
    tolerance_ns = int(args.sync_tolerance_ms * 1_000_000)

    required_pairs = [("base_link", "sensor_kit_base_link")]
    for camera in cameras:
        required_pairs.append(("sensor_kit_base_link", f"{camera}/camera_link"))
        required_pairs.append((f"{camera}/camera_link", f"{camera}/camera_optical_link"))

    while reader.has_next():
        topic, data, bag_timestamp = reader.read_next()

        if topic in wanted_camera_info_topics:
            camera = wanted_camera_info_topics[topic]
            if camera in camera_infos:
                continue
            msg_type = camera_info_type_cache.setdefault(topic, get_message(topic_types[topic]))
            msg = deserialize_message(data, msg_type)
            camera_infos[camera] = {
                "frame_id": msg.header.frame_id,
                "width": int(msg.width),
                "height": int(msg.height),
                "K": np.array(msg.k, dtype=np.float64).reshape(3, 3),
            }
            continue

        if topic in wanted_image_topics:
            camera = wanted_image_topics[topic]
            msg_type = image_type_cache.setdefault(topic, get_message(topic_types[topic]))
            msg = deserialize_message(data, msg_type)
            timestamp_ns = stamp_to_ns(msg.header.stamp) if msg.header.stamp.sec or msg.header.stamp.nanosec else int(bag_timestamp)
            image_format = getattr(msg, "format", "") or ""
            extension = ".png" if "png" in image_format.lower() else ".jpg"
            relative_path = Path("images") / camera / f"{timestamp_ns}{extension}"
            absolute_path = output_root / relative_path
            if not absolute_path.exists():
                with open(absolute_path, "wb") as handle:
                    handle.write(bytes(msg.data))
            camera_images[camera].append(
                {
                    "timestamp": timestamp_ns,
                    "path": str(relative_path),
                }
            )
            static_ready = all(pair in static_transforms for pair in required_pairs)
            if maybe_stop_early(camera_images, camera_infos, pose_entries, needed_count, static_ready):
                break
            continue

        if topic == "/tf_static":
            msg = deserialize_message(data, tf_static_type)
            for transform in msg.transforms:
                translation = (
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                )
                rotation = (
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                )
                static_transforms[(transform.header.frame_id, transform.child_frame_id)] = transform_to_matrix(
                    translation,
                    rotation,
                )
            continue

        if topic == "/tf":
            msg = deserialize_message(data, tf_type)
            for transform in msg.transforms:
                if transform.header.frame_id != "map" or transform.child_frame_id != "base_link":
                    continue
                translation = (
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                )
                rotation = (
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                )
                pose_entries.append(
                    {
                        "timestamp": stamp_to_ns(transform.header.stamp),
                        "ego_pose": transform_to_matrix(translation, rotation),
                    }
                )
            static_ready = all(pair in static_transforms for pair in required_pairs)
            if maybe_stop_early(camera_images, camera_infos, pose_entries, needed_count, static_ready):
                break

    missing_infos = [camera for camera in cameras if camera not in camera_infos]
    if missing_infos:
        raise RuntimeError(f"Missing camera_info topics for: {missing_infos}")

    if not pose_entries:
        raise RuntimeError("No map->base_link poses were found in /tf")

    missing_pairs = [pair for pair in required_pairs if pair not in static_transforms]
    if missing_pairs:
        raise RuntimeError(f"Missing static TF transforms: {missing_pairs}")

    camera_extrinsics = {}
    base_to_sensor_kit = static_transforms[("base_link", "sensor_kit_base_link")]
    for camera in cameras:
        sensor_kit_to_camera = static_transforms[("sensor_kit_base_link", f"{camera}/camera_link")]
        camera_to_optical = static_transforms[(f"{camera}/camera_link", f"{camera}/camera_optical_link")]
        camera_extrinsics[camera] = base_to_sensor_kit @ sensor_kit_to_camera @ camera_to_optical

    for camera in cameras:
        camera_images[camera].sort(key=lambda entry: entry["timestamp"])
    pose_entries.sort(key=lambda entry: entry["timestamp"])

    pose_timestamps = [entry["timestamp"] for entry in pose_entries]
    camera_timestamps = {
        camera: [entry["timestamp"] for entry in entries]
        for camera, entries in camera_images.items()
    }

    synchronized_frames = []
    last_camera_indices = {camera: -1 for camera in cameras}
    last_pose_index = -1

    for anchor_entry in camera_images[cameras[0]]:
        anchor_timestamp = anchor_entry["timestamp"]
        frame_cameras = {}
        valid = True

        for camera in cameras:
            index = nearest_index_after(camera_timestamps[camera], anchor_timestamp, last_camera_indices[camera])
            if index is None:
                valid = False
                break
            image_entry = camera_images[camera][index]
            if abs(image_entry["timestamp"] - anchor_timestamp) > tolerance_ns:
                valid = False
                break
            frame_cameras[camera] = {
                "image_timestamp": image_entry["timestamp"],
                "image_path": image_entry["path"],
                "K": camera_infos[camera]["K"].tolist(),
                "c2e_extr": camera_extrinsics[camera].tolist(),
                "frame_id": camera_infos[camera]["frame_id"],
                "width": camera_infos[camera]["width"],
                "height": camera_infos[camera]["height"],
            }
            last_camera_indices[camera] = index

        if not valid:
            continue

        pose_index = nearest_index_after(pose_timestamps, anchor_timestamp, last_pose_index)
        if pose_index is None:
            break
        pose_entry = pose_entries[pose_index]
        if abs(pose_entry["timestamp"] - anchor_timestamp) > tolerance_ns:
            continue

        last_pose_index = pose_index
        synchronized_frames.append(
            {
                "timestamp": anchor_timestamp,
                "ego_pose": pose_entry["ego_pose"].tolist(),
                "cameras": frame_cameras,
            }
        )

        if len(synchronized_frames) >= args.max_sync_frames:
            break

    if len(synchronized_frames) < args.context_span + 2:
        raise RuntimeError(
            f"Only {len(synchronized_frames)} synchronized frames were produced; "
            f"need at least {args.context_span + 2}"
        )

    scene_name = args.scene_name or bag_path.stem
    metadata = {
        "dataset_type": "mcap_preprocessed",
        "source_mcap": str(bag_path),
        "cameras": cameras,
        "context_span": args.context_span,
        "sync_tolerance_ms": args.sync_tolerance_ms,
        "scenes": [
            {
                "scene_name": scene_name,
                "scene_token": scene_name,
                "frames": synchronized_frames,
            }
        ],
    }

    metadata_path = output_root / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote metadata: {metadata_path}")
    print(f"Cameras: {cameras}")
    print(f"Synchronized frames: {len(synchronized_frames)}")
    print(f"Valid temporal windows: {len(synchronized_frames) - args.context_span - 1}")


if __name__ == "__main__":
    main()
