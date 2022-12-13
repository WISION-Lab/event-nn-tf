import json
import os.path as path

from utils.misc import (
    compute_padded_coordinates,
    listdir_filtered,
    prepare_padded_video_data,
    prepare_ragged_data,
)

JOINT_NAMES = [
    "neck",
    "belly",
    "face",
    "right shoulder",
    "left shoulder",
    "right hip",
    "left hip",
    "right elbow",
    "left elbow",
    "right knee",
    "left knee",
    "right wrist",
    "left wrist",
    "right ankle",
    "left ankle",
]

# JOINT_NAMES[i] corresponds to MPII_REMAP_NAMES[i].
MPII_REMAP_NAMES = [
    "upper neck",
    "pelvis",
    "head top",
    "right shoulder",
    "left shoulder",
    "right hip",
    "left hip",
    "right elbow",
    "left elbow",
    "right knee",
    "left knee",
    "right wrist",
    "left wrist",
    "right ankle",
    "left ankle",
]

# JHMDB joint i corresponds to MPII joint MPII_REMAP[i].
# [datasets.mpii.JOINT_NAMES.index(name) for name in MPII_REMAP_NAMES]
MPII_REMAP_INDICES = [8, 6, 9, 12, 13, 2, 3, 11, 14, 1, 4, 10, 15, 0, 5]


def load_data(data_dir, split, size=(240, 320), preprocess_func=None):
    if split in ("test", "train"):
        (frames_1, joints_1), n_items_1 = load_data(
            data_dir, "subset_{}_1".format(split), size=size, preprocess_func=preprocess_func
        )
        (frames_2, joints_2), n_items_2 = load_data(
            data_dir, "subset_{}_2".format(split), size=size, preprocess_func=preprocess_func
        )
        (frames_3, joints_3), n_items_3 = load_data(
            data_dir, "subset_{}_3".format(split), size=size, preprocess_func=preprocess_func
        )
        frames = frames_1.concatenate(frames_2).concatenate(frames_3)
        joints = joints_1.concatenate(joints_2).concatenate(joints_3)
        n_items = n_items_1 + n_items_2 + n_items_3
        return (frames, joints), n_items

    with open(path.join(data_dir, "{}.txt".format(split)), "r") as f:
        items = [line.strip() for line in f.readlines()]
    with open(path.join(data_dir, "labels.json"), "r") as f:
        json_data = json.load(f)

    frames = []
    joints = []
    for video_id in items:
        frames_i = listdir_filtered(path.join(data_dir, "frames", video_id), ".png")
        frames.append(frames_i)

        json_item = json_data[video_id]
        size_before = json_item["height"], json_item["width"]
        joints_i = [
            [list(compute_padded_coordinates(*joint, size_before, size)) for joint in joints_t]
            for joints_t in json_item["joints"]
        ]
        joints.append(joints_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_func)

    # Dataset shape (1, n_frames, 15, 2)
    joints = prepare_ragged_data(joints, inner_shape=(15, 2))

    return (frames, joints), n_items
