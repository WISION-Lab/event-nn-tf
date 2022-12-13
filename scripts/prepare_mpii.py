#!/usr/bin/env python3

import json
import os
import os.path as path
import shutil
from argparse import ArgumentParser

from scipy.io import loadmat

from models.openpose import MPII_JOINT_ORDER
from utils.misc import find_unused_dirname, read_image


def main(args):
    tmp_frames_dir = find_unused_dirname("tmp_frames")
    os.makedirs(tmp_frames_dir)
    tmp_combined_dir = find_unused_dirname("tmp_combined")
    os.makedirs(tmp_combined_dir)
    tmp_annotations_dir = find_unused_dirname("tmp_annotations")
    os.makedirs(tmp_annotations_dir)

    try:
        for batch_file in args.batch_files:
            shutil.unpack_archive(batch_file, tmp_frames_dir)
        for batch_dir in os.listdir(tmp_frames_dir):
            batch_dir = path.join(tmp_frames_dir, batch_dir)
            for video_dir in os.listdir(batch_dir):
                video_dir = path.join(batch_dir, video_dir)
                shutil.move(video_dir, tmp_combined_dir)
        all_video_ids = os.listdir(tmp_combined_dir)

        shutil.unpack_archive(args.annotations_zip, tmp_annotations_dir)
        annotations = loadmat(
            path.join(
                tmp_annotations_dir,
                path.splitext(path.basename(args.annotations_zip))[0],
                "mpii_human_pose_v1_u12_1.mat",
            )
        )
        annotations = annotations["RELEASE"][0, 0]
        keyframes = loadmat(args.keyframes_mat)

        n_items = annotations["annolist"].shape[1]
        train = []
        test = []
        for i in range(n_items):
            is_training_item = bool(annotations["img_train"][0, i])
            video_id, frame = path.split(
                keyframes["annolist_keyframes"][0, i]["image"]["name"][0, 0][0]
            )
            if video_id not in all_video_ids:
                continue
            if frame not in os.listdir(path.join(tmp_combined_dir, video_id)):
                continue
            size = read_image(path.join(tmp_combined_dir, video_id, frame)).shape

            if not is_training_item:
                test.append(
                    {
                        "video_id": video_id,
                        "frame": frame,
                        "height": size[0],
                        "width": size[1],
                    }
                )

            else:
                if annotations["annolist"][0, i]["annorect"].shape[0] == 0:
                    annotations_i = None
                    n_people = 0
                else:
                    annotations_i = annotations["annolist"][0, i]["annorect"][0]
                    n_people = annotations_i.shape[0]
                people = []
                for j in range(n_people):
                    if "annopoints" not in annotations_i[j].dtype.names:
                        continue
                    if annotations_i[j]["annopoints"].shape[0] == 0:
                        continue
                    annotation_i_j = annotations_i[j]["annopoints"][0, 0]["point"][0]
                    joints = [[float("nan"), float("nan")]] * args.n_joint_types
                    for k in range(annotation_i_j.shape[0]):
                        joint_id = int(annotation_i_j["id"][k])
                        joints[joint_id] = [
                            float(annotation_i_j["x"][k]),
                            float(annotation_i_j["y"][k]),
                        ]
                    people.append(
                        {
                            "joints": [joints[i] for i in MPII_JOINT_ORDER],
                            "head": [float(annotations_i[j][c]) for c in ["x1", "x2", "y1", "y2"]],
                        }
                    )
                train.append(
                    {
                        "people": people,
                        "video_id": video_id,
                        "frame": frame,
                        "height": size[0],
                        "width": size[1],
                    }
                )

        for data_split, split_name in (train, "train"), (test, "test"):
            frame_dir = path.join(args.out_dir, split_name, "frames")
            os.makedirs(frame_dir)
            for item in data_split:
                shutil.move(path.join(tmp_combined_dir, item["video_id"]), frame_dir)
            with open(path.join(args.out_dir, split_name, "data.json"), "w") as f:
                json.dump(data_split, f, indent=2)

    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(tmp_frames_dir)
        shutil.rmtree(tmp_combined_dir)
        shutil.rmtree(tmp_annotations_dir)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("annotations_zip", help="zip file containing annotations")
    parser.add_argument("keyframes_mat", help="mat file containing image-video mappings")
    parser.add_argument("batch_files", nargs="+", help="batch files containing videos")
    parser.add_argument(
        "out_dir", help="top-level directory where the processed output should be saved"
    )

    # Optional arguments
    parser.add_argument(
        "-n", "--n-joint-types", default=16, type=int, help="the number of distinct joint types"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
