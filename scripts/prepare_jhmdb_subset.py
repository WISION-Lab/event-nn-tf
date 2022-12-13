#!/usr/bin/env python3

import json
import os
import os.path as path
import shutil
from argparse import ArgumentParser

import numpy as np
from scipy.io import loadmat

from utils.misc import listdir, listdir_filtered, find_unused_dirname, read_image


def main(args):
    # Create temporary directories where we'll unpack the archives.
    tmp_frames_dir = find_unused_dirname("tmp_frames")
    os.makedirs(tmp_frames_dir)
    tmp_subset_splits_dir = find_unused_dirname("tmp_subset_splits")
    os.makedirs(tmp_subset_splits_dir)
    tmp_joint_positions_dir = find_unused_dirname("tmp_joint_positions")
    os.makedirs(tmp_joint_positions_dir)

    try:
        # Unpack the zip and tar.gz archives.
        shutil.unpack_archive(args.images_tar_gz, tmp_frames_dir)
        shutil.unpack_archive(args.subset_splits_zip, tmp_subset_splits_dir)
        shutil.unpack_archive(args.joint_positions_zip, tmp_joint_positions_dir)

        # Copy video folders and rename them to have integer IDs.
        frame_dir = path.join(args.out_dir, "frames")
        os.makedirs(frame_dir)
        i = 0
        id_dict = {}
        for category_dir in listdir(
            path.join(tmp_frames_dir, "Rename_Images"), include_hidden=False
        ):
            for video_dir in listdir(category_dir, include_hidden=False):
                basename = path.basename(video_dir)
                video_id = "{:05d}".format(i)
                id_dict[basename] = video_id
                frame_dir_i = path.join(frame_dir, video_id)
                os.makedirs(frame_dir_i)
                for image in listdir_filtered(video_dir, ".png"):
                    shutil.move(image, frame_dir_i)
                i += 1

        # Write the elements of each split to a text file.
        for split in "1", "2", "3":
            train, test = [], []
            for filename in listdir_filtered(tmp_subset_splits_dir, ".txt"):
                if not filename.endswith("{}.txt".format(split)):
                    continue
                with open(filename, "r") as f:
                    for line in f:
                        tokens = line.strip().split(" ")
                        video_id = id_dict[path.splitext(" ".join(tokens[:-1]))[0]]
                        if tokens[-1] == "1":
                            train.append(video_id)
                        else:
                            test.append(video_id)
                for subset, name in (train, "subset_train"), (test, "subset_test"):
                    with open(path.join(args.out_dir, "{}_{}.txt".format(name, split)), "w") as f:
                        f.writelines(os.linesep.join(sorted(subset)))

        # Extract annotations and write them to a json file.
        data = {}
        for category_dir in listdir(
            path.join(tmp_joint_positions_dir, "joint_positions"), include_hidden=False
        ):
            for video_dir in listdir(category_dir, include_hidden=False):
                annotations = loadmat(path.join(video_dir, "joint_positions.mat"))["pos_img"]
                joints = [
                    [[float(joint[0]), float(joint[1])] for joint in frame]
                    for frame in np.transpose(annotations)
                ]
                video_id = id_dict[path.basename(video_dir)]
                frames = listdir_filtered(path.join(frame_dir, video_id), ".png")
                height, width = read_image(frames[0]).shape[0:2]
                data[video_id] = {"joints": joints, "height": height, "width": width}
        with open(path.join(args.out_dir, "labels.json"), "w") as f:
            json.dump(data, f, indent=2)

    except KeyboardInterrupt:
        pass
    finally:
        # Clean up temporary directories before exiting.
        shutil.rmtree(tmp_frames_dir)
        shutil.rmtree(tmp_subset_splits_dir)
        shutil.rmtree(tmp_joint_positions_dir)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("images_tar_gz", help="tar.gz file containing images")
    parser.add_argument("subset_splits_zip", help="zip file containing splits")
    parser.add_argument(
        "joint_positions_zip", help="zip file containing ground-truth joint positions"
    )
    parser.add_argument(
        "out_dir", help="top-level directory where the processed output should be saved"
    )

    # Optional arguments

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
