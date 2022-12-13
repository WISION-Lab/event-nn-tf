#!/usr/bin/env python3

import json
import os
import os.path as path
import shutil
from argparse import ArgumentParser

from utils.misc import find_unused_dirname, listdir_filtered


def main(args):
    # Create a temporary directory where we'll unpack the archive.
    tmp_dir = find_unused_dirname("tmp")
    os.makedirs(tmp_dir)

    try:
        # Unpack the tar archive.
        shutil.unpack_archive(args.data_tar, tmp_dir)

        # Copy frame files into video subdirectories.
        frame_dir = path.join(args.out_dir, "frames")
        for file in listdir_filtered(path.join(tmp_dir, "vid_data", "vid_val"), ".JPEG"):
            video_id, frame_number = file.split(".")[0].split("_")[-2:]
            video_dir = path.join(frame_dir, video_id)
            if not path.isdir(video_dir):
                os.makedirs(video_dir)
            shutil.move(file, path.join(video_dir, "{}.jpg".format(frame_number)))

        with open(path.join(tmp_dir, "vid_data", "annotations", "vid_val.json"), "r") as f:
            raw_data = json.load(f)

        # Process and save annotations.
        video_ids = {}
        frame_numbers = {}
        frame_heights = {}
        frame_widths = {}
        for item in raw_data["images"]:
            video_id, frame_number = item["file_name"].split(".")[0].split("_")[-2:]
            image_id = item["id"]
            video_ids[image_id] = video_id
            frame_numbers[image_id] = frame_number
            frame_heights[image_id] = item["height"]
            frame_widths[image_id] = item["width"]
        data = {}
        for annotation in raw_data["annotations"]:
            image_id = annotation["image_id"]
            video_id = video_ids[image_id]
            frame_number = frame_numbers[image_id]
            if video_id not in data:
                data[video_id] = {}
            if frame_number not in data[video_id]:
                data[video_id][frame_number] = {
                    "annotations": [],
                    "height": frame_heights[image_id],
                    "width": frame_widths[image_id],
                }
            box = annotation["bbox"]
            x_1 = box[0]
            y_1 = box[1]
            x_2 = box[0] + box[2]
            y_2 = box[1] + box[3]
            # noinspection PyTypeChecker
            data[video_id][frame_number]["annotations"].append(
                {"box": [y_1, x_1, y_2, x_2], "category": annotation["category_id"] - 1}
            )
        for video_id, video_dict in data.items():
            data[video_id] = [video_dict[frame_id] for frame_id in sorted(video_dict.keys())]
        with open(path.join(args.out_dir, "val.json"), "w") as f:
            json.dump(data, f, indent=2)

    except KeyboardInterrupt:
        pass
    finally:
        # Clean up temporary directories before exiting.
        shutil.rmtree(tmp_dir)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("data_tar", help="tar.gz file containing the data")
    parser.add_argument(
        "out_dir", help="top-level directory where the processed output should be saved"
    )

    # Optional arguments

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
