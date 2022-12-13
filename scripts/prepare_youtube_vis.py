#!/usr/bin/env python3

import json
import os
import os.path as path
import shutil
from argparse import ArgumentParser

from utils.misc import find_unused_dirname


def main(args):
    tmp_dir = find_unused_dirname("tmp")
    os.makedirs(tmp_dir)

    try:
        for zip_file in args.train_zip, args.val_zip, args.test_zip:
            shutil.unpack_archive(zip_file, tmp_dir)

        with open(args.train_json, "r") as f:
            train_json_data = json.load(f)
        with open(args.val_json, "r") as f:
            val_json_data = json.load(f)
        with open(args.test_json, "r") as f:
            test_json_data = json.load(f)

        train_data, val_data, test_data = {}, {}, {}
        for data_split, json_data in zip(
            [train_data, val_data, test_data],
            [train_json_data, val_json_data, test_json_data],
        ):
            for video in json_data["videos"]:
                item = {
                    "video_id": path.dirname(video["file_names"][0]),
                    "height": video["height"],
                    "width": video["width"],
                }
                if data_split is train_data:
                    labelled_frames = sorted(map(path.basename, video["file_names"]))
                    item.update(
                        {
                            "labelled_frames": labelled_frames,
                            "annotations": [[] for _ in labelled_frames],
                        }
                    )
                data_split[video["id"]] = item
        for annotation in train_json_data["annotations"]:
            category = annotation["category_id"] - 1  # 0-based index
            for t, box in enumerate(annotation["bboxes"]):
                if box is not None:
                    x_1 = box[0]
                    y_1 = box[1]
                    x_2 = box[0] + box[2]
                    y_2 = box[1] + box[3]
                    train_data[annotation["video_id"]]["annotations"][t].append(
                        {
                            "box": [y_1, x_1, y_2, x_2],
                            "category": category,
                        }
                    )

        for data_split, split_name, frame_location in zip(
            [train_data, val_data, test_data],
            ["train", "val", "test"],
            ["train_all_frames", "valid_all_frames", "test_all_frames"],
        ):
            data_split = list(data_split.values())
            frame_dir = path.join(args.out_dir, split_name, "frames")
            os.makedirs(frame_dir)
            for item in data_split:
                shutil.move(
                    path.join(tmp_dir, frame_location, "JPEGImages", item["video_id"]), frame_dir
                )
            with open(path.join(args.out_dir, split_name, "data.json"), "w") as f:
                json.dump(data_split, f, indent=2)

    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(tmp_dir)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("train_zip", help="zip file containing training-set frames")
    parser.add_argument("val_zip", help="zip file containing validation-set frames")
    parser.add_argument("test_zip", help="zip file containing test-set frames")
    parser.add_argument("train_json", help="json file containing training annotations")
    parser.add_argument("val_json", help="json file containing validation metadata")
    parser.add_argument("test_json", help="json file containing test metadata")
    parser.add_argument(
        "out_dir", help="top-level directory where the processed output should be saved"
    )

    # Optional arguments

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
