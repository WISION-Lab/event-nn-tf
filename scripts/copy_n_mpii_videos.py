#!/usr/bin/env python3

import json
import os
import os.path as path
import shutil
from argparse import ArgumentParser


def main(args):
    with open(args.json_file, "r") as f:
        json_data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    for i, item in enumerate(json_data):
        if 0 <= args.n_videos <= i:
            break
        shutil.copy(path.join(args.video_dir, "{}.mp4".format(item["video_id"])), args.out_dir)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("video_dir", help="directory containing video files")
    parser.add_argument("json_file", help="json file containing information about dataset items")
    parser.add_argument("out_dir", help="location where a subset of videos should be copied")

    # Optional arguments
    parser.add_argument(
        "-n",
        "--n-videos",
        default=-1,
        type=int,
        help="the maximum number of videos to copy (negative to copy all videos)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
