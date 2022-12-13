#!/usr/bin/env python3

import json
import os.path as path
from argparse import ArgumentParser

from utils.misc import listdir_filtered


def main(args):
    base_dir = path.join(args.dataset_dir, args.split)
    with open(path.join(base_dir, "data.json"), "r") as f:
        json_data = json.load(f)

    for item in json_data:
        video_id = item["video_id"]
        n_frames = len(listdir_filtered(path.join(base_dir, "frames", video_id), ".jpg"))
        min_satisfied = n_frames >= args.min_frames
        max_satisfied = (args.max_frames < 0) or (n_frames <= args.max_frames)
        if min_satisfied and max_satisfied:
            print(video_id)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("dataset_dir", help="top-level directory containing the dataset")
    parser.add_argument("split", help="the name of the split to filter")

    # Optional arguments
    parser.add_argument(
        "-m",
        "--min-frames",
        default=0,
        type=int,
        help="the minimum video length to consider (inclusive)",
    )
    parser.add_argument(
        "-M",
        "--max-frames",
        default=-1,
        type=int,
        help="the minimum video length to consider (inclusive); negative for no maximum",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
