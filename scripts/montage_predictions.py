#!/usr/bin/env python3

import os
import os.path as path
import subprocess
from argparse import ArgumentParser


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for item in os.listdir(args.prediction_dir):
        if not path.isdir(path.join(args.prediction_dir, item)):
            continue
        inputs = [
            path.join(args.prediction_dir, item, "{}{}".format(i, args.input_ext))
            for i in args.indices
        ]
        subprocess.call(
            [
                "magick",
                "montage",
                "-background",
                args.background,
                "-tile",
                "{}x".format(len(args.indices)),
                "-geometry",
                "+{}+{}".format(args.spacing, args.spacing),
            ]
            + inputs
            + [path.join(args.output_dir, "{}{}".format(item, args.output_ext))]
        )


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("prediction_dir", help="directory containing prediction image folders")
    parser.add_argument("output_dir", help="directory in which montages should be saved")
    parser.add_argument(
        "indices", nargs="+", type=int, help="the frame indices to include in the montage"
    )

    # Optional arguments
    parser.add_argument("-b", "--background", default="#FFFFFF", help="the background color")
    parser.add_argument("-i", "--input-ext", default=".jpg", help="the extension for input images")
    parser.add_argument(
        "-o", "--output-ext", default=".jpg", help="the extension for the output image"
    )
    parser.add_argument("-s", "--spacing", default=10, type=int, help="the spacing between items")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
