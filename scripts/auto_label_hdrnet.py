#!/usr/bin/env python3

import os
import os.path as path
from argparse import ArgumentParser

import tensorflow as tf

from models.hdrnet import hdrnet, preprocess_image
from utils.misc import listdir_filtered, read_image, replace_ext, save_image


def main(args):
    size = tuple(args.image_size)
    model = hdrnet(size, npz_weights=args.npz_weights)

    test_dir = path.join(args.dataset_dir, "test", "frames")
    auto_dir = path.join(args.dataset_dir, "auto_{}".format(args.label_name))
    for video_id in os.listdir(test_dir):
        input_video_dir = path.join(test_dir, video_id)
        output_video_dir = path.join(auto_dir, "frames", video_id)
        os.makedirs(output_video_dir, exist_ok=True)
        output_label_dir = path.join(auto_dir, "labels", video_id)
        os.makedirs(output_label_dir, exist_ok=True)
        for frame_filename in listdir_filtered(
            input_video_dir, args.image_extension, join_parent=False
        ):
            frame = read_image(path.join(input_video_dir, frame_filename))
            frame = tf.image.resize_with_pad(frame, size[0], size[1])
            model_input = tf.expand_dims(preprocess_image(frame), axis=0)
            label = model(model_input)[0]

            # Save as .png to prevent compression losses.
            png_filename = replace_ext(frame_filename, ".png")
            save_image(frame, path.join(output_video_dir, png_filename))
            save_image(label, path.join(output_label_dir, png_filename))


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("dataset_dir", help="top-level directory containing the dataset")
    parser.add_argument("npz_weights", help="npz file containing the model weights")

    # Optional arguments
    parser.add_argument(
        "-e", "--image-extension", default=".jpg", help="the filename extension for input images"
    )
    parser.add_argument(
        "-i",
        "--image-size",
        nargs=2,
        default=[540, 960],
        type=int,
        help="the size at which the model should process images",
    )
    parser.add_argument(
        "-l",
        "--label-name",
        default="hdrnet",
        help="what to name the directory where labels are stored",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
