#!/usr/bin/env python3

import os
import os.path as path
from argparse import ArgumentParser

import tensorflow as tf

from models.pwcnet import pwcnet, postprocess_image, preprocess_image
from utils.misc import listdir_filtered, read_image, replace_ext, resize_image, save_image


def main(args):
    size = tuple(args.image_size)
    model = pwcnet(size, npz_weights=args.npz_weights)

    test_dir = path.join(args.dataset_dir, "test", "frames")
    auto_dir = path.join(args.dataset_dir, "auto_{}".format(args.label_name))
    for video_id in os.listdir(test_dir):
        input_video_dir = path.join(test_dir, video_id)
        output_video_dir = path.join(auto_dir, "frames", video_id)
        os.makedirs(output_video_dir, exist_ok=True)
        output_flow_dir = path.join(auto_dir, "flow", video_id)
        os.makedirs(output_flow_dir, exist_ok=True)
        frame_filenames = listdir_filtered(
            input_video_dir, args.image_extension, join_parent=False
        )
        filename_pairs = list(zip(frame_filenames[:-1], frame_filenames[1:]))
        for i, filename_pair in enumerate(filename_pairs):
            frame_0 = read_image(path.join(input_video_dir, filename_pair[0]))
            frame_1 = read_image(path.join(input_video_dir, filename_pair[1]))
            frame_0 = resize_image(frame_0, size)
            frame_1 = resize_image(frame_1, size)
            save_image(frame_0, path.join(output_video_dir, replace_ext(filename_pair[0], ".png")))
            save_image(frame_1, path.join(output_video_dir, replace_ext(filename_pair[1], ".png")))
            model_input = tf.expand_dims(
                tf.stack([preprocess_image(frame_0), preprocess_image(frame_1)]), axis=0
            )
            flow = model(model_input)[0]
            flow = postprocess_image(flow, frame_0.shape[0:2])
            tf.io.write_file(
                path.join(output_flow_dir, replace_ext(filename_pair[0], ".proto")),
                tf.io.serialize_tensor(flow),
            )


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("dataset_dir", help="top-level directory containing the dataset")
    parser.add_argument("npz_weights", help="npz file containing the model weights")

    # Optional arguments
    parser.add_argument(
        "-e", "--image-extension", default=".png", help="the filename extension for input images"
    )
    parser.add_argument(
        "-i",
        "--image-size",
        nargs=2,
        default=[320, 512],
        type=int,
        help="the size at which the model should process images",
    )
    parser.add_argument(
        "-l",
        "--label-name",
        default="pwcnet",
        help="what to name the directory where labels are stored",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
