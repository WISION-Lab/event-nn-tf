#!/usr/bin/env python3

import json
import os
import os.path as path
from argparse import ArgumentParser

import tensorflow as tf

from models.yolo import yolo_v3, postprocess_image, preprocess_image
from utils.misc import listdir_filtered, read_image, save_image


def main(args):
    size = tuple(args.image_size)
    model = yolo_v3(size, darknet_weights=args.darknet_weights)
    test_dir = path.join(args.dataset_dir, "test")
    with open(path.join(test_dir, "data.json"), "r") as f:
        json_data = json.load(f)

    auto_dir = path.join(args.dataset_dir, "auto_{}".format(args.label_name))
    os.makedirs(auto_dir, exist_ok=True)
    auto_labelled = []
    for item in json_data:
        video_id = item["video_id"]
        input_video_dir = path.join(test_dir, "frames", video_id)
        output_video_dir = path.join(auto_dir, "frames", video_id)
        os.makedirs(output_video_dir, exist_ok=True)
        annotations = []
        for frame_filename in listdir_filtered(
            input_video_dir, args.image_extension, join_parent=False
        ):
            frame = read_image(path.join(input_video_dir, frame_filename))
            frame = tf.image.resize_with_pad(frame, size[0], size[1])

            # Save and re-load so that this input incorporates any JPG
            # compression losses.
            frame_path = path.join(output_video_dir, frame_filename)
            save_image(frame, frame_path)
            frame = read_image(frame_path)

            model_input = tf.expand_dims(preprocess_image(frame), axis=0)
            boxes, classes = model(model_input)
            boxes, _, classes = postprocess_image(boxes[0], classes[0])
            annotations_t = []
            for box, category in zip(boxes, classes):
                annotations_t.append(
                    {
                        "box": [float(coordinate) for coordinate in box],
                        "category": int(category),
                    }
                )
            annotations.append(annotations_t)
        auto_labelled.append(
            {
                "annotations": annotations,
                "video_id": item["video_id"],
                "height": size[0],
                "width": size[1],
            }
        )
    with open(path.join(args.dataset_dir, "auto", "data.json"), "w") as f:
        json.dump(auto_labelled, f, indent=2)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("dataset_dir", help="top-level directory containing the dataset")
    parser.add_argument("darknet_weights", help=".weights file containing the model weights")

    # Optional arguments
    parser.add_argument(
        "-e", "--image-extension", default=".jpg", help="the filename extension for input images"
    )
    parser.add_argument(
        "-i",
        "--image-size",
        nargs=2,
        default=[288, 512],
        type=int,
        help="the size at which the model should process images",
    )
    parser.add_argument(
        "-l",
        "--label-name",
        default="yolo",
        help="what to name the directory where labels are stored",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
