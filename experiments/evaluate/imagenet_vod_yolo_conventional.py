#!/usr/bin/env python3

import os
import os.path as path
import pickle
import subprocess
from argparse import ArgumentParser
from shutil import rmtree

import tensorflow as tf

from datasets import imagenet_vod
from datasets.imagenet_vod import IMAGENET_VOD_NAMES
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.yolo import FileDump, yolo_v3, preprocess_image
from utils.misc import print_dict


def main(args):
    # Load the data.
    (frames, boxes, classes, n_annotations), n_items = imagenet_vod.load_data(
        path.join("data", "imagenet_vod"),
        "val",
        size=tuple(args.size),
        preprocess_func=preprocess_image,
    )
    labels = tf.data.Dataset.zip((boxes, classes, n_annotations))
    data = tf.data.Dataset.zip((frames, labels))

    # Uncomment to cap the number of items (for quick tests).
    # n_items = 10

    # Load the model.
    model = yolo_v3(
        tuple(args.size), n_classes=30, h5_weights=path.join("weights", "yolo_v3_imagenet_vod.h5")
    )
    for layer in model.gates:
        layer.policy = Threshold(Constant(), warmup=True)

    # Compute event metric performance.
    # Lower the score threshold here because the computation of mAP already
    # involves systematic variation of the threshold.
    name = "imagenet_vod_yolo_conventional{}".format(args.id)
    eval_path = path.join("outputs", "evaluate_{}".format(name))
    rmtree(eval_path, ignore_errors=True)
    model.evaluate_conventional(
        [FileDump(eval_path, score_threshold=0.1, class_names=IMAGENET_VOD_NAMES, verbose=False)],
        data,
        data_steps=n_items,
        temporal_axis=1,
        verbose=True,
        static_graph=True,
    )
    metric_python = path.join(
        path.expanduser("~"), "miniconda3", "envs", "detection-metrics", "bin", "python"
    )
    print("Conventional metrics:", flush=True)
    os.makedirs(path.join(eval_path, "results"))
    os.chdir("detection")
    subprocess.call(
        [
            metric_python,
            "pascalvoc.py",
            "-gt",
            path.join("..", eval_path, "true"),
            "-det",
            path.join("..", eval_path, "pred"),
            "-sp",
            path.join("..", eval_path, "results"),
            "-np",
            "--threshold",
            "0.5",
        ]
    )
    os.chdir("..")

    ops = model.count_ops_conventional(
        frames, data_steps=1, temporal_axis=1, verbose=True, static_graph=True
    )
    print("Conventional ops:")
    print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
    base_dir = path.join("operations", "evaluate")
    if not path.isdir(base_dir):
        os.makedirs(base_dir)
    with open(path.join(base_dir, "{}.p".format(name)), "wb") as f:
        pickle.dump(ops, f)


def parse_args():
    parser = ArgumentParser()

    # Optional arguments
    parser.add_argument(
        "-i", "--id", default="", type=str, help="an ID to append to the output name"
    )
    parser.add_argument(
        "-s", "--size", nargs=2, default=[240, 320], type=int, help="the size of input images"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
