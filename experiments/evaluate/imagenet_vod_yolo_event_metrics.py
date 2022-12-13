#!/usr/bin/env python3

import os
import os.path as path
import subprocess
from argparse import ArgumentParser
from shutil import rmtree

import tensorflow as tf

from datasets import imagenet_vod
from datasets.imagenet_vod import IMAGENET_VOD_NAMES
from eventnn.policies import Threshold
from eventnn.schedules import Constant, PeriodicReset
from models.yolo import FileDump, apply_bn_gamma, yolo_v3, preprocess_image


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
        if args.reset_period > 0:
            schedule = PeriodicReset()
            schedule.period.assign(args.reset_period)
        else:
            schedule = Constant()
        schedule.scale.assign(args.scale)
        layer.policy = Threshold(schedule, warmup=True)
        layer.memory_loss = args.memory_loss
    apply_bn_gamma(model)

    # Compute event metric performance.
    # Lower the score threshold here because the computation of mAP already
    # involves systematic variation of the threshold.
    name = "s{}_m{}_r{}{}".format(args.scale, args.memory_loss, args.reset_period, args.id)
    eval_path = path.join("outputs", "evaluate_imagenet_vod_yolo_event_{}".format(name))
    rmtree(eval_path, ignore_errors=True)
    model.evaluate_event(
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
    print("Event metrics:", flush=True)
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


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("scale", type=float, help="the threshold scale to apply")

    # Optional arguments
    parser.add_argument(
        "-i", "--id", default="", type=str, help="an ID to append to the output name"
    )
    parser.add_argument(
        "-m",
        "--memory-loss",
        action="store_true",
        help="enable memory loss (e.g., skip-convolution)",
    )
    parser.add_argument(
        "-r",
        "--reset-period",
        default=0,
        type=int,
        help="the frequency with which the model should be re-flushed (0 for no re-flushing)",
    )
    parser.add_argument(
        "-s", "--size", nargs=2, default=[224, 384], type=int, help="the size of input images"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
