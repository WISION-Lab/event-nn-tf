#!/usr/bin/env python3

import os
import os.path as path
import pickle
from argparse import ArgumentParser

from datasets import imagenet_vod
from eventnn.policies import Threshold
from eventnn.schedules import Constant, PeriodicReset
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.yolo import apply_bn_gamma, yolo_v3, preprocess_image
from utils.misc import print_dict


def main(args):
    # Load the data.
    (frames, boxes, classes, n_annotations), n_items = imagenet_vod.load_data(
        path.join("data", "imagenet_vod"),
        "val",
        size=tuple(args.size),
        preprocess_func=preprocess_image,
    )

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

    ops = model.count_ops_event(
        frames, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=True
    )
    print("Event ops:")
    print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
    base_dir = path.join("operations", "evaluate", "imagenet_vod_yolo_event")
    if not path.isdir(base_dir):
        os.makedirs(base_dir)
    name = "s{}_m{}_r{}{}".format(args.scale, args.memory_loss, args.reset_period, args.id)
    with open(path.join(base_dir, "{}.p".format(name)), "wb") as f:
        pickle.dump(ops, f)


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
