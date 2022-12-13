#!/usr/bin/env python3

import os
import os.path as path
import pickle
from argparse import ArgumentParser

import tensorflow as tf

from datasets import jhmdb
from eventnn.policies import Threshold
from eventnn.schedules import Constant, PeriodicReset
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from metrics.pose import PCKSinglePerson
from models.openpose import openpose_mpii, postprocess_pck_jhmdb_mpii, preprocess_image
from utils.misc import print_dict


def main(args):
    # Load the data.
    (frames, joints), n_items = jhmdb.load_data(
        path.join("data", "jhmdb"), "test", size=tuple(args.size), preprocess_func=preprocess_image
    )
    if args.factor != 0:
        frames = frames.map(lambda x: x[:, : args.factor * (tf.shape(x)[1] // args.factor) + 1])
        joints = joints.map(lambda x: x.to_tensor())
        joints = joints.map(lambda x: x[:, : args.factor * (tf.shape(x)[1] // args.factor) + 1])
    labels = tf.data.Dataset.zip((frames, joints))
    data = tf.data.Dataset.zip((frames, labels))

    # Uncomment to cap the number of items (for quick tests).
    # n_items = 10

    # Load the model.
    model = openpose_mpii(tuple(args.size), npz_weights=path.join("weights", "openpose_mpii.npz"))
    for layer in model.gates:
        if args.reset_period > 0:
            schedule = PeriodicReset()
            schedule.period.assign(args.reset_period)
        else:
            schedule = Constant()
        schedule.scale.assign(args.scale)
        layer.policy = Threshold(schedule, warmup=True)
        layer.memory_loss = args.memory_loss

    # Compute event metric performance.
    results = model.evaluate_event(
        [PCKSinglePerson(postprocess_func=postprocess_pck_jhmdb_mpii)],
        data,
        data_steps=n_items,
        temporal_axis=1,
        verbose=True,
        static_graph=True,
    )
    print("Event metrics:")
    print_dict(results)

    ops = model.count_ops_event(
        frames, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=True
    )
    print("Event ops:")
    print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
    base_dir = path.join("operations", "evaluate", "jhmdb_openpose_event")
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
        "-f", "--factor", default=0, type=int, help="the truncation factor to use"
    )
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
        "-s", "--size", nargs=2, default=[240, 320], type=int, help="the size of input images"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
