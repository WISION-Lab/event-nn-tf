#!/usr/bin/env python3

import os
import os.path as path
import pickle
from argparse import ArgumentParser

import tensorflow as tf

from datasets import jhmdb
from eventnn.policies import Threshold
from eventnn.schedules import Constant
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
        layer.policy = Threshold(Constant(), warmup=True)

    results = model.evaluate_conventional(
        [PCKSinglePerson(postprocess_func=postprocess_pck_jhmdb_mpii)],
        data,
        data_steps=n_items,
        temporal_axis=1,
        verbose=True,
        static_graph=True,
    )
    print("Conventional metrics:")
    print_dict(results)

    ops = model.count_ops_conventional(
        frames, data_steps=1, temporal_axis=1, verbose=True, static_graph=True
    )
    print("Conventional ops:")
    print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
    base_dir = path.join("operations", "evaluate", "jhmdb_openpose_conventional")
    if not path.isdir(base_dir):
        os.makedirs(base_dir)
    with open(path.join(base_dir, "{}.p".format(args.id)), "wb") as f:
        pickle.dump(ops, f)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "id", type=str, help="an ID to assign to the output name"
    )

    # Optional arguments
    parser.add_argument(
        "-f", "--factor", default=0, type=int, help="the truncation factor to use"
    )
    parser.add_argument(
        "-s", "--size", nargs=2, default=[240, 320], type=int, help="the size of input images"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
