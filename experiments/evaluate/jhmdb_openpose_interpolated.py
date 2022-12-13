#!/usr/bin/env python3

import os
import os.path as path
import pickle
from argparse import ArgumentParser

import numpy as np
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
        path.join("data", "jhmdb"),
        "test",
        size=tuple(args.size),
        preprocess_func=preprocess_image,
    )
    labels = tf.data.Dataset.zip((frames, joints))
    frames = frames.map(lambda x: x[:, :: args.factor])

    # Uncomment to cap the number of items (for quick tests).
    # n_items = 10

    # Load the model.
    model = openpose_mpii(tuple(args.size), npz_weights=path.join("weights", "openpose_mpii.npz"))
    for layer in model.gates:
        layer.policy = Threshold(Constant(), warmup=True)

    outputs = model.predict_conventional(
        frames,
        data_steps=n_items,
        temporal_axis=1,
        verbose=True,
        static_graph=True,
    )

    metric = PCKSinglePerson()
    for y_true, y_pred in zip(labels, zip(*outputs)):

        # Interpolate between predicted frames.
        tau = y_pred[0].shape[0]
        person_interp = np.full(((tau - 1) * args.factor + 1,) + y_true[1].shape[-2:], -1.0)
        y_true = (np.array(y_true[0]), y_true[1].numpy())
        y_true_sliced = (y_true[0][:, :: args.factor], y_true[1][:, :: args.factor])
        for t in range(tau - 1):
            y_pred_1 = (y_pred[0][np.newaxis, t], y_pred[1][np.newaxis, t])
            y_pred_2 = (y_pred[0][np.newaxis, t + 1], y_pred[1][np.newaxis, t + 1])
            y_true_1 = (y_true_sliced[0][:, t], y_true_sliced[1][:, t])
            y_true_2 = (y_true_sliced[0][:, t + 1], y_true_sliced[1][:, t + 1])
            person_pred_1 = postprocess_pck_jhmdb_mpii(y_true_1, y_pred_1)[1][0]
            person_pred_2 = postprocess_pck_jhmdb_mpii(y_true_2, y_pred_2)[1][0]
            for i in range(person_pred_1.shape[0]):
                for j in range(person_pred_1.shape[1]):
                    value_1 = person_pred_1[i, j]
                    value_2 = person_pred_2[i, j]
                    if value_1 != -1.0 and value_2 != -1.0:
                        t_1 = t * args.factor
                        t_2 = (t + 1) * args.factor + 1
                        person_interp[t_1:t_2, i, j] = np.linspace(
                            value_1, value_2, num=args.factor + 1
                        )

        # Update the PCK metric.
        person_true = y_true[1][0, : person_interp.shape[0]]
        metric.update_state(person_true, person_interp)
    print("Conventional metrics:")
    print_dict({metric.name: metric.result()})

    ops = model.count_ops_conventional(
        frames, data_steps=1, temporal_axis=1, verbose=True, static_graph=True
    )
    n_model = 0
    n_total = 0
    for video in frames:
        n_model += video.shape[1]
        n_total += (video.shape[1] - 1) * args.factor + 1
    scale = n_model / n_total
    print("Conventional ops:")
    ops_reduced = reduce_ops_all(filter_wrapped_ops(ops, model))
    print_dict({key: value * scale for key, value in ops_reduced.items()})
    base_dir = path.join("operations", "evaluate", "jhmdb_openpose_interpolated")
    if not path.isdir(base_dir):
        os.makedirs(base_dir)
    name = "f{}{}".format(args.factor, args.id)
    with open(path.join(base_dir, "{}.p".format(name)), "wb") as f:
        pickle.dump(ops, f)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("factor", type=int, help="the interpolation factor to use")

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
