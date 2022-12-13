#!/usr/bin/env python3

import os.path as path
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from datasets import jhmdb
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from models.openpose import openpose_mpii, postprocess_pck_jhmdb_mpii, preprocess_image


def main(args):
    # Load the data.
    (frames, joints), n_items = jhmdb.load_data(
        path.join("data", "jhmdb"),
        "test",
        size=tuple(args.size),
        preprocess_func=preprocess_image,
    )
    labels = tf.data.Dataset.zip((frames, joints))

    # Uncomment to cap the number of items (for quick tests).
    # n_items = 10

    # Load the model.
    model = openpose_mpii(tuple(args.size), npz_weights=path.join("weights", "openpose_mpii.npz"))
    for layer in model.gates:
        schedule = Constant()
        schedule.scale.assign(args.scale)
        layer.policy = Threshold(schedule, warmup=True)

    outputs = model.predict_event(
        frames,
        data_steps=n_items,
        temporal_axis=1,
        verbose=True,
        static_graph=True,
    )

    # Determine the average motion of joints between frames.
    total_motion_pred = 0.0
    total_motion_true = 0.0
    n_motions_pred = 0
    n_motions_true = 0
    for y_true, y_pred in zip(labels, zip(*outputs)):
        y_true = (np.array(y_true[0]), y_true[1].numpy())
        for t in range(y_pred[0].shape[0] - 1):
            y_pred_1 = (y_pred[0][np.newaxis, t], y_pred[1][np.newaxis, t])
            y_pred_2 = (y_pred[0][np.newaxis, t + 1], y_pred[1][np.newaxis, t + 1])
            y_true_1 = (y_true[0][:, t], y_true[1][:, t])
            y_true_2 = (y_true[0][:, t + 1], y_true[1][:, t + 1])
            person_pred_1 = postprocess_pck_jhmdb_mpii(y_true_1, y_pred_1)[1][0]
            person_pred_2 = postprocess_pck_jhmdb_mpii(y_true_2, y_pred_2)[1][0]
            person_true_1 = y_true_1[1][0]
            person_true_2 = y_true_2[1][0]
            for joint_1, joint_2 in zip(person_pred_1, person_pred_2):
                if np.all(joint_1 != -1.0) and np.all(joint_2 != 1.0):
                    total_motion_pred += np.linalg.norm(joint_1 - joint_2)
                    n_motions_pred += 1
            for joint_1, joint_2 in zip(person_true_1, person_true_2):
                if np.all(~np.isnan(joint_1)) and np.all(~np.isnan(joint_2)):
                    total_motion_true += np.linalg.norm(joint_1 - joint_2)
                    n_motions_true += 1
    print("Mean predicted motion: {:.4g}".format(total_motion_pred / n_motions_pred))
    print("Mean actual motion: {:.4g}".format(total_motion_true / n_motions_true))


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("scale", type=float, help="the threshold scale to apply")

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
