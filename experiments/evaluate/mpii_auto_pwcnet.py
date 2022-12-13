#!/usr/bin/env python3

import os.path as path
import pickle

import tensorflow as tf

from datasets import sintel
from datasets.mpii import MAJOR_CAMERA_MOTION_TEST, MINOR_CAMERA_MOTION_TEST, NO_CAMERA_MOTION_TEST
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.pwcnet import pwcnet
from utils.misc import print_dict

size = 320, 512

# Load the model.
model = pwcnet(size, npz_weights=path.join("weights", "pwcnet.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.01)

for video_id_filter, name in zip(
    [MAJOR_CAMERA_MOTION_TEST, MINOR_CAMERA_MOTION_TEST, NO_CAMERA_MOTION_TEST, None],
    ["major", "minor", "none", "all"],
):
    print("Camera motion: {}".format(name))

    # Load the data.
    (frames, flow), n_items = sintel.load_data(
        path.join("data", "mpii"),
        "auto_pwcnet",
        size=size,
        n_frames_filter=41,
        video_id_filter=video_id_filter,
        input_size=size,
    )
    data = tf.data.Dataset.zip((frames, flow))

    # Compute event metric performance.
    results = model.evaluate_event(
        [sintel.EPE()],
        data,
        data_steps=n_items,
        temporal_axis=1,
        verbose=True,
        static_graph=True,
    )
    print("Event metrics:")
    print_dict(results)

    # Count conventional operations.
    ops = model.count_ops_conventional(
        frames, data_steps=1, temporal_axis=1, verbose=True, static_graph=True
    )
    print("Conventional ops:")
    print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
    with open(
        path.join("operations", "evaluate", "mpii_auto_pwcnet_conventional_{}.p".format(name)),
        "wb",
    ) as f:
        pickle.dump(ops, f)

    # Count event operations.
    ops = model.count_ops_event(
        frames, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=True
    )
    ops_filtered = filter_wrapped_ops(ops, model)
    print("Event:")
    print_dict(reduce_ops_all(ops_filtered))
    with open(
        path.join("operations", "evaluate", "mpii_auto_pwcnet_event_{}.p".format(name)), "wb"
    ) as f:
        pickle.dump(ops, f)

    print()
