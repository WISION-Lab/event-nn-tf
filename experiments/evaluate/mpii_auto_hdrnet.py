#!/usr/bin/env python3

import os.path as path
import pickle

import tensorflow as tf

from datasets import mpii_hdrnet
from datasets.mpii import MAJOR_CAMERA_MOTION_TEST, MINOR_CAMERA_MOTION_TEST, NO_CAMERA_MOTION_TEST
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.hdrnet import hdrnet
from utils.misc import print_dict

size = 540, 960

# Load the model.
model = hdrnet(size, npz_weights=path.join("weights", "hdrnet_local_laplacian_normal.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.1)

for video_id_filter, name in zip(
    [MAJOR_CAMERA_MOTION_TEST, MINOR_CAMERA_MOTION_TEST, NO_CAMERA_MOTION_TEST, None],
    ["major", "minor", "none", "all"],
):
    print("Camera motion: {}".format(name))

    # Load the data.
    (frames, labels), n_items = mpii_hdrnet.load_data(
        path.join("data", "mpii"),
        "auto_hdrnet",
        size=size,
        n_frames_filter=41,
        video_id_filter=video_id_filter,
    )
    data = tf.data.Dataset.zip((frames, labels))

    # Compute event metric performance.
    results = model.evaluate_event(
        [mpii_hdrnet.PSNR()],
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
    ops_filtered = filter_wrapped_ops(ops, model)
    print("Conventional (all):")
    print_dict(reduce_ops_all(ops_filtered))
    remove_keys = list(filter(lambda x: x.startswith("guidemap"), ops_filtered.keys()))
    for key in remove_keys:
        ops_filtered.pop(key)
    print("Conventional (excluding guidemap overhead):")
    print_dict(reduce_ops_all(ops_filtered))
    with open(
        path.join("operations", "evaluate", "mpii_auto_hdrnet_conventional_{}.p".format(name)),
        "wb",
    ) as f:
        pickle.dump(ops, f)

    # Count event operations.
    ops = model.count_ops_event(
        frames, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=True
    )
    ops_filtered = filter_wrapped_ops(ops, model)
    print("Event (all):")
    print_dict(reduce_ops_all(ops_filtered))
    remove_keys = list(filter(lambda x: x.startswith("guidemap"), ops_filtered.keys()))
    for key in remove_keys:
        ops_filtered.pop(key)
    print("Event (excluding guidemap overhead):")
    print_dict(reduce_ops_all(ops_filtered))
    with open(
        path.join("operations", "evaluate", "mpii_auto_hdrnet_event_{}.p".format(name)), "wb"
    ) as f:
        pickle.dump(ops, f)

    print()
