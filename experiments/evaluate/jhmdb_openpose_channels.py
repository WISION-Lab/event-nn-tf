#!/usr/bin/env python3

import os.path as path
import pickle

import tensorflow as tf

from datasets import jhmdb
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from metrics.pose import PCKSinglePerson
from models.openpose import openpose_mpii, postprocess_pck_jhmdb_mpii, preprocess_image
from utils.misc import print_dict

size = 240, 320

# Load the data.
(frames, joints), n_items = jhmdb.load_data(
    path.join("data", "jhmdb"), "test", size=size, preprocess_func=preprocess_image
)
labels = tf.data.Dataset.zip((frames, joints))
data = tf.data.Dataset.zip((frames, labels))

for chunk_shape, chunk_channels, threshold in zip([None, (1, 1)], [False, True], [0.05, 0.02]):
    print("Chunk shape: {}".format(chunk_shape))
    print("Chunk channels: {}".format(chunk_channels))

    # Load the model.
    model = openpose_mpii(size, npz_weights=path.join("weights", "openpose_mpii.npz"))
    for layer in model.gates:
        layer.policy = Threshold(
            Constant(), warmup=True, chunk_shape=chunk_shape, chunk_channels=chunk_channels
        )
        layer.policy.schedule.scale.assign(threshold)

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

    # Count event operations.
    ops = model.count_ops_event(
        frames, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=True
    )
    print("Event ops:")
    print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
    name = "with" if chunk_channels else "without"
    filename = "jhmdb_openpose_channels_{}.p".format(name)
    with open(path.join("operations", "evaluate", filename), "wb") as f:
        pickle.dump(ops, f)

    print()

    # Delete the model to prevent OOM errors.
    del model
