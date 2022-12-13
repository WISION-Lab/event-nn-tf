#!/usr/bin/env python3

import os.path as path
import pickle

import tensorflow as tf

from datasets import mpii
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from metrics.pose import PCKhMultiPerson
from models.openpose import openpose_mpii, postprocess_pckh_mpii
from utils.misc import print_dict

size = 288, 512

# Load the data.
(frames, frame_masks, joints, heads), n_items = mpii.load_data(
    path.join("data", "mpii"), "train", size=size
)
labels = tf.data.Dataset.zip((frames, frame_masks, joints, heads))
data = tf.data.Dataset.zip((frames, labels))

# Load the model.
model = openpose_mpii(size, npz_weights=path.join("weights", "openpose_mpii.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.05)

# Compute conventional metric performance.
results = model.evaluate_conventional(
    [PCKhMultiPerson(postprocess_func=postprocess_pckh_mpii)],
    data,
    data_steps=n_items,
    temporal_axis=1,
    verbose=True,
    static_graph=True,
)
print("Conventional metrics:")
print_dict(results)

# Compute event metric performance.
results = model.evaluate_event(
    [PCKhMultiPerson(postprocess_func=postprocess_pckh_mpii)],
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
with open(path.join("operations", "evaluate", "mpii_train_conventional.p"), "wb") as f:
    pickle.dump(ops, f)

# Count event operations.
ops = model.count_ops_event(
    frames, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=True
)
print("Event ops:")
print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
with open(path.join("operations", "evaluate", "mpii_train_event.p"), "wb") as f:
    pickle.dump(ops, f)
