#!/usr/bin/env python3

import os.path as path
import pickle

import tensorflow as tf

from datasets import sintel
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.pwcnet import pwcnet
from utils.misc import print_dict

size = 448, 1024

# Load the data.
(frames, flow), n_items = sintel.load_data(path.join("data", "sintel"), "auto", size=size)
data = tf.data.Dataset.zip((frames, flow))

# Load the model.
model = pwcnet(size, npz_weights=path.join("weights", "pwcnet.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.01)

# Compute event metric performance.
results = model.evaluate_event(
    [sintel.EPE()], data, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=False
)
print("Event metrics:")
print_dict(results)

# Count conventional operations.
ops = model.count_ops_conventional(
    frames, data_steps=1, temporal_axis=1, verbose=True, static_graph=False
)
print("Conventional ops:")
print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
with open(path.join("operations", "evaluate", "sintel_auto_conventional.p"), "wb") as f:
    pickle.dump(ops, f)

# Count event operations.
ops = model.count_ops_event(
    frames, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=False
)
print("Event ops:")
print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
with open(path.join("operations", "evaluate", "sintel_auto_event.p"), "wb") as f:
    pickle.dump(ops, f)
