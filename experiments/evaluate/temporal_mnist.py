#!/usr/bin/env python3

import os.path as path
import pickle
from collections import OrderedDict

import numpy as np
from tensorflow.keras.metrics import CategoricalAccuracy

from datasets import temporal_mnist
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.dense import dense
from utils.misc import print_dict

# Load the data.
x_test, y_test = temporal_mnist.load_data(path.join("data", "temporal_mnist.npz"), "test")

# Load the model.
model = dense(
    input_size=x_test.shape[-1], npz_weights=path.join("weights", "sigma_delta_dense_mnist.npz")
)
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)

# Compute conventional metric performance.
results = model.evaluate_conventional(
    [CategoricalAccuracy()], x=x_test, y=y_test, temporal_axis=1, static_graph=True
)
print("Conventional metrics:")
print_dict(results)
with open(path.join("outputs", "temporal_mnist_conventional.p"), "wb") as f:
    pickle.dump(results, f)

# Count conventional operations.
ops = model.count_ops_conventional(x_test, temporal_axis=1, static_graph=True)
print("Conventional ops:")
print_dict(reduce_ops_all(filter_wrapped_ops(ops, model, filter_types=["Dense", "Bias"])))
with open(path.join("operations", "evaluate", "temporal_mnist_conventional.p"), "wb") as f:
    pickle.dump(ops, f)

# Test various thresholds so we can draw a Pareto curve.
all_results = OrderedDict()
all_ops = OrderedDict()
for threshold_pow in np.linspace(start=-1.5, stop=-0.2, num=14):
    print()
    print("Threshold: 10^{:.2g}".format(threshold_pow), flush=True)

    threshold = 10.0 ** threshold_pow
    for layer in model.gates:
        layer.policy.schedule.scale.assign(threshold)

    # Compute event metric performance.
    results = model.evaluate_event(
        [CategoricalAccuracy()], x=x_test, y=y_test, temporal_axis=1, static_graph=True
    )
    print("Event metrics:")
    print_dict(results)
    all_results[str(threshold_pow)] = results

    # Count event operations.
    ops = model.count_ops_event(x_test, temporal_axis=1, static_graph=True)
    print("Event ops:")
    print_dict(reduce_ops_all(filter_wrapped_ops(ops, model, filter_types=["Dense", "Bias"])))
    all_ops[str(threshold_pow)] = ops

with open(path.join("outputs", "temporal_mnist_event.p"), "wb") as f:
    pickle.dump(all_results, f)

with open(path.join("operations", "evaluate", "temporal_mnist_event.p"), "wb") as f:
    pickle.dump(all_ops, f)
