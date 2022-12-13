#!/usr/bin/env python3

import os.path as path
import pickle

import matplotlib.pyplot as plt

from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.dense import dense

# Load and preprocess sigma-delta results.
with open(path.join("outputs", "temporal_mnist_sigma_delta.p"), "rb") as f:
    results_sigma_delta = pickle.load(f, encoding="latin1")
x_sigma_delta = []
y_sigma_delta = []
for key, results in results_sigma_delta.items():
    if key == "unoptimized":
        continue
    results = results[("temp_mnist", "test", "td")]
    x_sigma_delta.append((results["MFlops"] + results["MFlops-multadd"]) * 1.0e6)
    y_sigma_delta.append(1.0 - 0.01 * results["class_error"])

# Load and preprocess event error rates.
with open(path.join("outputs", "temporal_mnist_event.p"), "rb") as f:
    results_event = pickle.load(f)
y_event = []
for results in results_event.values():
    y_event.append((results["categorical_accuracy"]))

# Load and preprocess event operation counts.
with open(path.join("operations", "evaluate", "temporal_mnist_event.p"), "rb") as f:
    ops_event = pickle.load(f)
x_event = []
model = dense(input_size=784)
for ops in ops_event.values():
    ops_reduced = reduce_ops_all(filter_wrapped_ops(ops, model, filter_types=["Dense", "Bias"]))
    x_event.append(ops_reduced["math_ops"])

plt.style.use(path.join("styles", "eccv.mplstyle"))
fig = plt.figure(figsize=(2.1, 1.4))
ax = fig.subplots()

for i, (x, y, name) in enumerate(
    [
        (x_event, y_event, "Ours"),
        (x_sigma_delta, y_sigma_delta, "Rounding"),
    ]
):
    ax.plot(x, y, color="C{}".format(i + 1), zorder=-i)
    ax.scatter(x, y, s=10.0, color="C{}".format(i + 1), label=name, zorder=-i)
ax.set(ylabel="Accuracy", xscale="log", ylim=(0.85, 1.0))
ax.set_xlabel("Ops", labelpad=-9, loc="left")
ax.legend()

fig.savefig(path.join("figures", "sigma_delta_comparison.pdf"))
