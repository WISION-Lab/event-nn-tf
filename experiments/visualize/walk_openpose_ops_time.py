#!/usr/bin/env python3

import os.path as path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from models.openpose import openpose_mpii
from utils.misc import conv_ops_by_layer, centered_average

with open(path.join("operations", "examples", "walk_openpose_conventional.p"), "rb") as f:
    ops_conventional = pickle.load(f)
with open(path.join("operations", "examples", "walk_openpose_event.p"), "rb") as f:
    ops_event = pickle.load(f)

model = openpose_mpii((64, 64))

ops_conventional = conv_ops_by_layer(ops_conventional, model)
ops_event = conv_ops_by_layer(ops_event, model)

# Commented lines are for the paper version of the figure.

# plt.style.use(path.join("styles", "eccv.mplstyle"))
# fig = plt.figure(figsize=(4.7, 0.8))

# Quadruple the figure size when inserting into the poster.
plt.style.use(path.join("styles", "poster.mplstyle"))
fig = plt.figure(figsize=(4.2, 1.2))
ax = fig.subplots()

y = [
    np.sum(ops_event, axis=0) / np.sum(ops_conventional, axis=0),
    np.sum(ops_event[:31], axis=0) / np.sum(ops_conventional[:31], axis=0),
    np.sum(ops_event[31:62], axis=0) / np.sum(ops_conventional[31:62], axis=0),
    np.sum(ops_event[62:], axis=0) / np.sum(ops_conventional[62:], axis=0),
]
for y_i, label in zip(y, ["All", "Shallow", "Middle", "Deep"]):
    ax.plot(np.arange(1, len(y_i)), centered_average(y_i[1:], size=10), label=label)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
# ax.set(ylabel="Cost fraction", ylim=(0, 0.1), xticks=[0, 70, 160, 210, 250, 340])
ax.set(
    xlabel="Time Step",
    ylabel="Cost Fraction",
    ylim=(0, 0.1),
    # xticks=[0, 70, 160, 210, 250, 340],
    xticks=[70, 160, 210, 250, 340],
)

# fig.savefig(path.join("figures", "walk_openpose_ops_time.pdf"))
fig.savefig(path.join("figures", "walk_openpose_ops_time_poster.pdf"))
