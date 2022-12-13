#!/usr/bin/env python3

import os.path as path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from datasets.mpii import split_by_camera_motion
from eventnn.utils import filter_wrapped_ops, reduce_ops_layers_time
from models.openpose import openpose_mpii

# noinspection DuplicatedCode
with open(path.join("operations", "evaluate", "mpii_auto_openpose_conventional_all.p"), "rb") as f:
    ops_conventional = pickle.load(f)
with open(path.join("operations", "evaluate", "mpii_auto_openpose_event_all.p"), "rb") as f:
    ops_event = pickle.load(f)

model = openpose_mpii((64, 64))

ops_conventional = filter_wrapped_ops(ops_conventional, model)
ops_conventional = np.array(reduce_ops_layers_time(ops_conventional)["math_ops"])

ops_event = filter_wrapped_ops(ops_event, model)
ops_event = np.array(reduce_ops_layers_time(ops_event)["math_ops"])
ops_event = ops_event / ops_conventional
ops_event = split_by_camera_motion(ops_event, path.join("data", "mpii"), "auto_openpose")

plt.style.use(path.join("styles", "eccv.mplstyle"))
fig = plt.figure(figsize=(2.4, 1.3))
ax = fig.subplots()

ax.boxplot(ops_event[:-1], labels=["None", "Minor", "Major"])
ax.set(xlabel="Camera motion", ylim=(0.0, 0.25), yticks=[0.0, 0.25])
ax.set_ylabel("Cost fraction", labelpad=-14)
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.tick_params(axis="x", length=0)

fig.savefig(path.join("figures", "mpii_ops_dist.pdf"))
