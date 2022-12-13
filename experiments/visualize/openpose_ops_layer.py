#!/usr/bin/env python3

import os.path as path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import median_filter

from datasets.mpii import split_by_camera_motion
from eventnn.utils import filter_wrapped_ops, reduce_ops_items_time, reduce_ops_time
from models.openpose import openpose_mpii

# noinspection DuplicatedCode
with open(path.join("operations", "evaluate", "mpii_auto_openpose_conventional_all.p"), "rb") as f:
    ops_mpii_conventional = pickle.load(f)
with open(path.join("operations", "evaluate", "mpii_auto_openpose_event_all.p"), "rb") as f:
    ops_mpii_event = pickle.load(f)

with open(path.join("operations", "evaluate", "jhmdb_openpose_conventional", "base.p"), "rb") as f:
    ops_jhmdb_conventional = pickle.load(f)
with open(path.join("operations", "evaluate", "jhmdb_openpose_event.p"), "rb") as f:
    ops_jhmdb_event = pickle.load(f)

model = openpose_mpii((64, 64))

ops_mpii_split = filter_wrapped_ops(ops_mpii_event, model, filter_types=["Conv2D"])
ops_mpii_split = reduce_ops_time(ops_mpii_split)
ops_mpii_split = [item["math_ops"] for item in ops_mpii_split.values()]
ops_mpii_split = list(filter(any, ops_mpii_split))
ops_mpii_split = split_by_camera_motion(
    list(zip(*ops_mpii_split)), path.join("data", "mpii"), "auto_openpose"
)
ops_mpii_split = np.array([np.mean(item, axis=0) for item in ops_mpii_split[:-1]])


def conv_ops_by_layer(ops):
    ops = filter_wrapped_ops(ops, model, filter_types=["Conv2D"])
    ops = reduce_ops_items_time(ops)
    ops = [item["math_ops"] for item in ops.values()]
    ops = np.array(list(filter(lambda x: x > 0, ops)))
    return ops


ops_mpii_conventional = conv_ops_by_layer(ops_mpii_conventional)
ops_mpii_event = conv_ops_by_layer(ops_mpii_event)
ratio_mpii_event = ops_mpii_event / ops_mpii_conventional
ratio_mpii_split = ops_mpii_split / ops_mpii_conventional

ops_jhmdb_conventional = conv_ops_by_layer(ops_jhmdb_conventional)
ops_jhmdb_event = conv_ops_by_layer(ops_jhmdb_event)
ratio_jhmdb_event = ops_jhmdb_event / ops_mpii_conventional

filter_size = 5

plt.style.use(path.join("styles", "eccv.mplstyle"))
fig = plt.figure(figsize=(2.6, 1.5))
ax = fig.subplots()

for series, label in zip(
    ratio_mpii_split, ["MPII - None", "MPII - Minor", "MPII - Major"]
):
    ax.plot(median_filter(series, size=filter_size), label=label)
ax.plot(median_filter(ratio_mpii_event, size=filter_size), label="MPII - All")
ax.plot(median_filter(ratio_jhmdb_event, size=filter_size), label="JHMDB")
ax.legend()
ax.set(ylim=(0.0, 0.3), xticks=[0, 90], yticks=[0.0, 0.3])
ax.set_xlabel("Layer Index", labelpad=-6)
ax.set_ylabel("Cost Fraction", labelpad=-9)
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

fig.savefig(path.join("figures", "openpose_ops_layer.pdf"))
