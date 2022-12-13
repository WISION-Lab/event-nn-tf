#!/usr/bin/env python3

import os.path as path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from models.openpose import openpose_mpii
from utils.animation import op_line
from utils.misc import conv_ops_by_layer, trailing_average

with open(path.join("operations", "examples", "pan_yolo_conventional.p"), "rb") as f:
    ops_conventional = pickle.load(f)

with open(path.join("operations", "examples", "pan_yolo_event.p"), "rb") as f:
    ops_event = pickle.load(f)

with open(path.join("operations", "examples", "pan_yolo_memory_loss.p"), "rb") as f:
    ops_memory_loss = pickle.load(f)

model = openpose_mpii((64, 64))

ops_conventional = conv_ops_by_layer(ops_conventional, model)
ops_conventional = np.mean(ops_conventional, axis=0)
ops_event = conv_ops_by_layer(ops_event, model)
ops_event = np.mean(ops_event, axis=0)
ops_memory_loss = conv_ops_by_layer(ops_memory_loss, model)
ops_memory_loss = np.mean(ops_memory_loss, axis=0)
ops_ratio_event = ops_event / ops_conventional
ops_ratio_event[0] = ops_ratio_event[1]
ops_ratio_event = trailing_average(ops_ratio_event, size=5)
ops_ratio_memory_loss = ops_memory_loss / ops_conventional
ops_ratio_memory_loss[0] = ops_ratio_memory_loss[1]
ops_ratio_memory_loss = trailing_average(ops_ratio_memory_loss, size=5)

plt.style.use(path.join("styles", "ppt.mplstyle"))
op_line(
    [ops_ratio_memory_loss, ops_ratio_event],
    path.join("animations", "pan_yolo_ops.mp4"),
    colors=["#767171", "C3"],
    labels=["Without Long-Term Memory", "With Long-Term Memory"],
    ylim=(0.0, 0.5),
    legend_loc="lower right",
)
