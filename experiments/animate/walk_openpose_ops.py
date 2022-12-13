#!/usr/bin/env python3

import os.path as path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from models.openpose import openpose_mpii
from utils.animation import op_line
from utils.misc import conv_ops_by_layer, trailing_average

with open(path.join("operations", "examples", "walk_openpose_conventional.p"), "rb") as f:
    ops_conventional = pickle.load(f)

with open(path.join("operations", "examples", "walk_openpose_event.p"), "rb") as f:
    ops_event = pickle.load(f)

model = openpose_mpii((64, 64))

ops_conventional = conv_ops_by_layer(ops_conventional, model)
ops_conventional = np.mean(ops_conventional, axis=0)
ops_event = conv_ops_by_layer(ops_event, model)
ops_event = np.mean(ops_event, axis=0)
ops_ratio = ops_event / ops_conventional
ops_ratio[0] = ops_ratio[1]
ops_ratio = trailing_average(ops_ratio, size=5)

plt.style.use(path.join("styles", "ppt.mplstyle"))
op_line(
    [ops_ratio],
    path.join("animations", "walk_openpose_ops.mp4"),
    colors=["C3"],
    ylim=(0.0, 0.1),
)
