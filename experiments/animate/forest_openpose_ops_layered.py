#!/usr/bin/env python3

import os.path as path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from models.openpose import openpose_mpii
from utils.animation import op_line
from utils.misc import conv_ops_by_layer, trailing_average

with open(path.join("operations", "examples", "forest_openpose_conventional.p"), "rb") as f:
    ops_conventional = pickle.load(f)

with open(path.join("operations", "examples", "forest_openpose_event.p"), "rb") as f:
    ops_event = pickle.load(f)

model = openpose_mpii((64, 64))

ops_conventional = conv_ops_by_layer(ops_conventional, model)
ops_event = conv_ops_by_layer(ops_event, model)
ops_ratio = ops_event / ops_conventional
ops_ratio[:, 0] = ops_ratio[:, 1]
shallow = trailing_average(np.mean(ops_ratio[:31], axis=0), size=5)
middle = trailing_average(np.mean(ops_ratio[31:62], axis=0), size=5)
deep = trailing_average(np.mean(ops_ratio[62:], axis=0), size=5)

plt.style.use(path.join("styles", "ppt.mplstyle"))
op_line(
    [shallow, middle, deep],
    path.join("animations", "forest_openpose_ops_layered.mp4"),
    labels=["Shallow", "Middle", "Deep"],
    ylim=(0.0, 0.2),
)
