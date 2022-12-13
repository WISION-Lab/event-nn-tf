#!/usr/bin/env python3

import numpy as np

from models.openpose import openpose_mpii
from models.yolo import yolo_v3

for model, name in (openpose_mpii((240, 320)), "OpenPose"), (yolo_v3((224, 384)), "YOLO"):
    n_overhead = 0
    for layer in model.accumulators:
        n_overhead += int(np.prod(layer.accumulator.shape))
    for layer in model.gates:
        n_overhead += 2 * int(np.prod(layer.best.shape))
    print("{} overhead: {:.4g} MB".format(name, 4 * n_overhead / 1e6))
    n_weights = 0
    for weight in model.trainable_weights:
        n_weights += int(np.prod(weight.shape))
    print("{} weights: {:.4g} MB".format(name, 4 * n_weights / 1e6))
