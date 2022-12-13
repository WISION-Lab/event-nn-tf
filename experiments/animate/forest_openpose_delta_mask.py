#!/usr/bin/env python3

import os
import os.path as path
import pickle

import numpy as np

from models.openpose import GATE_NAMES
from utils.animation import delta_cmap, mask_cmap
from utils.misc import save_video

with open(path.join("outputs", "forest_openpose_delta_mask.p"), "rb") as f:
    outputs = pickle.load(f)

# Remove time step 0 because warmup=True.
outputs = [np.concatenate([output[1:2], output[1:]], axis=0) for output in outputs]

output_filter = ["vgg_pool_1", "vgg_pool_2", "block_2_heatmap", "block_4_heatmap"]
indices = [4, 3, 4, 4, 4, 4, 4, 4]
base_dir = path.join("animations", "forest_openpose_delta_mask")
os.makedirs(base_dir, exist_ok=True)
n = len(outputs) // 2
for deltas, masks, gate_name, index in zip(outputs[:n], outputs[n:], GATE_NAMES, indices):
    if gate_name not in output_filter:
        continue
    save_video(
        delta_cmap(deltas[..., index], scale=2.0),
        path.join(base_dir, "deltas_{}.mp4".format(gate_name)),
        resize_method="nearest",
    )
    save_video(
        mask_cmap(masks[..., index], scale=2.0),
        path.join(base_dir, "masks_{}.mp4".format(gate_name)),
        resize_method="nearest",
    )
