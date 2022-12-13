#!/usr/bin/env python3

import os
import os.path as path

from numpy.random import default_rng

from datasets import mpii_hdrnet
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from models.hdrnet import hdrnet
from utils.misc import save_video

size = 540, 960

# Load the model.
model = hdrnet(size, npz_weights=path.join("weights", "hdrnet_local_laplacian_normal.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.1)

# Load the data.
(frames, _), n_items = mpii_hdrnet.load_data(
    path.join("data", "mpii"), "auto_hdrnet", size=size, n_frames_filter=41
)

output_dir = path.join("outputs", "mpii_auto_hdrnet")
os.makedirs(output_dir, exist_ok=True)
subset = default_rng(seed=0).permutation(n_items)[:10]
for i, video in enumerate(frames):
    if i not in subset:
        continue
    save_video(video[0], path.join(output_dir, "{}_input.mp4".format(i)), keep_images=True)

    output = model.predict_event(video, temporal_axis=1, static_graph=True)
    save_video(output[0], path.join(output_dir, "{}_event.mp4".format(i)), keep_images=True)

    output = model.predict_conventional(video, temporal_axis=1, static_graph=True)
    save_video(output[0], path.join(output_dir, "{}_conventional.mp4".format(i)), keep_images=True)
