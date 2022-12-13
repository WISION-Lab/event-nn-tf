#!/usr/bin/env python3

import os
import os.path as path

from numpy.random import default_rng

from datasets import mpii
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from models.openpose import (
    openpose_mpii,
    postprocess_video,
    undo_preprocess_video,
    visualize_video,
)
from utils.misc import save_video

size = 288, 512

# Load the model.
model = openpose_mpii(size, npz_weights=path.join("weights", "openpose_mpii.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.05)

# Load the data.
(frames, _, _), n_items = mpii.load_data(
    path.join("data", "mpii"), "auto_openpose", size=size, n_frames_filter=41
)

output_dir = path.join("outputs", "mpii_auto_openpose")
os.makedirs(output_dir, exist_ok=True)
subset = default_rng(seed=1).permutation(n_items)[:10]
for i, video in enumerate(frames):
    if i not in subset:
        continue
    video_raw = undo_preprocess_video(video[0])

    pafs, heatmaps = model.predict_event(video, temporal_axis=1, static_graph=True)
    output = postprocess_video(video_raw, pafs[0], heatmaps[0], mode="mpii")
    visualized = visualize_video(video_raw, output)
    save_video(visualized, path.join(output_dir, "{}_event.mp4".format(i)), keep_images=True)

    pafs, heatmaps = model.predict_conventional(video, temporal_axis=1, static_graph=True)
    output = postprocess_video(video_raw, pafs[0], heatmaps[0], mode="mpii")
    visualized = visualize_video(video_raw, output)
    save_video(
        visualized, path.join(output_dir, "{}_conventional.mp4".format(i)), keep_images=True
    )
