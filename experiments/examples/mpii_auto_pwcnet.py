#!/usr/bin/env python3

import os
import os.path as path

from numpy.random import default_rng

from datasets import sintel
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from models.pwcnet import pwcnet, postprocess_video, undo_preprocess_video, visualize_video
from utils.misc import save_video

size = 320, 512

# Load the model.
model = pwcnet(size, npz_weights=path.join("weights", "pwcnet.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.01)

# Load the data.
(frames, _), n_items = sintel.load_data(
    path.join("data", "mpii"), "auto_pwcnet", size=size, n_frames_filter=41, input_size=size
)

output_dir = path.join("outputs", "mpii_auto_pwcnet")
os.makedirs(output_dir, exist_ok=True)
subset = default_rng(seed=2).permutation(n_items)[:10]
for i, video in enumerate(frames):
    if i not in subset:
        continue
    video_raw = undo_preprocess_video(video[0])
    save_video(video_raw, path.join(output_dir, "{}_input.mp4".format(i)), keep_images=True)

    output = model.predict_event(video, temporal_axis=1, static_graph=True)
    output = postprocess_video(output[0], video.shape[3:5])
    visualized = visualize_video(output)
    save_video(visualized, path.join(output_dir, "{}_event.mp4".format(i)), keep_images=True)

    output = model.predict_conventional(video, temporal_axis=1, static_graph=True)
    output = postprocess_video(output[0], video.shape[3:5])
    visualized = visualize_video(output)
    save_video(
        visualized, path.join(output_dir, "{}_conventional.mp4".format(i)), keep_images=True
    )
