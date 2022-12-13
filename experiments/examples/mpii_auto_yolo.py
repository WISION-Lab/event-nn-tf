#!/usr/bin/env python3

import os
import os.path as path

from numpy.random import default_rng

from datasets import youtube_vis
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from models.yolo import apply_bn_gamma, postprocess_video, visualize_video, yolo_v3
from utils.misc import save_video

size = 288, 512

# Load the model.
model = yolo_v3(size, darknet_weights=path.join("weights", "yolo_v3.weights"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.06)
apply_bn_gamma(model)

# Load the data.
(frames, _, _, _), n_items = youtube_vis.load_data(
    path.join("data", "mpii"), "auto_yolo", size=size, n_frames_filter=41
)

output_dir = path.join("outputs", "mpii_auto_yolo")
os.makedirs(output_dir, exist_ok=True)
subset = default_rng(seed=3).permutation(n_items)[:10]
for i, video in enumerate(frames):
    if i not in subset:
        continue

    boxes, classes = model.predict_event(video, temporal_axis=1, static_graph=True)
    results = postprocess_video(boxes[0], classes[0])
    visualized = visualize_video(video[0], results, with_text=False, rect_thickness=3)
    save_video(visualized, path.join(output_dir, "{}_event.mp4".format(i)), keep_images=True)

    boxes, classes = model.predict_conventional(video, temporal_axis=1, static_graph=True)
    results = postprocess_video(boxes[0], classes[0])
    visualized = visualize_video(video[0], results, with_text=False, rect_thickness=3)
    save_video(
        visualized, path.join(output_dir, "{}_conventional.mp4".format(i)), keep_images=True
    )
