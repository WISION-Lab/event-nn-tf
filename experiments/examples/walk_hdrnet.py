#!/usr/bin/env python3

import os.path as path

import tensorflow as tf

from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.hdrnet import hdrnet, preprocess_video
from utils.misc import print_dict, read_video, save_video

video_raw = read_video(path.join("inputs", "walk"))
video = preprocess_video(video_raw)
video = tf.expand_dims(video, axis=0)

model = hdrnet(
    video.shape[2:4], npz_weights=path.join("weights", "hdrnet_local_laplacian_normal.npz")
)
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.1)

output = model.predict_event(video, temporal_axis=1, static_graph=True)
save_video(output[0], path.join("outputs", "walk_hdrnet_event.mp4"), keep_images=True)

output = model.predict_conventional(video, temporal_axis=1, static_graph=True)
save_video(output[0], path.join("outputs", "walk_hdrnet_conventional.mp4"), keep_images=True)

ops = model.count_ops_event(video, temporal_axis=1, static_graph=True)
ops_filtered = filter_wrapped_ops(ops, model)
print("Event ops (all):")
print_dict(reduce_ops_all(ops_filtered))
remove_keys = list(filter(lambda x: x.startswith("guidemap"), ops_filtered.keys()))
for key in remove_keys:
    ops_filtered.pop(key)
print("Event ops (excluding guidemap overhead):")
print_dict(reduce_ops_all(ops_filtered))

ops = model.count_ops_conventional(video, temporal_axis=1, static_graph=True)
ops_filtered = filter_wrapped_ops(ops, model)
print("Conventional ops:")
print_dict(reduce_ops_all(ops_filtered))
