#!/usr/bin/env python3

import os.path as path
import pickle

import tensorflow as tf

from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.openpose import openpose_mpii, postprocess_video, preprocess_video, visualize_video
from utils.misc import print_dict, read_video, rescale_video, save_video

# Don't downsize (unlike other videos) because the person we want to
# detect is pretty small.
video_raw = read_video(path.join("inputs", "forest"))
video = preprocess_video(video_raw)
video = tf.expand_dims(video, axis=0)
video_raw = rescale_video(video_raw, scale=0.5)

model = openpose_mpii(video.shape[2:4], npz_weights=path.join("weights", "openpose_mpii.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.05)

pafs, heatmaps = model.predict_event(video, temporal_axis=1, static_graph=True)
output = postprocess_video(video_raw, pafs[0], heatmaps[0], mode="mpii")
visualized = visualize_video(video_raw, output)
save_video(visualized, path.join("outputs", "forest_openpose_event.mp4"), keep_images=True)

pafs, heatmaps = model.predict_conventional(video, temporal_axis=1, static_graph=True)
output = postprocess_video(video_raw, pafs[0], heatmaps[0], mode="mpii")
visualized = visualize_video(video_raw, output)
save_video(visualized, path.join("outputs", "forest_openpose_conventional.mp4"), keep_images=True)

ops = model.count_ops_event(video, temporal_axis=1, temporal_mean=False, static_graph=True)
print("Event ops:")
print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
with open(path.join("operations", "examples", "forest_openpose_event.p"), "wb") as f:
    pickle.dump(ops, f)

ops = model.count_ops_conventional(video, temporal_axis=1, temporal_mean=False, static_graph=True)
print("Conventional ops:")
print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
with open(path.join("operations", "examples", "forest_openpose_conventional.p"), "wb") as f:
    pickle.dump(ops, f)
