#!/usr/bin/env python3

import os.path as path

import tensorflow as tf

from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.pwcnet import postprocess_video, preprocess_video, pwcnet, visualize_video
from utils.misc import print_dict, read_video, rescale_video, save_video

video_raw = read_video(path.join("inputs", "walk"))
video_raw = rescale_video(video_raw, scale=0.5)
video = preprocess_video(video_raw)
video = tf.expand_dims(video, axis=0)

model = pwcnet(video.shape[3:5], npz_weights=path.join("weights", "pwcnet.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.01)

output = model.predict_event(video, temporal_axis=1, static_graph=True)
output = postprocess_video(output[0], video_raw.shape[1:3])
visualized = visualize_video(output)
save_video(visualized, path.join("outputs", "walk_pwcnet_event.mp4"), keep_images=True)

output = model.predict_conventional(video, temporal_axis=1, static_graph=True)
output = postprocess_video(output[0], video_raw.shape[1:3])
visualized = visualize_video(output)
save_video(visualized, path.join("outputs", "walk_pwcnet_conventional.mp4"), keep_images=True)

ops = model.count_ops_event(video, temporal_axis=1, static_graph=True)
ops = reduce_ops_all(filter_wrapped_ops(ops, model))
print("Event ops:")
print_dict(ops)

ops = model.count_ops_conventional(video, temporal_axis=1, static_graph=True)
ops = reduce_ops_all(filter_wrapped_ops(ops, model))
print("Conventional ops:")
print_dict(ops)
