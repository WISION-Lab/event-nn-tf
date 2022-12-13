#!/usr/bin/env python3

import os.path as path

import tensorflow as tf

from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.yolo import (
    apply_bn_gamma,
    postprocess_video,
    preprocess_video,
    visualize_video,
    yolo_v3,
)
from utils.misc import print_dict, read_video, rescale_video, save_video

video_raw = read_video(path.join("inputs", "walk"))
video_raw = rescale_video(video_raw, scale=0.5)
video = preprocess_video(video_raw)
video = tf.expand_dims(video, axis=0)

model = yolo_v3(video.shape[2:4], darknet_weights=path.join("weights", "yolo_v3.weights"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.06)
apply_bn_gamma(model)

boxes, classes = model.predict_event(video, temporal_axis=1, static_graph=True)
results = postprocess_video(boxes[0], classes[0])
visualized = visualize_video(video_raw, results, with_text=False, rect_thickness=3)
save_video(visualized, path.join("outputs", "walk_yolo_event.mp4"), keep_images=True)

boxes, classes = model.predict_conventional(video, temporal_axis=1, static_graph=True)
results = postprocess_video(boxes[0], classes[0])
visualized = visualize_video(video_raw, results, with_text=False, rect_thickness=3)
save_video(visualized, path.join("outputs", "walk_yolo_conventional.mp4"), keep_images=True)

ops = model.count_ops_event(video, temporal_axis=1, static_graph=True)
ops = reduce_ops_all(filter_wrapped_ops(ops, model))
print("Event ops:")
print_dict(ops)

ops = model.count_ops_conventional(video, temporal_axis=1, static_graph=True)
ops = reduce_ops_all(filter_wrapped_ops(ops, model))
print("Conventional ops:")
print_dict(ops)
