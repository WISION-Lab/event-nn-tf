#!/usr/bin/env python3

import os.path as path

import numpy as np
import tensorflow as tf

from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import extract_ops_item, reduce_ops_layers
from models.yolo import (
    apply_bn_gamma,
    postprocess_image,
    preprocess_image,
    visualize_image,
    yolo_v3,
)
from utils.animation import op_bar
from utils.misc import read_video, rescale_video, save_video

video_raw = rescale_video(read_video(path.join("inputs", "walk")), scale=0.5)
video = tf.stack([preprocess_image(frame) for frame in video_raw])
video = tf.expand_dims(video, axis=0)

model = yolo_v3(video.shape[2:4], darknet_weights=path.join("weights", "yolo_v3.weights"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.06)
apply_bn_gamma(model)

boxes, classes = model.predict_event(video, temporal_axis=1)
ops_event = model.count_ops_event(video, temporal_axis=1)
ops_conventional = model.count_ops_conventional(video, temporal_axis=1)
frames_annotated = []
for t, frame in enumerate(video_raw):
    results = postprocess_image(boxes[0][t], classes[0][t])
    frame_annotated = visualize_image(frame, *results, font_thickness=2, image_scale=2.0)
    frames_annotated.append(frame_annotated)
frames_annotated = np.stack(frames_annotated, axis=0)
ops_array = reduce_ops_layers(extract_ops_item(ops_event, 0))["math_ops"][0]
max_ops = np.max(reduce_ops_layers(extract_ops_item(ops_conventional, 0))["math_ops"][0])
op_video = op_bar(frames_annotated, ops_array, max_ops)
save_video(op_video, path.join("outputs", "walk_yolo_op_bar.mp4"))
