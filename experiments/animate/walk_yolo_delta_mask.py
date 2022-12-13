#!/usr/bin/env python3

import os.path as path

import tensorflow as tf

from eventnn.model import EventModel
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from models.yolo import apply_bn_gamma, preprocess_image, yolo_v3
from utils.animation import delta_mask
from utils.misc import read_video, rescale_video, save_video

video_raw = read_video(path.join("inputs", "walk"))
video_raw = rescale_video(video_raw, scale=0.5)
video = tf.stack([preprocess_image(frame) for frame in video_raw])
video = tf.expand_dims(video, axis=0)

model = yolo_v3(video.shape[2:4], darknet_weights=path.join("weights", "yolo_v3.weights"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.06)
apply_bn_gamma(model)

delta_outputs = [model.gates[i].output[0] for i in range(1, 61, 10)]
mask_outputs = [model.gates[i].output[1] for i in range(1, 61, 10)]
model_transparent = EventModel(inputs=model.inputs, outputs=delta_outputs + mask_outputs)
outputs = model_transparent.predict_event(video, temporal_axis=1)

# Remove time step 0 because warmup=True.
outputs_vis = [output[0][1:] for output in outputs]
delta_mask = delta_mask(
    video_raw[1:],
    [
        tf.unstack(outputs_vis[0][..., :4], axis=-1),
        tf.unstack(outputs_vis[1][..., :32], axis=-1),
        tf.unstack(outputs_vis[2][..., :32], axis=-1),
        tf.unstack(outputs_vis[3][..., :64], axis=-1),
        tf.unstack(outputs_vis[4][..., :64], axis=-1),
        tf.unstack(outputs_vis[5][..., :64], axis=-1),
    ],
    [
        tf.unstack(outputs_vis[6][..., :4], axis=-1),
        tf.unstack(outputs_vis[7][..., :32], axis=-1),
        tf.unstack(outputs_vis[8][..., :32], axis=-1),
        tf.unstack(outputs_vis[9][..., :64], axis=-1),
        tf.unstack(outputs_vis[10][..., :64], axis=-1),
        tf.unstack(outputs_vis[11][..., :64], axis=-1),
    ],
)
save_video(delta_mask, path.join("outputs", "walk_yolo_delta_mask.mp4"))
