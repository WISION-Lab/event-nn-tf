#!/usr/bin/env python3

import os.path as path
import pickle

import tensorflow as tf

from eventnn.model import EventModel
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from models.openpose import GATE_NAMES, openpose_mpii, preprocess_video
from utils.misc import read_video

# Don't downsize (unlike other videos) because the person we want to
# detect is pretty small.
video_raw = read_video(path.join("inputs", "forest"))
video = preprocess_video(video_raw)
video = tf.expand_dims(video, axis=0)

model = openpose_mpii(video.shape[2:4], npz_weights=path.join("weights", "openpose_mpii.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.05)

delta_outputs = []
mask_outputs = []
for name in GATE_NAMES:
    layer = model.get_layer(name)
    layer.all_outputs = True
    delta_outputs.append(layer.output[0])
    mask_outputs.append(layer.output[1])
model_transparent = EventModel(inputs=model.inputs, outputs=delta_outputs + mask_outputs)
outputs = model_transparent.predict_event(video, temporal_axis=1)
outputs = [output[0][..., :16] for output in outputs]
with open(path.join("outputs", "forest_openpose_delta_mask.p"), "wb") as f:
    pickle.dump(outputs, f)
