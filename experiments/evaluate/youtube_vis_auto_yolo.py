#!/usr/bin/env python3

import os
import os.path as path
import pickle
import subprocess
from shutil import rmtree

import tensorflow as tf

from datasets import youtube_vis
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.yolo import FileDump, apply_bn_gamma, yolo_v3
from utils.misc import print_dict

max_items = 200
size = 288, 512

# Load the data.
(frames, boxes, classes, n_annotations), n_items = youtube_vis.load_data(
    path.join("data", "youtube_vis"), "auto", size=size
)
labels = tf.data.Dataset.zip((boxes, classes, n_annotations))
data = tf.data.Dataset.zip((frames, labels))
n_steps = min(n_items, max_items)

# Load the model.
model = yolo_v3(size, darknet_weights=path.join("weights", "yolo_v3.weights"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.06)
apply_bn_gamma(model)

# Compute event metric performance.
# Lower the score threshold here because the computation of mAP already
# involves systematic variation of the threshold.
eval_path = path.join("outputs", "evaluate_youtube_vis")
rmtree(eval_path, ignore_errors=True)
results = model.evaluate_event(
    [FileDump(eval_path, score_threshold=0.1)],
    data,
    data_steps=n_steps,
    temporal_axis=1,
    verbose=True,
    static_graph=True,
)
metric_python = path.join(
    path.expanduser("~"), "miniconda3", "envs", "detection-metrics", "bin", "python"
)
print("Event metrics:")
os.makedirs(path.join(eval_path, "results"))
os.chdir("detection")
subprocess.call(
    [
        metric_python,
        "pascalvoc.py",
        "-gt",
        path.join("..", eval_path, "true"),
        "-det",
        path.join("..", eval_path, "pred"),
        "-sp",
        path.join("..", eval_path, "results"),
        "-np",
        "--threshold",
        "0.5",
    ]
)
os.chdir("..")

# Count conventional operations.
ops = model.count_ops_conventional(
    frames, data_steps=1, temporal_axis=1, verbose=True, static_graph=True
)
print("Conventional ops:")
print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
with open(path.join("operations", "youtube_vis_auto_yolo_conventional.p"), "wb") as f:
    pickle.dump(ops, f)

# Count event operations.
ops = model.count_ops_event(
    frames, data_steps=n_steps, temporal_axis=1, verbose=True, static_graph=True
)
print("Event ops:")
print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
with open(path.join("operations", "youtube_vis_auto_yolo_event.p"), "wb") as f:
    pickle.dump(ops, f)
