#!/usr/bin/env python3

import os
import os.path as path
import pickle
import subprocess
from shutil import rmtree

import tensorflow as tf

from datasets import youtube_vis
from datasets.mpii import MAJOR_CAMERA_MOTION_TEST, MINOR_CAMERA_MOTION_TEST, NO_CAMERA_MOTION_TEST
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.yolo import FileDump, apply_bn_gamma, yolo_v3
from utils.misc import print_dict

size = 288, 512

# Load the model.
model = yolo_v3(size, darknet_weights=path.join("weights", "yolo_v3.weights"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.06)
apply_bn_gamma(model)

for video_id_filter, name in zip(
    [MAJOR_CAMERA_MOTION_TEST, MINOR_CAMERA_MOTION_TEST, NO_CAMERA_MOTION_TEST, None],
    ["major", "minor", "none", "all"],
):
    print("Camera motion: {}".format(name))

    # Load the data.
    (frames, boxes, classes, n_annotations), n_items = youtube_vis.load_data(
        path.join("data", "mpii"),
        "auto_yolo",
        size=size,
        n_frames_filter=41,
        video_id_filter=video_id_filter,
    )
    labels = tf.data.Dataset.zip((boxes, classes, n_annotations))
    data = tf.data.Dataset.zip((frames, labels))

    # Compute event metric performance.
    eval_path = path.join("outputs", "evaluate_youtube_vis")
    rmtree(eval_path, ignore_errors=True)
    results = model.evaluate_event(
        [FileDump(eval_path, score_threshold=0.1)],
        data,
        data_steps=n_items,
        temporal_axis=1,
        verbose=True,
        static_graph=True,
    )
    print("Event metrics:")
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
    with open(
        path.join("operations", "evaluate", "mpii_auto_yolo_conventional_{}.p".format(name)), "wb"
    ) as f:
        pickle.dump(ops, f)

    # Count event operations.
    ops = model.count_ops_event(
        frames, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=True
    )
    print("Event ops:")
    print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))
    with open(
        path.join("operations", "evaluate", "mpii_auto_yolo_event_{}.p".format(name)), "wb"
    ) as f:
        pickle.dump(ops, f)

    print()
