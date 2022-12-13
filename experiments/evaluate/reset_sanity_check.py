#!/usr/bin/env python3

import os
import os.path as path
import subprocess
from shutil import rmtree

import tensorflow as tf

from datasets import imagenet_vod
from datasets.imagenet_vod import IMAGENET_VOD_NAMES
from eventnn.policies import Threshold
from eventnn.schedules import Constant, PeriodicReset
from eventnn.utils import filter_wrapped_ops, reduce_ops_all
from models.yolo import FileDump, apply_bn_gamma, yolo_v3, preprocess_image
from utils.misc import print_dict

size = 224, 384
n_items = 10
threshold = 0.02


def evaluate(eval_function):
    # Lower the score threshold here because the computation of mAP already
    # involves systematic variation of the threshold.
    eval_path = path.join("outputs", "evaluate_reset_sanity_check")
    rmtree(eval_path, ignore_errors=True)
    eval_function(
        [FileDump(eval_path, score_threshold=0.1, class_names=IMAGENET_VOD_NAMES, verbose=False)],
        data,
        data_steps=n_items,
        temporal_axis=1,
        verbose=True,
        static_graph=True,
    )
    metric_python = path.join(
        path.expanduser("~"), "miniconda3", "envs", "detection-metrics", "bin", "python"
    )
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


# Load the data.
(frames, boxes, classes, n_annotations), _ = imagenet_vod.load_data(
    path.join("data", "imagenet_vod"), "val", size=size, preprocess_func=preprocess_image
)
labels = tf.data.Dataset.zip((boxes, classes, n_annotations))
data = tf.data.Dataset.zip((frames, labels))

# Load the model.
model = yolo_v3(size, n_classes=30, h5_weights=path.join("weights", "yolo_v3_imagenet_vod.h5"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)

# Compute conventional metric performance.
print("Conventional metrics:", flush=True)
evaluate(model.evaluate_conventional)

# Count conventional operations.
ops = model.count_ops_conventional(
    frames, data_steps=1, temporal_axis=1, verbose=True, static_graph=True
)
print("Conventional ops:")
print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))

for reset_period in 0, 1, 2, 4, 8:
    print("Reset period: {}".format(reset_period))

    # Delete the old model to prevent OOM errors.
    del model

    # Load the model.
    model = yolo_v3(size, n_classes=30, h5_weights=path.join("weights", "yolo_v3_imagenet_vod.h5"))
    for layer in model.gates:
        if reset_period > 0:
            schedule = PeriodicReset()
            schedule.period.assign(reset_period)
        else:
            schedule = Constant()
        schedule.scale.assign(threshold)
        layer.policy = Threshold(schedule, warmup=True)
        layer.memory_loss = True
    apply_bn_gamma(model)

    # Compute event metric performance.
    print("Event metrics:", flush=True)
    evaluate(model.evaluate_event)

    # Count conventional operations.
    ops = model.count_ops_event(
        frames, data_steps=n_items, temporal_axis=1, verbose=True, static_graph=True
    )
    print("Event ops:")
    print_dict(reduce_ops_all(filter_wrapped_ops(ops, model)))

    print()
