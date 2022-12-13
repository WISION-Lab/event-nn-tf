#!/usr/bin/env python3

import os.path as path
import pickle

import tensorflow as tf

from datasets import jhmdb
from eventnn.policies import Threshold
from eventnn.schedules import Constant
from metrics.pose import PCKSinglePerson
from models.openpose import openpose_mpii, postprocess_pck_jhmdb_mpii, preprocess_image

size = 240, 320

# Load the data.
(frames, joints), n_items = jhmdb.load_data(
    path.join("data", "jhmdb"), "test", size=size, preprocess_func=preprocess_image
)
labels = tf.data.Dataset.zip((frames, joints))
data = tf.data.Dataset.zip((frames, labels))

# Load the model.
model = openpose_mpii(size, npz_weights=path.join("weights", "openpose_mpii.npz"))
for layer in model.gates:
    layer.policy = Threshold(Constant(), warmup=True)
    layer.policy.schedule.scale.assign(0.05)

# Compute event metric performance.
results = model.evaluate_event(
    [PCKSinglePerson(postprocess_func=postprocess_pck_jhmdb_mpii)],
    data,
    data_steps=n_items,
    temporal_axis=1,
    max_time=40,
    verbose=True,
    static_graph=True,
)
with open(path.join("outputs", "jhmdb_openpose_time_event.p"), "wb") as f:
    pickle.dump(results, f)
