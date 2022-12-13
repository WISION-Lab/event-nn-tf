#!/usr/bin/env python3

import os.path as path
import pickle

import tensorflow as tf

from datasets import jhmdb
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

# Compute event metric performance.
results = model.evaluate_conventional(
    [PCKSinglePerson(postprocess_func=postprocess_pck_jhmdb_mpii)],
    data,
    data_steps=n_items,
    temporal_axis=1,
    max_time=40,
    verbose=True,
    static_graph=True,
)
with open(path.join("outputs", "jhmdb_openpose_time_conventional.p"), "wb") as f:
    pickle.dump(results, f)
