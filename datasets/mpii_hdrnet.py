import os
import os.path as path

import tensorflow as tf
from tensorflow.keras.metrics import Metric

from models.hdrnet import preprocess_image
from utils.misc import listdir_filtered, prepare_padded_video_data


class PSNR(Metric):
    def __init__(self, max_val=1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_val = max_val
        self.total_psnr = tf.Variable(0.0, trainable=False, name="total_psnr", dtype=tf.float64)
        self.n_psnr = tf.Variable(0, trainable=False, name="n_psnr", dtype=tf.int64)

    def get_config(self):
        config = super().get_config()
        config.update({"max_val": self.max_val})
        return config

    def update_state(self, y_true, y_pred):
        mean_psnr = tf.reduce_mean(tf.image.psnr(y_true, y_pred, self.max_val))
        self.total_psnr.assign_add(tf.cast(mean_psnr, self.total_psnr.dtype))
        self.n_psnr.assign_add(1)

    def reset_state(self):
        self.total_psnr.assign(0.0)
        self.n_psnr.assign(0)

    def result(self):
        return self.total_psnr / tf.cast(self.n_psnr, self.total_psnr.dtype)


def load_data(data_dir, split, size=(288, 512), n_frames_filter=41, video_id_filter=None):
    if split.startswith("auto"):
        return _load_auto_data(data_dir, split, size, n_frames_filter, video_id_filter)
    elif split == "test":
        return _load_test_data(data_dir, size, n_frames_filter, video_id_filter)
    else:
        raise ValueError("Invalid split name {}.".format(split))


def _load_auto_data(data_dir, split, size, n_frames_filter, video_id_filter):
    base_dir = path.join(data_dir, split)

    frames = []
    labels = []
    for video_id in sorted(os.listdir(path.join(base_dir, "frames"))):
        frames_i = listdir_filtered(path.join(base_dir, "frames", video_id), ".png")
        if (video_id_filter is not None) and (video_id not in video_id_filter):
            continue
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        frames.append(frames_i)

        labels_i = listdir_filtered(path.join(base_dir, "labels", video_id), ".png")
        labels.append(labels_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_image)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    labels = prepare_padded_video_data(labels, size, preprocess_func=preprocess_image)

    return (frames, labels), n_items


def _load_test_data(data_dir, size, n_frames_filter, video_id_filter):
    base_dir = path.join(data_dir, "test")

    frames = []
    for video_id in sorted(os.listdir(path.join(base_dir, "frames"))):
        frames_i = listdir_filtered(path.join(base_dir, "frames", video_id), ".png")
        if (video_id_filter is not None) and (video_id not in video_id_filter):
            continue
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        frames.append(frames_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_image)

    return frames, n_items
