import os
import os.path as path

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric

from models.pwcnet import preprocess_image
from utils.misc import listdir_filtered, read_image, resize_image

# Resources consulted on Sintel dataset:
# https://link.springer.com/chapter/10.1007/978-3-642-33783-3_44

# Resources consulted on EPE evaluation (including matched and unmatched
# EPE metrics):
# https://link.springer.com/content/pdf/10.1007/s11263-010-0390-2.pdf
# https://link.springer.com/chapter/10.1007/978-3-642-33783-3_44

MAJOR_CAMERA_MOTION_TEST = [
    "PERTURBED_market_3",
    "PERTURBED_shaman_1",
    "ambush_1",
    "ambush_3",
    "bamboo_3",
    "cave_3",
    "market_1",
    "market_4",
    "mountain_2",
    "temple_1",
    "tiger",
    "wall",
]

MINOR_CAMERA_MOTION_TEST = []

NO_CAMERA_MOTION_TEST = []


class EPE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_epe = tf.Variable(0.0, trainable=False, name="total_epe", dtype=tf.float64)
        self.n_epe = tf.Variable(0, trainable=False, name="n_epe", dtype=tf.int64)

    def update_state(self, y_true, y_pred):
        y_pred = resize_image(y_pred, y_true.shape[1:3])
        mean_epe = tf.reduce_mean(tf.norm(y_true - y_pred, axis=-1))
        self.total_epe.assign_add(tf.cast(mean_epe, self.total_epe.dtype))
        self.n_epe.assign_add(1)

    def reset_state(self):
        self.total_epe.assign(0.0)
        self.n_epe.assign(0)

    def result(self):
        return self.total_epe / tf.cast(self.n_epe, self.total_epe.dtype)


# Note that the resizing in this function does *not* preserve aspect
# ratio. An artificial black boundary (created by padding to the correct
# aspect ratio) can confuse motion estimation algorithms.
def load_data(
    data_dir,
    split,
    size=(256, 512),
    n_frames_filter=None,
    video_id_filter=None,
    input_size=(436, 1024),
):
    if split.startswith("auto"):
        return _load_auto_dataset(
            data_dir, split, size, n_frames_filter, video_id_filter, input_size
        )
    if split == "test":
        return _load_test_dataset(data_dir, size, n_frames_filter, video_id_filter, input_size)
    elif split == "train":
        return _load_train_dataset(data_dir, size, n_frames_filter, video_id_filter, input_size)
    else:
        raise ValueError("Invalid split name {}.".format(split))


def read_flo(filename):
    with open(filename, "rb") as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        if tag != _FLOW_TAG:
            raise IOError("Invalid tag in flow file {}.".format(filename))
        w = np.fromfile(f, dtype=np.int32, count=1)[0]
        h = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32, count=w * h * 2)
        if len(f.read()) > 0:
            raise IOError("Unexpected extra bytes in flow file {}.".format(filename))
    data = data.reshape(h, w, 2)
    return tf.convert_to_tensor(data)


# Tag that should be present at the beginning of a .flo file
_FLOW_TAG = 202021.25


def _load_test_dataset(data_dir, size, n_frames_filter, video_id_filter, input_size):
    base_dir = path.join(data_dir, "test")
    frames = []
    for video_id in sorted(os.listdir(path.join(base_dir, "frames"))):
        frames_i = listdir_filtered(path.join(base_dir, "frames", video_id), ".png")
        if (video_id_filter is not None) and (video_id not in video_id_filter):
            continue
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        frames.append(list(zip(frames_i[:-1], frames_i[1:])))

    n_items = len(frames)

    # Dataset shape (1, n_frames, 2, size[0], size[1], 3)
    frames = _prepare_frames(frames, size, input_size)

    return frames, n_items


# noinspection DuplicatedCode
def _load_auto_dataset(data_dir, split, size, n_frames_filter, video_id_filter, input_size):
    base_dir = path.join(data_dir, split)
    frames = []
    flow = []
    for video_id in sorted(os.listdir(path.join(base_dir, "frames"))):
        frames_i = listdir_filtered(path.join(base_dir, "frames", video_id), ".png")
        if (video_id_filter is not None) and (video_id not in video_id_filter):
            continue
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        frames.append(list(zip(frames_i[:-1], frames_i[1:])))

        flow_i = listdir_filtered(path.join(base_dir, "flow", video_id), ".proto")
        flow.append(flow_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, 2, size[0], size[1], 3)
    frames = _prepare_frames(frames, size, input_size)

    # Dataset shape (1, n_frames, size[0], size[1], 2)
    flow = _prepare_flow(flow, size, input_size)

    return (frames, flow), n_items


# noinspection DuplicatedCode
def _load_train_dataset(data_dir, size, n_frames_filter, video_id_filter, input_size):
    base_dir = path.join(data_dir, "train")
    frames = []
    flow = []
    occlusions = []
    for video_id in sorted(os.listdir(path.join(base_dir, "frames"))):
        frames_i = listdir_filtered(path.join(base_dir, "frames", video_id), ".png")
        if (video_id_filter is not None) and (video_id not in video_id_filter):
            continue
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        frames.append(list(zip(frames_i[:-1], frames_i[1:])))

        flow_i = listdir_filtered(path.join(base_dir, "flow", video_id), ".proto")
        flow.append(flow_i)

        occlusions_i = listdir_filtered(path.join(base_dir, "occlusions", video_id), ".png")
        occlusions.append(occlusions_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, 2, size[0], size[1], 3)
    frames = _prepare_frames(frames, size, input_size)

    # Dataset shape (1, n_frames, size[0], size[1], 2)
    flow = _prepare_flow(flow, size, input_size)

    # Dataset shape (1, n_frames, size[0], size[1])
    occlusions = tf.data.Dataset.from_tensors(tf.ragged.constant(occlusions))
    occlusions_spec = tf.TensorSpec(size, dtype=tf.float32)
    occlusions = _prepare_dataset(occlusions, _read_occlusions, size, occlusions_spec, input_size)

    return (frames, flow, occlusions), n_items


def _prepare_dataset(dataset, function, size, spec, input_size):
    dataset = dataset.unbatch()
    dataset = dataset.map(
        lambda x: tf.map_fn(lambda y: function(y, size, input_size), x, fn_output_signature=spec),
        num_parallel_calls=4,
    )
    dataset = dataset.batch(1, drop_remainder=True).prefetch(1)
    return dataset


def _prepare_flow(flow, size, input_size):
    flow = tf.data.Dataset.from_tensors(tf.ragged.constant(flow))
    flow_spec = tf.TensorSpec(size + (2,), dtype=tf.float32)
    flow = _prepare_dataset(flow, _read_flow, size, flow_spec, input_size)
    return flow


def _prepare_frames(frames, size, input_size):
    frames = tf.data.Dataset.from_tensors(tf.ragged.constant(frames, inner_shape=(2,)))
    frames_spec = tf.TensorSpec((2,) + size + (3,), dtype=tf.float32)
    frames = _prepare_dataset(frames, _read_frame_pair, size, frames_spec, input_size)
    return frames


def _read_flow(filename, size, input_size):
    flow_tensor = tf.io.parse_tensor(tf.io.read_file(filename), tf.float32)
    flow_tensor = tf.reshape(flow_tensor, input_size + (2,))
    flow_tensor = tf.image.resize(flow_tensor, size)
    return flow_tensor


def _read_frame(filename, size, input_size):
    image = read_image(filename)
    image = tf.reshape(image, input_size + (3,))
    image = tf.image.resize(image, size)
    image = preprocess_image(image)
    return image


def _read_frame_pair(filenames, size, input_size):
    return tf.stack(
        [_read_frame(filenames[0], size, input_size), _read_frame(filenames[1], size, input_size)],
        axis=0,
    )


def _read_occlusions(filename, size, input_size):
    occlusions_tensor = read_image(filename)
    occlusions_tensor = tf.reshape(occlusions_tensor, input_size + (1,))
    occlusions_tensor = tf.image.resize(occlusions_tensor, size)
    occlusions_tensor = tf.squeeze(occlusions_tensor, axis=-1)
    return occlusions_tensor
