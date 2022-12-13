import os
import os.path as path
import shutil
import subprocess
from random import Random

import numpy as np
import tensorflow as tf

from eventnn.utils import filter_wrapped_ops


def centered_average(data, size):
    output = np.zeros_like(data)
    w = (size - 1) // 2
    n = len(data)
    for i in range(n):
        j_1 = 0 if (i < w) else (i - w)
        j_2 = n if (i + w + 1 > n) else (i + w + 1)
        output[i] = np.mean(data[j_1:j_2])
    return output


def compute_padded_coordinates(x, y, size_before, size_after):
    if size_before[0] / size_before[1] > size_after[0] / size_after[1]:
        # Image is too tall
        scale = size_after[0] / size_before[0]
        pad_h = 0
        pad_w = (size_after[1] - scale * size_before[1]) / 2
    else:
        # Image is too wide
        scale = size_after[1] / size_before[1]
        pad_h = (size_after[0] - scale * size_before[0]) / 2
        pad_w = 0
    return (x * scale + pad_w), (y * scale + pad_h)


def conv_ops_by_layer(ops, model):
    ops = filter_wrapped_ops(ops, model, filter_types=["Conv2D"])
    ops = [item["math_ops"] for item in ops.values()]
    ops = np.array(list(filter(lambda x: np.any(x), ops)))
    return ops[:, 0]


def find_unused_dirname(base):
    tmp_dir = base
    i = 0
    while path.exists(tmp_dir):
        tmp_dir = "{}_{}".format(base, i)
        i += 1
    return tmp_dir


def listdir(directory, join_parent=True, include_hidden=True):
    files = []
    for file in os.listdir(directory):
        if not include_hidden and file.startswith("."):
            continue
        files.append(path.join(directory, file) if join_parent else file)
    return sorted(files)


def listdir_filtered(directory, ext, join_parent=True, include_hidden=True):
    files = []
    for file in listdir(directory, join_parent=join_parent, include_hidden=include_hidden):
        if path.splitext(file)[-1] != ext:
            continue
        files.append(file)
    return sorted(files)


def prepare_padded_video_data(frame_filenames, size, preprocess_func=None):
    def _read_frame(filename):
        image = read_image(filename)
        image = tf.image.resize_with_pad(image, size[0], size[1])
        if preprocess_func is not None:
            image = preprocess_func(image)
        return image

    frame_filenames = tf.data.Dataset.from_tensors(tf.ragged.constant(frame_filenames))
    frame_filenames = frame_filenames.unbatch()
    frame_spec = tf.TensorSpec(size + (3,), dtype=tf.float32)
    frame_filenames = frame_filenames.map(
        lambda x: tf.map_fn(_read_frame, x, fn_output_signature=frame_spec), num_parallel_calls=4
    )
    return frame_filenames.batch(1, drop_remainder=True).prefetch(1)


def prepare_ragged_data(data, inner_shape=None):
    data = tf.data.Dataset.from_tensors(tf.ragged.constant(data, inner_shape=inner_shape))
    return data.unbatch().batch(1, drop_remainder=True).prefetch(1)


def print_dict(ops):
    label_length = max(len(key) for key in ops)
    for key in ops:
        print("{}  {:.4g}".format(key.ljust(label_length), ops[key]), flush=True)


def read_image(filename):
    return visual_as_float(tf.image.decode_image(tf.io.read_file(filename)))


def read_video(dirname, image_ext=".jpg", start=1):
    frames = []
    i = start
    while True:
        filename = path.join(dirname, "{}{}".format(i, image_ext))
        if not path.isfile(filename):
            break
        frames.append(read_image(filename))
        i += 1
    return tf.stack(frames)


def replace_ext(filename, ext):
    return "{}{}".format(path.splitext(filename)[0], ext)


def rescale_image(image, scale, **kwargs):
    new_size = (int(scale * image.shape[0]), int(scale * image.shape[1]))
    return tf.image.resize(image, new_size, **kwargs)


def rescale_video(video, scale, **kwargs):
    return tf.stack([rescale_image(frame, scale, **kwargs) for frame in video])


def resize_image(image, size, **kwargs):
    return tf.image.resize(image, size, **kwargs)


def resize_image_to_multiple(image, multiple):
    pad_h = -image.shape[0] % multiple
    pad_w = -image.shape[1] % multiple
    image = resize_image(image, (image.shape[0] + pad_h, image.shape[1] + pad_w))
    return image


def resize_video(video, size, **kwargs):
    return tf.stack([resize_image(frame, size, **kwargs) for frame in video])


def save_image(image, filename, **kwargs):
    image = visual_as_uint8(image)
    ext = path.splitext(filename)[1]
    if ext == ".jpg" or ext == ".jpeg":
        encoded = tf.io.encode_jpeg(image, **kwargs)
    elif ext == ".png":
        encoded = tf.io.encode_png(image, **kwargs)
    else:
        raise ValueError("Unsupported image extension {}.".format(ext))
    tf.io.write_file(filename, encoded)


def save_video(video, filename, keep_images=False, resize_method="bilinear", **kwargs):
    """
    Requires that FFmpeg be available through the command line.
    """
    image_dir = find_unused_dirname(path.splitext(filename)[0])
    os.makedirs(image_dir)
    try:
        h, w = video.shape[1:3]
        if h % 2 or w % 2:
            video = resize_video(video, (h + h % 2, w + w % 2), method=resize_method)
        for t, frame in enumerate(video):
            save_image(frame, path.join(image_dir, "{}.jpg".format(t)), quality=100, **kwargs)
        subprocess.call(["ffmpeg", "-y", "-i", path.join(image_dir, "%d.jpg"), filename])
    finally:
        if not keep_images:
            shutil.rmtree(image_dir)


def scale_coordinates(items, scale):
    return tuple(int(scale * a) for a in items)


def shuffle(dataset, seed=42):
    dataset = dataset.copy()
    Random(seed).shuffle(dataset)
    return dataset


def split(dataset, split_fractions):
    splits = []
    i = 0
    for fraction in split_fractions:
        n_items = int(fraction * len(dataset))
        splits.append(dataset[i : i + n_items])
        i += n_items
    splits.append(dataset[i:])
    return splits


def trailing_average(data, size):
    output = np.zeros_like(data)
    n = len(data)
    for i in range(n):
        j = 0 if (i < size) else (i - size)
        output[i] = np.mean(data[j : i + 1])
    return output


def visual_as_float(data):
    return tf.image.convert_image_dtype(data, tf.float32)


def visual_as_uint8(data):
    return tf.image.convert_image_dtype(data, tf.uint8)
