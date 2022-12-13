from copy import deepcopy

import numpy as np
import tensorflow as tf

from eventnn.layers import LayerWrapper

# Count multiplications, additions, and activation functions.
# DEFAULT_OP_FILTER = (
#     "BatchNormalization",
#     "Bias",
#     "Conv1D",
#     "Conv1DTranspose",
#     "Conv2D",
#     "Conv2DTranspose",
#     "Conv3D",
#     "Conv3DTranspose",
#     "Dense",
#     "Fuse",
#     "LeakyReLU",
#     "ReLU",
# )

# Count MAC operations.
DEFAULT_OP_FILTER = (
    "Conv1D",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DTranspose",
    "Dense",
)


def extract_ops_item(ops, i):
    output = {}
    for layer_name, ops_j in ops.items():
        output[layer_name] = {}
        for key in ops_j:
            output[layer_name][key] = [ops_j[key][i]]
    return output


def filter_wrapped_ops(ops, model, filter_types=DEFAULT_OP_FILTER):
    output = deepcopy(ops)
    for ops_i, layer_i in zip(output.values(), model.event_layers):
        is_layer_wrapper = isinstance(layer_i, LayerWrapper)
        if is_layer_wrapper:
            if _typename(layer_i.layer) not in filter_types:
                for key in "math_ops", "read_ops":
                    ops_i[key] = _multi_array_zeros(ops_i[key])
    return output


def reduce_ops_all(ops, weighted=True):
    reduced = reduce_ops_layers(ops)
    output = {}
    for key in reduced:
        output[key] = _multi_array_mean(reduced[key], weighted)
    return output


def reduce_ops_items_time(ops, weighted=True):
    output = {}
    for layer_name, ops_i in ops.items():
        output[layer_name] = {}
        for key in ops_i:
            output[layer_name][key] = _multi_array_mean(ops_i[key], weighted)
    return output


def reduce_ops_layers(ops):
    output = {}
    for ops_i in ops.values():
        if len(output) == 0:
            for key in ops_i:
                output[key] = deepcopy(ops_i[key])
        else:
            for key in ops_i:
                for k in range(len(ops_i[key])):
                    output[key][k] += ops_i[key][k]
    return output


def reduce_ops_layers_time(ops):
    reduced = reduce_ops_layers(ops)
    output = {}
    for key in reduced:
        output[key] = _temporal_mean(reduced[key])
    return output


def reduce_ops_time(ops):
    output = {}
    for layer_name, ops_i in ops.items():
        output[layer_name] = {}
        for key in ops_i:
            output[layer_name][key] = _temporal_mean(ops_i[key])
    return output


def _array_iter(data, batch_size, dtype, n_total, n_batches):
    if data is None:
        return None
    else:
        data = np.array(data).astype(dtype.as_numpy_dtype())
        padded = np.empty((n_batches * batch_size,) + data.shape[1:], dtype=data.dtype)
        padded[:n_total] = data
        padded = padded.reshape((n_batches, batch_size) + padded.shape[1:])
        return iter(tf.convert_to_tensor(padded))


def _data_generator(x, y, data_steps, batch_size, dtype):
    if isinstance(x, tf.data.Dataset):
        x_iter = iter(x)
        for i in range(data_steps):
            batch = next(x_iter)
            yield batch, _nested_tensor_batch_size(batch)

    elif isinstance(x, (np.ndarray, tf.Tensor)):
        n_total = _n_total(x, data_steps, batch_size)
        n_batches = (n_total - 1) // batch_size + 1
        n_last = n_total - (n_batches - 1) * batch_size
        x_iter = _array_iter(x, batch_size, dtype, n_total, n_batches)
        y_iter = _array_iter(y, batch_size, dtype, n_total, n_batches)
        for i in range(n_batches):
            batch = next(x_iter) if (y is None) else (next(x_iter), next(y_iter))
            n_items = batch_size if (i < n_batches - 1) else n_last
            yield batch, n_items

    else:
        raise TypeError("Data type {} not supported.".format(type(x)))


def _append_or_add(list_, i, value):
    if len(list_) - 1 >= i:
        list_[i] += value
    else:
        list_.append(np.copy(value))


def _multi_array_mean(arrays, weighted):
    means = np.array(_temporal_mean(arrays))
    if weighted:
        weights = np.array([len(x) for x in arrays])
        weights = weights / np.sum(weights)
        return np.sum(weights * means)
    else:
        return np.mean(means)


def _multi_array_zeros(arrays):
    return [np.zeros_like(x) for x in arrays]


def _n_total(data, data_steps, batch_size):
    if isinstance(data, tf.data.Dataset):
        return data_steps * batch_size
    elif isinstance(data, (np.ndarray, tf.Tensor)):
        return data.shape[0]
    else:
        raise TypeError("Data type {} not supported.".format(type(data)))


def _nested_tensor_batch_size(item):
    while not isinstance(item, tf.Tensor):
        item = item[0]
    return item.shape[0]


def _temporal_mean(arrays):
    return [np.mean(x) for x in arrays]


def _typename(x):
    return type(x).__name__
