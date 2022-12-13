import numpy as np
import tensorflow as tf
from flow_vis import flow_uv_to_colors
from tensorflow.keras.layers import *
from tensorflow_addons.image import interpolate_bilinear
from tensorflow_addons.layers import CorrelationCost

from eventnn.layers import Accumulator, Bias, Gate, Fuse, LayerWrapper, Mask, Unmask
from eventnn.model import EventModel
from utils.misc import resize_image, resize_image_to_multiple, visual_as_float


# Resources consulted:
# https://arxiv.org/abs/1709.02371
# https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
# https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/README.md


class Warp(Layer):
    def call(self, inputs, **kwargs):
        x, flow = inputs
        x_grid, y_grid = tf.meshgrid(
            tf.range(flow.shape[2]), tf.range(flow.shape[1]), indexing="xy"
        )
        x_grid = tf.cast(x_grid, x.dtype)
        y_grid = tf.cast(y_grid, x.dtype)
        query = tf.stack([x_grid, y_grid], axis=-1)
        query = tf.reshape(query, flow.shape) + flow
        query = tf.reshape(query, (query.shape[0], -1, 2))
        warped = interpolate_bilinear(x, query, indexing="xy")
        warped = tf.reshape(warped, flow.shape[:-1] + (x.shape[-1],))
        return warped


def preprocess_image(image):
    # The model expects BGR input.
    image = visual_as_float(image)
    image = resize_image_to_multiple(image, 64)
    image = tf.reverse(image, axis=[-1])
    return image


def preprocess_video(video):
    video = tf.stack([preprocess_image(frame) for frame in video])
    video = tf.repeat(video, repeats=2, axis=0)[1:-1]
    video = tf.reshape(video, (-1, 2) + tuple(video.shape[1:]))
    return video


def pwcnet(input_size, batch_size=1, npz_weights=None):
    inputs = Input(batch_input_shape=(batch_size, 2) + input_size + (3,))
    x = inputs

    if npz_weights is not None:
        w = np.load(npz_weights)
    else:
        w = None

    # Model input
    x_1, x_2 = tf.unstack(x, axis=1)
    x_1 = Mask()(x_1)
    x_2 = Mask()(x_2)
    x_1 = Gate()(x_1)
    x_2 = Gate()(x_2)

    # Feature pyramid
    x_1_all = []
    x_2_unmasked_all = []
    for i, filters in enumerate([16, 32, 64, 96, 128, 196]):
        weight_ids = ["a", "aa", "b"] if i < 5 else ["aa", "a", "b"]
        for j, weight_id in enumerate(weight_ids):
            weight_name = "conv{}{}.0".format(i + 1, weight_id)
            strides = 2 if j == 0 else 1
            x_1 = _conv_block(x_1, w, weight_name, filters=filters, strides=strides)
            x_2 = _conv_block(
                x_2, w, weight_name, filters=filters, strides=strides, final=(i == 5) and (j == 2)
            )
        x_1_all.append(x_1)
        x_2_unmasked = Accumulator()(x_2) if (i < 5) else x_2
        x_2_unmasked = Unmask()(x_2_unmasked)
        x_2_unmasked_all.append(x_2_unmasked)

    # Low resolution flow block
    x_1 = x_1_all[-1]
    x_1_unmasked = Accumulator()(x_1)
    x_1_unmasked = Unmask()(x_1_unmasked)
    x_2_unmasked = x_2_unmasked_all[-1]
    x = CorrelationCost(
        kernel_size=1,
        max_displacement=4,
        stride_1=1,
        stride_2=1,
        pad=4,
        data_format="channels_last",
    )([x_1_unmasked, x_2_unmasked])
    x = LeakyReLU(alpha=0.1)(x)
    x = Mask()(x)
    x = Gate()(x)
    for i, filters in enumerate([128, 128, 96, 64, 32]):
        x = _skip_conv_block(x, w, "conv6_{}.0".format(i), filters=filters, strides=1)
    flow = _conv_block(x, w, "predict_flow6", filters=2, strides=1, use_relu=False)
    flow = _transposed_conv_block(flow, w, "deconv6")

    # Increasing resolution flow blocks
    for i, flow_scale in enumerate([0.625, 1.25, 2.5, 5.0]):
        x_1 = x_1_all[-2 - i]
        x_1_unmasked = Accumulator()(x_1)
        x_1_unmasked = Unmask()(x_1_unmasked)
        x_2_unmasked = x_2_unmasked_all[-2 - i]
        flow_unmask = Accumulator()(flow)
        flow_unmask = Unmask()(flow_unmask)
        warp = Warp()([x_2_unmasked, flow_unmask * flow_scale])
        corr = CorrelationCost(
            kernel_size=1,
            max_displacement=4,
            stride_1=1,
            stride_2=1,
            pad=4,
            data_format="channels_last",
        )([x_1_unmasked, warp])
        corr = LeakyReLU(alpha=0.1)(corr)
        corr = Mask()(corr)
        corr = Gate()(corr)
        x = _transposed_conv_block(x, w, "upfeat{}".format(6 - i))
        x = tuple(tf.concat(x_i, axis=-1) for x_i in zip(corr, x_1, flow, x))
        for j, filters in enumerate([128, 128, 96, 64, 32]):
            x = _skip_conv_block(x, w, "conv{}_{}.0".format(5 - i, j), filters=filters, strides=1)
        flow = _conv_block(
            x, w, "predict_flow{}".format(5 - i), filters=2, strides=1, use_relu=False
        )
        if i < 3:
            flow = _transposed_conv_block(flow, w, "deconv{}".format(5 - i))

    # DC block
    for i, (filters, dilation_rate) in enumerate(
        zip([128, 128, 128, 96, 64, 32], [1, 2, 4, 8, 16, 1])
    ):
        x = _conv_block(
            x,
            w,
            "dc_conv{}.0".format(i + 1),
            filters=filters,
            strides=1,
            dilation_rate=dilation_rate,
        )
    x = _conv_block(x, w, "dc_conv7", filters=2, strides=1, use_relu=False)
    x = tuple(tf.stack(x_i, axis=1) for x_i in zip(flow, x))
    x = LayerWrapper(Fuse(), incremental=True, buffered=False)(x)
    x = Accumulator()(x)
    x = Unmask()(x)
    output = 20.0 * x

    return EventModel(inputs=inputs, outputs=[output])


def postprocess_image(output, original_size):
    return resize_image(output, original_size)


def postprocess_video(output, original_size):
    return tf.stack([postprocess_image(frame, original_size) for frame in output])


def undo_preprocess_image(image):
    image = tf.reverse(image, axis=[-1])
    return image


def undo_preprocess_video(video):
    video = tf.concat([video[0:1, 0], video[:, 1]], axis=0)
    return tf.stack([undo_preprocess_image(frame) for frame in video])


def visualize_image(output, image=None, scale=True):
    if scale:
        output = output / np.max(np.linalg.norm(output, axis=-1, ord=2))
    colors = flow_uv_to_colors(output[..., 0], output[..., 1])
    colors = visual_as_float(colors)
    if image is None:
        return colors
    else:
        return tf.concat([visual_as_float(image), colors], axis=0)


def visualize_video(output, scale=True, video=None):
    if scale:
        output = output / np.max(np.linalg.norm(output, axis=-1, ord=2))
    if video is None:
        return tf.stack([visualize_image(output_t, scale=False) for output_t in output])
    else:
        return tf.stack(
            [
                visualize_image(output_t, image=frame, scale=False)
                for frame, output_t in zip(video, output)
            ]
        )


def _conv_block(x, w, weight_name, filters, strides, final=False, use_relu=True, dilation_rate=1):
    linear = Conv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
    )
    return _linear_block(x, w, weight_name, linear, use_relu, final)


def _linear_block(x, w, weight_name, linear, use_relu, final):
    linear_wrapper = LayerWrapper(linear, incremental=True, buffered=False)
    bias = Bias()
    bias_wrapper = LayerWrapper(bias, incremental=False, buffered=False)

    x = linear_wrapper(x)
    x = Accumulator()(x)
    x = bias_wrapper(x)
    if use_relu:
        x = LayerWrapper(LeakyReLU(alpha=0.1), incremental=False, buffered=False)(x)
    if not final:
        x = Gate()(x)

    linear_wrapper.build(linear_wrapper.input_shape)
    bias_wrapper.build(bias_wrapper.input_shape)
    if w is not None:
        # A PyTorch kernel has shape (c_out, c_in, h, w).
        w_linear = w["{}.weight".format(weight_name)]
        w_linear = w_linear.transpose([2, 3, 1, 0])
        linear.set_weights([w_linear])

        w_bias = w["{}.bias".format(weight_name)]
        w_bias = w_bias.reshape(bias.bias.shape)
        bias.set_weights([w_bias])

    return x


def _skip_conv_block(x, w, weight_name, filters, strides, use_relu=True, final=False):
    skip = x
    x = _conv_block(x, w, weight_name, filters, strides, use_relu=use_relu, final=final)
    return tuple(tf.concat(x_i, axis=-1) for x_i in zip(x, skip))


def _transposed_conv_block(x, w, weight_name, final=False):
    linear = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding="same", use_bias=False)
    return _linear_block(x, w, weight_name, linear, False, final)
