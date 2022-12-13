import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

from eventnn.layers import Accumulator, Bias, Gate, Fuse, LayerWrapper, Mask, Unmask
from eventnn.model import EventModel
from utils.misc import visual_as_float


# Resources consulted:
# https://dl.acm.org/doi/abs/10.1145/3072959.3073592
# https://github.com/mgharbi/hdrnet
# https://github.com/mgharbi/hdrnet/blob/master/hdrnet/layers.py
# https://github.com/mgharbi/hdrnet/blob/master/hdrnet/models.py
# https://groups.csail.mit.edu/graphics/hdrnet/


class PiecewiseLinearTransfer(Layer):
    def __init__(
        self,
        n_functions,
        shift_initializer="zeros",
        slope_initializer="zeros",
        shift_regularizer=None,
        slope_regularizer=None,
        shift_constraint=None,
        slope_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.shifts = None
        self.slopes = None
        self.n_functions = n_functions
        self.shift_initializer = shift_initializer
        self.slope_initializer = slope_initializer
        self.shift_regularizer = shift_regularizer
        self.slope_regularizer = slope_regularizer
        self.shift_constraint = shift_constraint
        self.slope_constraint = slope_constraint

    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        self.shifts = self.add_weight(
            name="shifts",
            shape=(len(batch_input_shape) - 2) * (1,) + (batch_input_shape[-1], self.n_functions),
            initializer=self.shift_initializer,
            regularizer=self.shift_regularizer,
            trainable=True,
            constraint=self.shift_constraint,
        )
        self.slopes = self.add_weight(
            name="slopes",
            shape=(len(batch_input_shape) - 2) * (1,) + (batch_input_shape[-1], self.n_functions),
            initializer=self.slope_initializer,
            regularizer=self.slope_regularizer,
            trainable=True,
            constraint=self.slope_constraint,
        )

    def call(self, inputs, **kwargs):
        x = inputs
        x = tf.expand_dims(x, axis=-1)
        x = self.slopes * tf.nn.relu(x - self.shifts)
        return tf.reduce_sum(x, axis=-1)


# This model only counts operations for the low-resolution coefficient
# prediction network (Sections 3.1 and 3.2 of the original paper). We do
# not count operations for the slicing layer (Section 3.3), guidance map
# auxiliary network (Section 3.4.1), or final affine transform (Section
# 3.4.2).
def hdrnet(large_size, small_size=(256, 256), batch_size=1, npz_weights=None, depth_multiplier=1):
    inputs = Input(batch_input_shape=(batch_size,) + large_size + (3,))
    x = inputs

    if npz_weights is not None:
        w = np.load(npz_weights)
    else:
        w = None

    # Model input
    x = tf.image.resize(x, small_size)
    x = Mask()(x)
    x = Gate()(x)

    # Low-level features path
    for i, filters in enumerate([8, 16, 32, 64]):
        x = _conv_block(
            x,
            w,
            "coefficients/splat/conv{}/weights".format(i + 1),
            "coefficients/splat/conv{}/biases".format(i + 1),
            filters=filters * depth_multiplier,
            kernel_size=3,
            strides=2,
        )

    # Local features path
    x_local = x
    for i in range(2):
        x_local = _conv_block(
            x_local,
            w,
            "coefficients/local/conv{}/weights".format(i + 1),
            "coefficients/local/conv{}/biases".format(i + 1),
            filters=64 * depth_multiplier,
            kernel_size=3,
            strides=1,
            use_bias=(i < 1),
            use_relu=(i < 1),
        )

    # Global features path
    x_global = x
    for i in range(2):
        x_global = _conv_block(
            x_global,
            w,
            "coefficients/global/conv{}/weights".format(i + 1),
            "coefficients/global/conv{}/biases".format(i + 1),
            filters=64 * depth_multiplier,
            kernel_size=3,
            strides=2,
        )
    x_global = tuple(Flatten()(x_i) for x_i in x_global)
    for i, units in enumerate([256, 128, 64]):
        x_global = _dense_block(
            x_global,
            w,
            "coefficients/global/fc{}/weights".format(i + 1),
            "coefficients/global/fc{}/biases".format(i + 1),
            units=units * depth_multiplier,
            use_relu=(i < 2),
        )

    # Global/local fusion
    # The paper says this is a pointwise affine mixing, but the actual
    # implementation just adds the two tensors (a special case of
    # pointwise affine mixing).
    x_global = tuple(tf.reshape(x_i, (-1, 1, 1, x_i.shape[-1])) for x_i in x_global)
    x_global = tuple(
        UpSampling2D(size=x_local[0].shape[1:3], interpolation="nearest")(x_i) for x_i in x_global
    )
    x = tuple(tf.stack(x_i, axis=1) for x_i in zip(x_local, x_global))
    x = LayerWrapper(Fuse(), incremental=True, buffered=False)(x)
    x = Accumulator()(x)
    x = LayerWrapper(ReLU(), incremental=False, buffered=False)(x)
    x = Gate()(x)

    # Bilateral grid
    x = _conv_block(
        x,
        w,
        "coefficients/prediction/conv1/weights",
        "coefficients/prediction/conv1/biases",
        filters=96,
        kernel_size=1,
        strides=1,
        use_relu=False,
        final=True,
    )
    x = Unmask()(x)
    x = tf.stack(tf.split(x, 12, axis=3), axis=4)
    x = tf.stack(tf.split(x, 4, axis=4), axis=5)

    # Full-resolution guidemap
    # These computations are not done in asynchronous space because of
    # the high resolution (implying high overhead) and because of the
    # relatively small number of operations.
    g = inputs
    g = Mask(conventional_only=True)(g)
    g = _conv_block(
        g,
        w,
        "guide/ccm",
        "guide/ccm_bias",
        filters=3,
        kernel_size=1,
        strides=1,
        conventional_only=True,
        name="guidemap_1",
    )
    g = Unmask(conventional_only=True)(g)
    transfer = PiecewiseLinearTransfer(n_functions=16)
    g = transfer(g)
    if w is not None:
        transfer.set_weights([w["inference/guide/shifts"], w["inference/guide/slopes"][0]])
    g = Mask(conventional_only=True)(g)
    g = _conv_block(
        g,
        w,
        "guide/channel_mixing/weights",
        "guide/channel_mixing/biases",
        filters=1,
        kernel_size=1,
        strides=1,
        use_relu=False,
        conventional_only=True,
        name="guidemap_2",
    )
    g = Unmask(conventional_only=True)(g)
    g = tf.clip_by_value(g, 0.0, 1.0)

    # Bilateral slicing and affine image transform
    affine = _bilateral_slice(x, g)
    linear = affine[..., :3]
    bias = affine[..., 3]
    image = inputs
    linear = tf.transpose(linear, [0, 1, 2, 4, 3])
    image = tf.expand_dims(image, axis=-1)
    output = bias + tf.reduce_sum(linear * image, axis=-2)
    output = tf.clip_by_value(output, 0.0, 1.0)

    return EventModel(inputs=inputs, outputs=[output])


def preprocess_image(image):
    return visual_as_float(image)


def preprocess_video(video):
    return tf.stack([preprocess_image(frame) for frame in video])


def visualize_image(image, output):
    return tf.concat([image, output], axis=0)


def visualize_video(video, output):
    return tf.stack([visualize_image(frame, output_t) for frame, output_t in zip(video, output)])


def _bilateral_slice(x, g):
    # Spatially upsample x to have the same size as g.
    old_shape = x.shape
    new_size = tuple(tf.shape(g)[1:3])
    x = tf.reshape(x, tuple(x.shape[:3]) + (-1,))
    x = tf.image.resize(x, new_size, method="bilinear")
    x = tf.reshape(x, (old_shape[0],) + new_size + tuple(old_shape[3:]))

    # Use the guidemap to slice into the channel dimension of x.
    d = x.shape[3]
    k = tf.reshape(tf.range(d, dtype=x.dtype), (1, 1, 1, d))
    g_slice = tf.maximum(1.0 - tf.abs(d * g - k), 0)
    for _ in range(2):
        g_slice = tf.expand_dims(g_slice, axis=-1)
    affine = tf.reduce_sum(x * g_slice, axis=3)
    return affine


def _conv_block(
    x,
    w,
    linear_weight_name,
    bias_weight_name,
    filters,
    kernel_size,
    strides,
    use_bias=True,
    use_relu=True,
    final=False,
    conventional_only=False,
    name=None,
):
    linear = Conv2D(
        filters, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False
    )
    return _linear_block(
        x,
        w,
        linear_weight_name,
        bias_weight_name,
        linear,
        use_bias,
        use_relu,
        final,
        conventional_only,
        name,
    )


def _copy_bias_weights(w, bias, bias_weight_name):
    if w is not None:
        w_bias = w["inference/{}".format(bias_weight_name)]
        w_bias = w_bias.reshape(bias.bias.shape)
        bias.set_weights([w_bias])


def _copy_linear_weights(w, linear, linear_weight_name):
    if w is not None:
        w_linear = w["inference/{}".format(linear_weight_name)]
        w_linear = w_linear.reshape(linear.kernel.shape)
        linear.set_weights([w_linear])


def _dense_block(
    x,
    w,
    linear_weight_name,
    bias_weight_name,
    units,
    use_bias=True,
    use_relu=True,
    final=False,
    conventional_only=False,
    name=None,
):
    linear = Dense(units, use_bias=False)
    return _linear_block(
        x,
        w,
        linear_weight_name,
        bias_weight_name,
        linear,
        use_bias,
        use_relu,
        final,
        conventional_only,
        name,
    )


def _linear_block(
    x,
    w,
    linear_weight_name,
    bias_weight_name,
    linear,
    use_bias,
    use_relu,
    final,
    conventional_only,
    name,
):
    linear_wrapper = LayerWrapper(
        linear,
        incremental=True,
        buffered=False,
        conventional_only=conventional_only,
        name=None if (name is None) else (name + "_linear"),
    )
    bias = Bias()
    bias_wrapper = LayerWrapper(
        bias,
        incremental=False,
        buffered=False,
        conventional_only=conventional_only,
        name=None if (name is None) else (name + "_bias"),
    )

    x = linear_wrapper(x)
    if final or use_bias or use_relu:
        x = Accumulator(conventional_only=conventional_only)(x)
    if use_bias:
        x = bias_wrapper(x)
    if use_relu:
        x = LayerWrapper(
            ReLU(),
            incremental=False,
            buffered=False,
            conventional_only=conventional_only,
            name=None if (name is None) else (name + "_relu"),
        )(x)
    if not final and (use_bias or use_relu):
        x = Gate(conventional_only=conventional_only)(x)

    linear_wrapper.build(linear_wrapper.input_shape)
    _copy_linear_weights(w, linear, linear_weight_name)
    if use_bias:
        bias_wrapper.build(bias_wrapper.input_shape)
        _copy_bias_weights(w, bias, bias_weight_name)

    return x
