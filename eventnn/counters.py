from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


# Doesn't inherent from tf.keras.Layer because counters don't have
# variables that need to be saved.
class Counter:
    # All implementers take a layer in the constructor (even if they
    # don't use it) to ensure a standard interface.
    # noinspection PyUnusedLocal
    def __init__(self, layer):
        self.built = False

    @abstractmethod
    def __call__(self, x, mask, incremental, sparse):
        """
        The exact behavior of ``incremental`` and ``sparse`` varies
        depending on the type of counter. For example,
        ``PointwiseCounter`` ignores both options (``incremental=True``
        and ``incremental=False`` are equivalent for pointwise layers,
        and pointwise layers don't generally have special behavior for
        sparse inputs).

        :param x: The layer input.
        :param mask: The layer input mask.
        :param incremental: If True, when recomputing a neuron, only
        count ops for those inputs with a nonzero mask value. If False,
        when recomputing a neuron, count ops for all of its inputs
        (except for any excluded by the ``sparse`` option).
        :param sparse: If True, don't count zero values in ``x`` toward
        the number of ops.
        """

    def build(self, batch_input_shape):
        self.built = True

    def get_trace_id(self):
        return str(id(self))


# noinspection PyAbstractClass
class MathCounter(Counter):
    pass


class BatchNormMathCounter(MathCounter):
    def __call__(self, x, mask, incremental, sparse):
        # At inference batch normalization can be simplified to one
        # multiplication and one addition. If x is zero (and sparse mode
        # is enabled) then the multiplication can be skipped.
        if sparse:
            return mask + tf.cast(tf.logical_and(mask != 0, x != 0), x.dtype)
        else:
            return 2 * mask


# This counter does not count operations corresponding to padded zero
# values. Because these values are known to be zero a priori, an
# efficient implementation could just skip computing them.
class LinearMathCounter(MathCounter):
    def __init__(self, layer):
        super().__init__(layer)
        if hasattr(layer, "use_bias") and layer.use_bias:
            raise ValueError(
                "Using a layer containing a bias with "
                "LinearMathCounter will lead to incorrect results (a "
                "layer with a bias is technically affine, not linear)."
            )
        self.transform = _non_trainable_copy(layer)

    def __call__(self, x, mask, incremental, sparse):
        if sparse:
            mask = tf.cast(tf.logical_and(mask != 0, x != 0), x.dtype)
        if incremental:
            return self.transform(mask)
        else:
            updated = self.transform(mask) != 0
            if sparse:
                return tf.where(
                    updated,
                    self.transform(tf.cast(x != 0, x.dtype)),
                    tf.zeros(updated.shape, dtype=x.dtype),
                )
            else:
                return tf.where(
                    updated,
                    self.transform(tf.ones_like(x)),
                    tf.zeros(updated.shape, dtype=x.dtype),
                )

    def build(self, batch_input_shape):
        self.transform.build(batch_input_shape)
        if hasattr(self.transform, "kernel"):
            kernel = self.transform.kernel
            kernel.assign(tf.ones_like(kernel))
        super().build(batch_input_shape)


# Unlike LinearMathCounter, this counter does consider padded zero
# values. Because these zero values could be the maximum within their
# window, they cannot be ignored. Technically, if multiple padded values
# were in a single window, then only one of  them would need to be
# considered. However, we consider this an edge case optimization and do
# not use it here.
class MaxPoolingMathCounter(MathCounter):
    def __init__(self, layer):
        super().__init__(layer)
        self.transform = _non_trainable_copy(layer)
        self.fan_in = np.prod(layer.pool_size)

    def __call__(self, x, mask, incremental, sparse):
        updated = self.transform(mask) != 0
        return tf.where(
            updated,
            tf.fill(updated.shape, tf.constant(self.fan_in, dtype=x.dtype)),
            tf.zeros(updated.shape, dtype=x.dtype),
        )

    def build(self, batch_input_shape):
        self.transform.build(batch_input_shape)
        super().build(batch_input_shape)


class PointwiseMathCounter(MathCounter):
    def __call__(self, x, mask, incremental, sparse):
        return mask


# noinspection PyAbstractClass
class ReadCounter(Counter):
    pass


class BatchNormReadCounter(ReadCounter):
    def __init__(self, layer):
        super().__init__(layer)
        self.axis = layer.axis

    def __call__(self, x, mask, incremental, sparse):
        # At inference batch normalization can be simplified to one
        # multiplication and one addition.
        n_dim = len(x.shape)
        if self.axis < 0:
            axis = n_dim + self.axis
        else:
            axis = self.axis
        mask = mask != 0
        if sparse:
            mask = tf.logical_and(mask, x != 0)
        reduce_axes = list(range(1, axis)) + list(range(axis + 1, n_dim))
        axis_updated = tf.reduce_any(mask, axis=reduce_axes)
        return 2 * tf.reduce_sum(tf.cast(axis_updated, x.dtype), axis=-1)


class BiasReadCounter(ReadCounter):
    def __call__(self, x, mask, incremental, sparse):
        mask = mask != 0
        if sparse:
            mask = tf.logical_and(mask, x != 0)
        channel_updated = tf.reduce_any(mask, axis=range(1, len(x.shape) - 1))
        return tf.reduce_sum(tf.cast(channel_updated, x.dtype), axis=-1)


class ConvReadCounter(ReadCounter):
    def __init__(self, layer):
        super().__init__(layer)
        self.transform = _non_trainable_copy(layer)

    def __call__(self, x, mask, incremental, sparse):
        n_dim = len(x.shape)
        mask = mask != 0
        if sparse:
            mask = tf.logical_and(mask, x != 0)
        if incremental:
            # We take a somewhat simplified view where an update to any
            # pixel in a channel requires reading all weights
            # originating in that channel.
            if self.transform.data_format == "channels_first":
                channel_axis = 1
            else:
                channel_axis = n_dim - 1
            reduce_axes = list(range(1, channel_axis)) + list(range(channel_axis + 1, n_dim))
            channel_updated = tf.reduce_any(mask, axis=reduce_axes)
            kernel_shape = self.transform.kernel.shape
            if isinstance(self.transform, (Conv1DTranspose, Conv2DTranspose, Conv3DTranspose)):
                kernel_input_axis = len(kernel_shape) - 1
            else:
                kernel_input_axis = len(kernel_shape) - 2
            kernel_size = tf.reduce_prod(kernel_shape) / kernel_shape[kernel_input_axis]
            kernel_size = tf.cast(kernel_size, x.dtype)
            return kernel_size * tf.reduce_sum(tf.cast(channel_updated, x.dtype), axis=-1)
        else:
            any_updated = tf.reduce_any(mask, axis=range(1, n_dim))
            kernel_size = tf.reduce_prod(self.transform.kernel.shape)
            kernel_size = tf.cast(kernel_size, x.dtype)
            return tf.where(any_updated, kernel_size, tf.constant(0, dtype=x.dtype))

    def build(self, batch_input_shape):
        self.transform.build(batch_input_shape)


class DenseReadCounter(ReadCounter):
    def __init__(self, layer):
        super().__init__(layer)
        self.units = layer.units

    def __call__(self, x, mask, incremental, sparse):
        mask = mask != 0
        if sparse:
            mask = tf.logical_and(mask, x != 0)
        if incremental:
            n_updated = tf.cast(tf.math.count_nonzero(mask, axis=range(1, len(x.shape))), x.dtype)
            return self.units * n_updated
        else:
            any_updated = tf.reduce_any(mask, axis=range(1, len(x.shape)))
            return tf.where(
                any_updated,
                tf.cast(x.shape[-1] * self.units, x.dtype),
                tf.constant(0, dtype=x.dtype),
            )


class LeakyReLUReadCounter(ReadCounter):
    def __call__(self, x, mask, incremental, sparse):
        # x * mask is negative iff x is negative and mask is positive.
        any_negative = tf.reduce_any(tf.logical_and(mask != 0, x < 0), axis=range(1, len(x.shape)))
        return tf.cast(any_negative, x.dtype)


class NoneReadCounter(ReadCounter):
    def __call__(self, x, mask, incremental, sparse):
        return tf.zeros(x.shape[0], dtype=x.dtype)


def _non_trainable_copy(layer):
    config = layer.get_config()
    config.update({"name": None, "trainable": False})
    return type(layer).from_config(config)
