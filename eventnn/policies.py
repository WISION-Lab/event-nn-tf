from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import Layer


# Inherits from tf.keras.Layer to allow parameter saving and loading.
class Policy(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.overhead_math_ops = None
        self.overhead_read_ops = None
        self.overhead_write_ops = None
        self._zero_ops = None
        self._count_ops = tf.Variable(False, trainable=False, name="count_ops")

    def build(self, batch_input_shape):
        self._zero_ops = tf.zeros(batch_input_shape[0], dtype=self.dtype)
        self.overhead_math_ops = tf.Variable(
            self._zero_ops, trainable=False, name="overhead_math_ops"
        )
        self.overhead_read_ops = tf.Variable(
            self._zero_ops, trainable=False, name="overhead_read_ops"
        )
        self.overhead_write_ops = tf.Variable(
            self._zero_ops, trainable=False, name="overhead_write_ops"
        )
        super().build(batch_input_shape)

    @property
    def count_ops(self):
        return self._count_ops.value()

    @count_ops.setter
    def count_ops(self, value):
        self._count_ops.assign(value)

    def get_trace_id(self):
        return str(id(self))

    def reset(self):
        if self.built:
            self.reset_counters()

    def reset_counters(self):
        self.overhead_math_ops.assign(self._zero_ops)
        self.overhead_read_ops.assign(self._zero_ops)
        self.overhead_write_ops.assign(self._zero_ops)

    # Policies may modify the overhead counters of the model when
    # self.count_ops is enabled.
    @abstractmethod
    def step(self, delta, mask_in):
        pass


class Threshold(Policy):
    def __init__(
        self,
        schedule,
        warmup=False,
        chunk_shape=None,
        chunk_channels=False,
        data_format="channels_last",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schedule = schedule
        self.warmup = warmup
        self.chunk_shape = chunk_shape
        self.chunk_channels = chunk_channels
        self.data_format = data_format
        self.adjustments = None
        self.first = tf.Variable(True, trainable=False, name="first")

    def build(self, batch_input_shape):
        self.adjustments = tf.Variable(
            tf.ones(batch_input_shape[1:], dtype=self.dtype), trainable=False, name="adjustments"
        )
        self.schedule.build(batch_input_shape)
        super().build(batch_input_shape)

    def get_trace_id(self):
        return "{}_({},{},{},{})".format(
            super().get_trace_id(),
            self.schedule.get_trace_id(),
            str(id(self.chunk_shape)),
            str(id(self.chunk_channels)),
            str(id(self.data_format)),
        )

    def reset(self):
        super().reset()
        self.first.assign(True)
        self.schedule.reset()

    def step(self, delta, mask_in):
        if self.warmup and self.first:
            mask_out = tf.ones(delta.shape, dtype=tf.bool)

        else:
            abs_delta = tf.abs(delta)
            if self.chunk_shape is not None:
                if self.data_format != "channels_last":
                    # Implement other data formats as needed.
                    raise ValueError('data_format "{}" not supported.'.format(self.data_format))
                shape = abs_delta.shape
                size = shape[1:-1]
                pad = [[0, 0]]
                for size_i, chunk_i in zip(size, self.chunk_shape):
                    pad.append([0, chunk_i - 1 - (size_i - 1) % chunk_i])
                pad.append([0, 0])
                abs_delta = tf.pad(abs_delta, pad, mode="SYMMETRIC")
                abs_delta = tf.nn.avg_pool(
                    abs_delta, ksize=self.chunk_shape, strides=self.chunk_shape, padding="VALID"
                )
                for i, chunk_i in enumerate(self.chunk_shape):
                    abs_delta = tf.repeat(abs_delta, chunk_i, axis=i + 1)
                abs_delta = tf.slice(abs_delta, tf.zeros_like(shape), shape)
                if self.chunk_channels:
                    abs_delta = tf.reduce_mean(abs_delta, axis=-1, keepdims=True)
                    abs_delta = tf.repeat(abs_delta, shape[-1], axis=-1)
            threshold = self.schedule.step() * self.adjustments
            mask_out = abs_delta > threshold

        if self.count_ops:
            n_dim = len(mask_in.shape)
            n_update = tf.cast(tf.math.count_nonzero(mask_in, axis=range(1, n_dim)), self.dtype)
            if self.chunk_shape is not None:
                # Compute |d_t| - |d_t-1| and MAC into local mean.
                self.overhead_math_ops.assign_add(2 * n_update)
                self.overhead_read_ops.assign_add(n_update)
                self.overhead_write_ops.assign_add(n_update)

                # Reset local mean to -h.
                transmit_chunks = tf.nn.max_pool(
                    tf.cast(mask_out, tf.uint8),
                    ksize=self.chunk_shape,
                    strides=self.chunk_shape,
                    padding="SAME",
                )
                if self.chunk_channels:
                    transmit_chunks = tf.reduce_max(transmit_chunks, axis=-1, keepdims=True)
                n_transmit_chunks = tf.cast(
                    tf.math.count_nonzero(transmit_chunks, axis=range(1, n_dim)), self.dtype
                )
                self.overhead_write_ops.assign_add(n_transmit_chunks)
            else:
                # Compute |d| - h.
                self.overhead_math_ops.assign_add(n_update)

        self.first.assign(False)
        return mask_out
