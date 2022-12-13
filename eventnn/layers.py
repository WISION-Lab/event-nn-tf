from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import *

from eventnn.counters import (
    BatchNormMathCounter,
    BatchNormReadCounter,
    BiasReadCounter,
    ConvReadCounter,
    DenseReadCounter,
    LeakyReLUReadCounter,
    LinearMathCounter,
    MaxPoolingMathCounter,
    NoneReadCounter,
    PointwiseMathCounter,
)


class Bias(Layer):
    def __init__(self, initializer="zeros", regularizer=None, constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.bias = None
        self.initializer = initializer
        self.regularizer = regularizer
        self.constraint = constraint

    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        self.bias = self.add_weight(
            name="bias",
            shape=(len(batch_input_shape) - 2) * (1,) + (batch_input_shape[-1],),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            constraint=self.constraint,
        )

    def call(self, inputs, **kwargs):
        return inputs + self.bias

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape


class Fuse(Layer):
    def call(self, inputs, **kwargs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, batch_input_shape):
        return (batch_input_shape[0],) + batch_input_shape[2:]


class EventLayer(Layer):
    def __init__(self, conventional_only=False, **kwargs):
        super().__init__(**kwargs)
        self.has_null_values = False
        self.math_ops = None
        self.read_ops = None
        self.overhead_math_ops = None
        self.overhead_read_ops = None
        self.overhead_write_ops = None
        self._zero_ops = None
        self._conventional_only = tf.Variable(
            conventional_only, trainable=False, name="conventional_only"
        )
        self._event_mode = tf.Variable(False, trainable=False, name="event_mode")
        self._count_ops = tf.Variable(False, trainable=False, name="count_ops")
        self._count_sparse = tf.Variable(False, trainable=False, name="count_sparse")

    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        if None in self.compute_single_input_shape(batch_input_shape):
            raise ValueError(
                "The input size (including batch size) must be fixed for an EventLayer."
            )
        self._zero_ops = tf.zeros(
            self.compute_single_output_shape(batch_input_shape)[0], dtype=self.dtype
        )
        self.math_ops = tf.Variable(self._zero_ops, trainable=False, name="math_ops")
        self.read_ops = tf.Variable(self._zero_ops, trainable=False, name="read_ops")
        self.overhead_math_ops = tf.Variable(
            self._zero_ops, trainable=False, name="overhead_math_ops"
        )
        self.overhead_read_ops = tf.Variable(
            self._zero_ops, trainable=False, name="overhead_read_ops"
        )
        self.overhead_write_ops = tf.Variable(
            self._zero_ops, trainable=False, name="overhead_write_ops"
        )

    def call(self, inputs, **kwargs):
        if self.event_mode and not self.conventional_only:
            return self.call_event(inputs)
        else:
            return self.call_conventional(inputs, **kwargs)

    @abstractmethod
    def call_conventional(self, inputs, **kwargs):
        pass

    @abstractmethod
    def call_event(self, inputs):
        pass

    @abstractmethod
    def compute_output_shape(self, batch_input_shape):
        pass

    @abstractmethod
    def compute_single_input_shape(self, batch_input_shape):
        pass

    @abstractmethod
    def compute_single_output_shape(self, batch_input_shape):
        pass

    @property
    def conventional_only(self):
        return self._conventional_only.value()

    @conventional_only.setter
    def conventional_only(self, value):
        self._conventional_only.assign(value)

    @property
    def count_ops(self):
        return self._count_ops.value()

    @count_ops.setter
    def count_ops(self, value):
        self._count_ops.assign(value)

    @property
    def count_sparse(self):
        return self._count_sparse.value()

    @count_sparse.setter
    def count_sparse(self, value):
        self._count_sparse.assign(value)

    @property
    def event_mode(self):
        return self._event_mode.value()

    @event_mode.setter
    def event_mode(self, value):
        self._event_mode.assign(value)

    def get_trace_id(self):
        return str(id(self))

    def reset(self):
        if not self.built:
            self.build(self.input_shape)
        self.reset_counters()

    def reset_counters(self):
        self.math_ops.assign(self._zero_ops)
        self.read_ops.assign(self._zero_ops)
        self.overhead_math_ops.assign(self._zero_ops)
        self.overhead_read_ops.assign(self._zero_ops)
        self.overhead_write_ops.assign(self._zero_ops)


class Accumulator(EventLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_null_values = True
        self.null_output = None
        self.accumulator = None

    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        shape = self.compute_single_input_shape(batch_input_shape)
        self.null_output = tf.Variable(
            tf.zeros(shape, dtype=self.dtype), trainable=False, name="null_output"
        )
        self.accumulator = tf.Variable(
            tf.zeros(shape, dtype=self.dtype), trainable=False, name="accumulator"
        )

    def call_conventional(self, inputs, **kwargs):
        x, mask = inputs
        return x, tf.ones_like(x)

    def call_event(self, inputs):
        x, mask = inputs
        self.accumulator.assign_add(tf.where(mask != 0, x, tf.zeros_like(x)))
        if self.count_ops:
            n_updated = tf.reduce_sum(mask, axis=range(1, len(mask.shape)))
            self.overhead_math_ops.assign_add(n_updated)
            self.overhead_read_ops.assign_add(n_updated)
            self.overhead_write_ops.assign_add(n_updated)
        return self.accumulator.value(), mask

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

    def compute_single_input_shape(self, batch_input_shape):
        return batch_input_shape[0]

    def compute_single_output_shape(self, batch_input_shape):
        return batch_input_shape[0]

    def reset(self):
        super().reset()
        self.accumulator.assign(self.null_output)

    def update_null_values(self, inputs):
        self.null_output.assign(inputs[0])


class Gate(EventLayer):
    def __init__(self, policy=None, delta_based=True, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy
        self.delta_based = delta_based
        self.has_null_values = True
        self.null_output = None
        self.best = None
        self.delta = None
        self._output_zeros = None
        self._all_outputs = tf.Variable(False, trainable=False, name="all_outputs")
        self._memory_loss = tf.Variable(False, trainable=False, name="memory_loss")

    @property
    def all_outputs(self):
        return self._all_outputs.value()

    @all_outputs.setter
    def all_outputs(self, value):
        self._all_outputs.assign(value)

    @property
    def memory_loss(self):
        return self._memory_loss.value()

    @memory_loss.setter
    def memory_loss(self, value):
        self._memory_loss.assign(value)

    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        shape = self.compute_single_output_shape(batch_input_shape)
        self.null_output = tf.Variable(
            tf.zeros(shape, dtype=self.dtype), trainable=False, name="null_output"
        )
        if (self.policy is not None) and (not self.policy.built):
            self.policy.build(shape)
        self._output_zeros = tf.zeros(shape, dtype=self.dtype)
        self.best = tf.Variable(self._output_zeros, trainable=False, name="best")
        self.delta = tf.Variable(self._output_zeros, trainable=False, name="delta")

    def call_conventional(self, inputs, **kwargs):
        x, mask = inputs
        return x, tf.ones_like(x)

    def call_event(self, inputs):
        x, mask_in = inputs
        mask_in = mask_in != 0
        new_best = tf.where(mask_in, x, self.best)
        self.delta.assign_add(new_best - self.best)
        self.best.assign(new_best)
        if self.policy is None:
            mask_out = tf.ones(self.delta.shape, tf.bool)
        else:
            mask_out = self.policy.step(self.delta, mask_in)
        if self.count_ops:
            n_dim = len(x.shape)
            n_update = tf.cast(tf.math.count_nonzero(mask_in, axis=range(1, n_dim)), self.dtype)
            n_transmit = tf.cast(tf.math.count_nonzero(mask_out, axis=range(1, n_dim)), self.dtype)
            self.overhead_math_ops.assign_add(2 * n_update + n_transmit)
            self.overhead_read_ops.assign_add(2 * n_update + n_transmit)
            self.overhead_write_ops.assign_add(2 * n_update + n_transmit)
        if self.delta_based:
            output = self.delta.value()
        else:
            output = self.best.value()
        if not self._all_outputs:
            output = tf.where(
                mask_out, output, tf.fill(output.shape, tf.constant(float("nan"), output.dtype))
            )
        if self.memory_loss:
            self.delta.assign(tf.zeros_like(self.delta))
        else:
            self.delta.assign(tf.where(mask_out, tf.zeros_like(self.delta), self.delta))
        mask_out = tf.cast(mask_out, self.dtype)
        return output, mask_out

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

    def compute_single_input_shape(self, batch_input_shape):
        return batch_input_shape[0]

    def compute_single_output_shape(self, batch_input_shape):
        return batch_input_shape[0]

    def get_trace_id(self):
        policy_trace_id = "" if (self.policy is None) else self.policy.get_trace_id()
        return "{}_({})".format(super().get_trace_id(), policy_trace_id)

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, value):
        self._policy = value
        if (value is not None) and hasattr(self, "input_shape"):
            self._policy.build(self.compute_output_shape(self.input_shape)[0])

    def reset(self):
        super().reset()
        if self.policy is not None:
            self.policy.reset()
        self.best.assign(self.null_output)
        self.delta.assign(self._output_zeros)

    def update_null_values(self, inputs):
        self.null_output.assign(inputs[0])


class LayerWrapper(EventLayer):
    def __init__(
        self, layer, incremental, buffered, math_counter=None, read_counter=None, **kwargs
    ):
        """
        :param incremental: If True, a change to any neuron input
        requires re-accessing all other inputs for that neuron. The
        value of incremental is used in two ways. First, it is passed to
        operation counters in call_event (see
        counters.Counter.__call__). Second, if False, the layer is given
        access to all inputs (in a real implementation these inputs
        would need to be stored in some buffer). If True, the layer's
        inputs are filtered through the update mask.
        """
        super().__init__(**kwargs)
        self.buffered = buffered
        self.incremental = incremental
        self.layer = layer
        self.has_null_values = buffered
        self.null_input = None
        if math_counter is None:
            self.math_counter = _auto_counter(layer, _AUTO_MATH_COUNTERS)
        else:
            self.math_counter = math_counter
        if read_counter is None:
            self.read_counter = _auto_counter(layer, _AUTO_READ_COUNTERS)
        else:
            self.read_counter = read_counter
        self.buffer = None

    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        shape = self.compute_single_input_shape(batch_input_shape)
        if self.buffered:
            self.null_input = tf.Variable(
                tf.zeros(shape, dtype=self.dtype),
                trainable=False,
                name="null_input",
            )
            self.buffer = tf.Variable(
                tf.zeros(shape, dtype=self.dtype), trainable=False, name="buffer"
            )
        for item in self.layer, self.math_counter, self.read_counter:
            if not item.built:
                item.build(shape)

    def call_conventional(self, inputs, **kwargs):
        x, mask = inputs
        if self.count_ops:
            math_ops = self.math_counter(x, mask, False, sparse=self.count_sparse)
            read_ops = self.read_counter(x, mask, False, sparse=self.count_sparse)
            self.math_ops.assign_add(tf.reduce_sum(math_ops, axis=range(1, len(math_ops.shape))))
            self.read_ops.assign_add(read_ops)
        x = self.layer(x, **kwargs)
        return x, tf.ones_like(x)

    def call_event(self, inputs):
        x, mask = inputs
        if self.buffered:
            self.buffer.assign(tf.where(mask != 0, x, self.buffer))
            x = self.buffer.value()
            if self.count_ops:
                # There's no easy way to know which buffer values are
                # going to be needed by the layer, so just read all of
                # them. This shouldn't be a significant problem because
                # buffered=True is only used with max pooling layers and
                # models containing ValueGate.
                self.overhead_read_ops.assign_add(
                    tf.repeat(
                        tf.cast(tf.reduce_prod(self.buffer.shape[1:]), self.dtype),
                        self.buffer.shape[0],
                    )
                )
                self.overhead_write_ops.assign_add(
                    tf.reduce_sum(mask, axis=range(1, len(mask.shape)))
                )
        math_ops = self.math_counter(
            x, mask, incremental=self.incremental, sparse=self.count_sparse
        )
        if self.count_ops:
            read_ops = self.read_counter(
                x, mask, incremental=self.incremental, sparse=self.count_sparse
            )
            self.math_ops.assign_add(tf.reduce_sum(math_ops, axis=range(1, len(math_ops.shape))))
            self.read_ops.assign_add(read_ops)
        if self.incremental:
            x = self.layer(tf.where(mask != 0, x, tf.zeros_like(x)))
        else:
            # If the inputs aren't properly buffered then elements of x
            # for which mask == 0 will be NaN.
            x = self.layer(x)
        mask = tf.cast(math_ops != 0, self.dtype)
        return x, mask

    def compute_output_shape(self, batch_input_shape):
        return (self.layer.compute_output_shape(batch_input_shape[0]),) * 2

    def compute_single_input_shape(self, batch_input_shape):
        return batch_input_shape[0]

    def compute_single_output_shape(self, batch_input_shape):
        return self.layer.compute_output_shape(batch_input_shape[0])

    def get_trace_id(self):
        sub_ids = [
            str(id(self.layer)),
            self.math_counter.get_trace_id(),
            self.read_counter.get_trace_id(),
        ]
        return "{}_({})".format(super().get_trace_id(), ",".join(sub_ids))

    def reset(self):
        super().reset()
        if self.buffered:
            self.buffer.assign(self.null_input)

    def update_null_values(self, inputs):
        self.null_input.assign(inputs[0])


class Mask(EventLayer):
    def call_conventional(self, inputs, **kwargs):
        return inputs, tf.ones_like(inputs)

    def call_event(self, inputs):
        return inputs, tf.ones_like(inputs)

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape, batch_input_shape

    def compute_single_input_shape(self, batch_input_shape):
        return batch_input_shape

    def compute_single_output_shape(self, batch_input_shape):
        return batch_input_shape


class Unmask(EventLayer):
    def call_conventional(self, inputs, **kwargs):
        return inputs[0]

    def call_event(self, inputs):
        return inputs[0]

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape[0]

    def compute_single_input_shape(self, batch_input_shape):
        return batch_input_shape[0]

    def compute_single_output_shape(self, batch_input_shape):
        return batch_input_shape[0]


# When adding an entry here you may also want to modify
# DEFAULT_OP_FILTER in layers.py to include the new layer type.
# noinspection DuplicatedCode
_AUTO_MATH_COUNTERS = {
    BatchNormalization: BatchNormMathCounter,
    Bias: PointwiseMathCounter,
    Conv1D: LinearMathCounter,
    Conv1DTranspose: LinearMathCounter,
    Conv2D: LinearMathCounter,
    Conv2DTranspose: LinearMathCounter,
    Conv3D: LinearMathCounter,
    Conv3DTranspose: LinearMathCounter,
    Dense: LinearMathCounter,
    Dropout: PointwiseMathCounter,
    Fuse: LinearMathCounter,
    LeakyReLU: PointwiseMathCounter,
    MaxPooling1D: MaxPoolingMathCounter,
    MaxPooling2D: MaxPoolingMathCounter,
    MaxPooling3D: MaxPoolingMathCounter,
    ReLU: PointwiseMathCounter,
    Softmax: PointwiseMathCounter,
}

# noinspection DuplicatedCode
_AUTO_READ_COUNTERS = {
    BatchNormalization: BatchNormReadCounter,
    Bias: BiasReadCounter,
    Conv1D: ConvReadCounter,
    Conv1DTranspose: ConvReadCounter,
    Conv2D: ConvReadCounter,
    Conv2DTranspose: ConvReadCounter,
    Conv3D: ConvReadCounter,
    Conv3DTranspose: ConvReadCounter,
    Dense: DenseReadCounter,
    Dropout: NoneReadCounter,
    Fuse: NoneReadCounter,
    LeakyReLU: LeakyReLUReadCounter,
    MaxPooling1D: NoneReadCounter,
    MaxPooling2D: NoneReadCounter,
    MaxPooling3D: NoneReadCounter,
    ReLU: NoneReadCounter,
    Softmax: NoneReadCounter,
}


def _auto_counter(layer, auto_type_dict):
    layer_type = type(layer)
    if layer_type in auto_type_dict:
        return auto_type_dict[layer_type](layer)
    else:
        raise ValueError(
            "Counter type unknown for layer {}. Define it manually "
            "using the math_counter or read_counter arguments.".format(layer_type)
        )
