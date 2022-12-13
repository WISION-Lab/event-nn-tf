import numpy as np
from tensorflow.keras.layers import *

from eventnn.layers import Accumulator, Bias, Gate, LayerWrapper, Mask, Unmask
from eventnn.model import EventModel


def dense(input_size, batch_size=1, npz_weights=None):
    inputs = Input(batch_input_shape=(batch_size, input_size))
    x = inputs

    if npz_weights is not None:
        w = np.load(npz_weights)
    else:
        w = None

    # Model input
    x = Mask()(x)
    x = Gate()(x)

    # Model layers
    x = _dense_block(x, w, "dense_1", units=200)
    x = _dense_block(x, w, "dense_2", units=200)
    x = _dense_block(x, w, "dense_3", units=10, final=True)

    # Model output
    x = Unmask()(x)

    return EventModel(inputs=inputs, outputs=[x])


def _dense_block(x, w, weight_name, units, final=False):
    linear = Dense(units, use_bias=False)
    linear_wrapper = LayerWrapper(linear, incremental=True, buffered=False)
    bias = Bias()
    bias_wrapper = LayerWrapper(bias, incremental=False, buffered=False)

    x = linear_wrapper(x)
    x = Accumulator()(x)
    x = bias_wrapper(x)
    if not final:
        x = LayerWrapper(ReLU(), incremental=False, buffered=False)(x)
        x = Gate()(x)
    else:
        x = LayerWrapper(Softmax(), incremental=False, buffered=False)(x)

    if w is not None:
        linear.set_weights([w["{}_weights".format(weight_name)]])
        bias.set_weights([w["{}_biases".format(weight_name)]])

    return x
