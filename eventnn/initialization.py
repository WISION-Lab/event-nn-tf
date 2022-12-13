import nlopt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from eventnn.utils import _data_generator, _n_total


def mean_input(model, x, data_steps=None):
    batch_size = model.input_shape[0]
    x_gen = _data_generator(x, None, data_steps, batch_size, model.input.dtype)
    total = 0
    for x_i, n_items in x_gen:
        total += tf.reduce_sum(x_i[:n_items], axis=0)
    mean = total / _n_total(x, data_steps, batch_size)
    return mean.numpy()


def mean_internal(
    model, layers, x, data_steps=None, optimize_steps=500, algorithm=nlopt.LD_MMA, verbose=False
):
    """
    Because gradients here are not, in general, smooth (due to ReLU
    activations), the optimization algorithm should not be one which
    expects differentiable gradients (i.e., one which tries to
    approximate the Hessian).
    """
    batch_size = model.input_shape[0]
    n_total = _n_total(x, data_steps, batch_size)
    x_gen = _data_generator(x, None, data_steps, batch_size, model.input.dtype)
    model_tmp = Model(inputs=model.input, outputs=[layer.output[0] for layer in layers])
    means = [0] * len(layers)

    for x_i, n_items in x_gen:
        activations = _as_list(model_tmp(x_i))
        for j, activations_j in enumerate(activations):
            means[j] += tf.reduce_sum(activations_j[:n_items], axis=0) / n_total

    iteration = 0

    def callback(flat_value, flat_grad):
        with tf.GradientTape() as tape:
            flat_tensor = tf.convert_to_tensor(flat_value)
            tape.watch(flat_tensor)
            null_inputs = tf.reshape(flat_tensor, model.input_shape[1:])
            null_batch = tf.repeat(tf.expand_dims(null_inputs, axis=0), batch_size, axis=0)
            null_activations = _as_list(model_tmp(null_batch))
            error = 0
            for null_activations_i, mean in zip(null_activations, means):
                error += tf.norm(null_activations_i[0] - mean)
        grad = tape.gradient(error, flat_tensor)
        flat_grad[:] = grad.numpy()
        cost = float(error)
        nonlocal iteration
        iteration += 1
        if verbose:
            print("Iteration: {} - Cost: {:.6g}".format(iteration, cost), flush=True)
        return cost

    n_dims = int(np.prod(model.input_shape[1:]))
    optimizer = nlopt.opt(algorithm, n_dims)
    optimizer.set_min_objective(callback)
    optimizer.set_maxeval(optimize_steps)
    optimal = optimizer.optimize(np.zeros(n_dims))
    return optimal.reshape(model.input_shape[1:])


def _as_list(value):
    return value if isinstance(value, list) else [value]
