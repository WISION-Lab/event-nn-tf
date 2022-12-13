import tensorflow as tf
from tensorflow.keras import losses, Model

from eventnn.utils import _data_generator, _n_total


# This function is just here for reference; EventMasked.injector needs
# to be re-added before it can be used.
def apply_sensitivity(model, x, y=None, data_steps=None):
    batch_size = model.input_shape[0]
    xy_gen = _data_generator(x, y, data_steps, batch_size, model.input.dtype)
    models_tmp = []
    for layer in model.event_masked[:-1]:
        models_tmp.append(Model(inputs=model.input, outputs=[layer.output[0]]))
    loss_func = losses.get(model.loss)
    grad_sums = [0] * len(model.event_masked[:-1])
    for (x_i, y_i_true), n_items in xy_gen:
        for j, layer in enumerate(model.event_masked[:-1]):
            layer.injector = models_tmp[j](x_i)
            with tf.GradientTape() as tape:
                tape.watch(layer.injector)
                loss = loss_func(y_i_true[:n_items], model(x_i)[:n_items])
            grads = tape.gradient(loss, layer.injector)
            grad_sums[j] += tf.reduce_sum(tf.abs(grads), axis=0)
            layer.injector = None
    n_total = _n_total(x, data_steps, batch_size)
    for layer, grad_sum in zip(model.event_masked, grad_sums):
        layer.policy.adjustments.assign(n_total / grad_sum)


def apply_variance(model, x, data_steps=None):
    for layer in model.gates:
        model_tmp = Model(inputs=model.inputs, outputs=layer.output)
        activations = model_tmp.predict(x, batch_size=model_tmp.input_shape[0], steps=data_steps)
        layer.policy.adjustments.assign(tf.math.reduce_std(activations[0], axis=0))
