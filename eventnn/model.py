import sys
from contextlib import contextmanager
from copy import copy

import nlopt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from eventnn.layers import Accumulator, EventLayer, Gate
from eventnn.utils import _data_generator, _n_total


class EventModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for node, name in (self.input, "input"), (self.output, "output"):
            if isinstance(node, (list, tuple)):
                if any(isinstance(x, (list, tuple)) for x in node):
                    raise ValueError(
                        "EventModel {} must be a tensor or a flat list of tensors.".format(name)
                    )
        if isinstance(self.input, (list, tuple)):
            self._null_inputs = []
            for i, (data, x) in enumerate(zip(self._dummy_input(), self.input)):
                self._null_inputs.append(
                    tf.Variable(data, shape=x.shape, name="null_inputs_{}".format(i))
                )
        else:
            self._null_inputs = tf.Variable(
                self._dummy_input(), shape=self.input.shape, name="null_inputs"
            )
        self.update_null_values()

    @contextmanager
    def all_outputs(self):
        all_outputs_old = [layer.all_outputs for layer in self.gates]
        for layer in self.gates:
            layer.all_outputs = True
        try:
            yield None
        finally:
            for layer, all_outputs in zip(self.gates, all_outputs_old):
                layer.all_outputs = all_outputs

    def build(self, batch_input_shape):
        for layer in self.event_layers:
            if not layer.built:
                layer.build(layer.input_shape)
        super().build(batch_input_shape)

    def call(self, inputs, training=False, mask=None):
        return super().call(inputs, training=training, mask=mask)

    def call_static(self, inputs, training=False, mask=None):
        trace_id = ",".join([layer.get_trace_id() for layer in self.event_layers])
        return self._call_static_internal(inputs, trace_id, training=training, mask=mask)

    def count_ops_conventional(
        self,
        x,
        data_steps=None,
        sparse=False,
        temporal_axis=None,
        temporal_mean=True,
        verbose=False,
        static_graph=False,
    ):
        return self._count_ops(
            x,
            1,
            data_steps,
            sparse,
            temporal_axis,
            temporal_mean,
            verbose,
            self.call_static if static_graph else self.call,
        )

    def count_ops_event(
        self,
        x,
        time_steps=1,
        data_steps=None,
        sparse=False,
        temporal_axis=None,
        temporal_mean=True,
        verbose=False,
        static_graph=False,
    ):
        return self._count_ops(
            x,
            time_steps,
            data_steps,
            sparse,
            temporal_axis,
            temporal_mean,
            verbose,
            self.step_static if static_graph else self.step,
        )

    @contextmanager
    def event_mode(self):
        event_mode_old = [layer.event_mode for layer in self.event_layers]
        for layer in self.event_layers:
            layer.event_mode = True
        try:
            yield None
        finally:
            for layer, event_mode in zip(self.event_layers, event_mode_old):
                layer.event_mode = event_mode

    def evaluate_conventional(
        self,
        metrics,
        x,
        y=None,
        data_steps=None,
        temporal_axis=None,
        max_time=None,
        verbose=False,
        static_graph=False,
    ):
        return self._evaluate(
            metrics,
            x,
            y,
            1,
            data_steps,
            temporal_axis,
            max_time,
            verbose,
            self.call_static if static_graph else self.call,
        )

    def evaluate_event(
        self,
        metrics,
        x,
        y=None,
        time_steps=1,
        data_steps=None,
        temporal_axis=None,
        max_time=None,
        verbose=False,
        static_graph=False,
    ):
        return self._evaluate(
            metrics,
            x,
            y,
            time_steps,
            data_steps,
            temporal_axis,
            max_time,
            verbose,
            self.step_static if static_graph else self.step,
        )

    def get_config(self):
        return super().get_config()

    def load_weights(self, *args, **kwargs):
        self.build(self.input_shape)
        super().load_weights(*args, **kwargs)

    @contextmanager
    def op_counting(self, sparse=False):
        layer_count_ops_old = [layer.count_ops for layer in self.event_layers]
        layer_count_sparse_old = [layer.count_sparse for layer in self.event_layers]
        policy_count_ops_old = [layer.policy.count_ops for layer in self.gates]
        for layer in self.event_layers:
            layer.count_ops = True
            layer.count_sparse = sparse
        for layer in self.gates:
            layer.policy.count_ops = True
        try:
            yield None
        finally:
            for layer, count_ops, count_sparse in zip(
                self.event_layers, layer_count_ops_old, layer_count_sparse_old
            ):
                layer.count_ops = count_ops
                layer.count_sparse = count_sparse
            for layer, count_ops in zip(self.gates, policy_count_ops_old):
                layer.policy.count_ops = count_ops

    def optimize(
        self,
        variables,
        metric,
        alpha,
        x,
        y=None,
        time_steps=1,
        data_steps=None,
        op_types=None,
        optimize_steps=1000,
        algorithm=nlopt.LN_SBPLX,
        lower_bounds=None,
        upper_bounds=None,
        initial_steps=None,
        max_metric=True,
        verbose=True,
    ):
        results_conventional = self.evaluate_conventional([metric], x, y, data_steps=data_steps)

        def cost_terms(flat_value):
            _apply_flat(variables, flat_value)
            results_event = self.evaluate_event(
                [metric], x, y=y, time_steps=time_steps, data_steps=data_steps
            )
            ops = self.count_ops_event(x, time_steps=time_steps, data_steps=data_steps)
            op_sums = _sum_types(ops, op_types)

            sign = 1 if max_metric else -1
            errors = np.maximum(
                0, sign * (results_conventional[metric.name] - results_event[metric.name])
            )
            return np.sum(op_sums * errors), np.min(errors)

        scale, _ = cost_terms(_flattened(variables))
        iteration = 0

        def callback(flat_value, _):
            cost_1, cost_2 = cost_terms(flat_value)
            cost = cost_1 + alpha * scale * cost_2
            nonlocal iteration
            iteration += 1
            if verbose:
                print(
                    "Iteration: {} - Cost: {:.6g}".format(iteration, cost),
                    file=sys.stderr,
                    flush=True,
                )
            return cost

        optimizer = nlopt.opt(algorithm, _flat_size(variables))
        if lower_bounds is not None:
            optimizer.set_lower_bounds(_defaulted_params(variables, lower_bounds, -np.inf))
        if upper_bounds is not None:
            optimizer.set_upper_bounds(_defaulted_params(variables, upper_bounds, +np.inf))
        starting = _flattened(variables)
        if initial_steps is not None:
            default_step = optimizer.get_initial_step(starting)
            optimizer.set_initial_step(_defaulted_params(variables, initial_steps, default_step))
        optimizer.set_min_objective(callback)
        optimizer.set_maxeval(optimize_steps)
        optimal = optimizer.optimize(starting)
        _apply_flat(variables, optimal)

    def predict_conventional(
        self, x, data_steps=None, temporal_axis=None, verbose=False, static_graph=False
    ):
        return self._predict(
            x,
            1,
            data_steps,
            temporal_axis,
            verbose,
            self.call_static if static_graph else self.call,
        )

    def predict_event(
        self,
        x,
        time_steps=1,
        data_steps=None,
        temporal_axis=None,
        verbose=False,
        static_graph=False,
    ):
        return self._predict(
            x,
            time_steps,
            data_steps,
            temporal_axis,
            verbose,
            self.step_static if static_graph else self.step,
        )

    def reset(self):
        for layer in self.event_layers:
            layer.reset()

    def reset_counters(self):
        for layer in self.event_layers:
            layer.reset_counters()
        for layer in self.gates:
            layer.policy.reset_counters()

    def step(self, inputs):
        with self.event_mode():
            return self.call(inputs, training=False)

    def step_static(self, inputs):
        with self.event_mode():
            return self.call_static(inputs, training=False)

    def update_null_values(self):
        to_update = list(filter(lambda layer: layer.has_null_values, self.event_layers))
        model_tmp = Model(inputs=self.input, outputs=[layer.input for layer in to_update])
        for layer, layer_null_input in zip(to_update, model_tmp(self.null_inputs)):
            layer.update_null_values(layer_null_input)

    @property
    def accumulators(self):
        return list(filter(lambda a: isinstance(a, Accumulator), self.layers))

    @property
    def event_layers(self):
        return list(filter(lambda a: isinstance(a, EventLayer), self.layers))

    @property
    def gates(self):
        return list(filter(lambda a: isinstance(a, Gate), self.layers))

    @property
    def null_inputs(self):
        if isinstance(self._null_inputs, (list, tuple)):
            return [null_input.value() for null_input in self._null_inputs]
        else:
            return self._null_inputs.value()

    @null_inputs.setter
    def null_inputs(self, value):
        if isinstance(self._null_inputs, (list, tuple)):
            for null_input, value_i in zip(self._null_inputs, value):
                null_input.assign(value_i)
        else:
            self._null_inputs.assign(value)
        self.update_null_values()

    @tf.function
    def _call_static_internal(self, inputs, trace_id, training=False, mask=None):
        # The trace ID is used by an EventModel to determine whether the
        # model's call graph needs to be retraced. It is possible to
        # modify the logic of an EventLayer by changing its attributes
        # (e.g., by setting Differencer.policy). The new logic is not
        # captured by the original model trace, so we need to manually
        # retrace the graph.
        print(
            "Retracing model. Trace ID hash: {}".format(hex(hash(trace_id))),
            file=sys.stderr,
            flush=True,
        )
        return super().call(inputs, training=training, mask=mask)

    def _compute_batch_size(self):
        if isinstance(self.input, (list, tuple)):
            return self.input[0].shape[0]
        else:
            return self.input.shape[0]

    def _count_ops(
        self, x, time_steps, data_steps, sparse, temporal_axis, temporal_mean, verbose, call_func
    ):
        ops = self._empty_op_dict()
        batch_size = self._compute_batch_size()
        n_total = _n_total(x, data_steps, batch_size)
        x_gen = _data_generator(x, None, data_steps, batch_size, self.input.dtype)
        with self.op_counting(sparse=sparse):
            for i, (x_i, n_items) in enumerate(x_gen):
                self.reset()
                x_slices = _t_slices(x_i, temporal_axis)
                n_steps = time_steps if (temporal_axis is None) else len(x_slices)
                self._make_count_arrays(ops, n_items, n_steps, temporal_mean)
                for t1, x_i_t in enumerate(x_slices):
                    for t2 in range(time_steps):
                        call_func(x_i_t)
                        if (temporal_axis is None) and not temporal_mean:
                            self._read_counters(ops, n_items, t2)
                    if (temporal_axis is not None) and not temporal_mean:
                        self._read_counters(ops, n_items, t1)
                if temporal_mean:
                    self._read_counters(ops, n_items, 0, scale=1 / n_steps)
                if verbose:
                    _print_step_status(i, n_total)
        self.reset()
        return ops

    def _dummy_input(self):
        if isinstance(self.input, (list, tuple)):
            return [tf.zeros(_concrete_shape(x.shape), dtype=x.dtype) for x in self.input]
        else:
            return tf.zeros(_concrete_shape(self.input.shape), dtype=self.input.dtype)

    def _empty_op_dict(self):
        ops = {}
        for layer in self.event_layers:
            ops[layer.name] = {
                "math_ops": [],
                "read_ops": [],
                "overhead_math_ops": [],
                "overhead_read_ops": [],
                "overhead_write_ops": [],
            }
        return ops

    def _evaluate(
        self, metrics, x, y, time_steps, data_steps, temporal_axis, max_time, verbose, call_func
    ):
        metric_copies = _copy_metrics(metrics, time_steps, temporal_axis, max_time)
        batch_size = self._compute_batch_size()
        n_total = _n_total(x, data_steps, batch_size)
        xy_gen = _data_generator(x, y, data_steps, batch_size, self.input.dtype)
        for i, ((x_i, y_true_i), n_items) in enumerate(xy_gen):
            self.reset()
            x_slices = _t_slices(x_i, temporal_axis)
            y_true_slices = _t_slices(y_true_i, temporal_axis)
            for t1, (x_i_t, y_true_i_t) in enumerate(zip(x_slices, y_true_slices)):
                y_pred_i_t = None
                for t2 in range(time_steps):
                    y_pred_i_t = call_func(x_i_t)
                    if temporal_axis is None:
                        for metric in metric_copies:
                            _update_metric(metric[t2], y_true_i_t, y_pred_i_t, n_items)
                if temporal_axis is not None:
                    for metric in metric_copies:
                        _update_metric(
                            metric if (max_time is None) else metric[t1],
                            y_true_i_t,
                            y_pred_i_t,
                            n_items,
                        )
            if verbose:
                _print_step_status(i, n_total)
                _print_evaluation_status(metrics, metric_copies, temporal_axis, max_time)
        self.reset()
        return _read_metrics(metrics, metric_copies, temporal_axis, max_time)

    def _make_count_arrays(self, ops, n_items, n_steps, temporal_mean):
        template = np.zeros(1 if temporal_mean else n_steps, dtype=np.int64)
        for layer in self.event_layers:
            for _ in range(n_items):
                ops[layer.name]["overhead_math_ops"].append(np.copy(template))
                ops[layer.name]["overhead_read_ops"].append(np.copy(template))
                ops[layer.name]["overhead_write_ops"].append(np.copy(template))
                ops[layer.name]["math_ops"].append(np.copy(template))
                ops[layer.name]["read_ops"].append(np.copy(template))

    def _predict(self, x, time_steps, data_steps, temporal_axis, verbose, call_func):
        y_pred_all = [[] for _ in self.outputs]
        batch_size = self._compute_batch_size()
        n_total = _n_total(x, data_steps, batch_size)
        x_gen = _data_generator(x, None, data_steps, batch_size, self.input.dtype)
        for i, (x_i, n_items) in enumerate(x_gen):
            self.reset()
            t_slices = _t_slices(x_i, temporal_axis)
            n_steps = time_steps if (temporal_axis is None) else x_i.shape[temporal_axis]
            for t1, x_i_t in enumerate(t_slices):
                y_pred_i_t = None
                for t2 in range(time_steps):
                    y_pred_i_t = call_func(x_i_t)
                    if len(self.outputs) == 1:
                        # call_func returns a bare tensor when there is
                        # only one output.
                        y_pred_i_t = [y_pred_i_t]
                    if temporal_axis is None:
                        self._record_prediction(y_pred_all, y_pred_i_t, n_items, n_steps, t2)
                if temporal_axis is not None:
                    self._record_prediction(y_pred_all, y_pred_i_t, n_items, n_steps, t1)
            if verbose:
                _print_step_status(i, n_total)
        self.reset()
        return y_pred_all[0] if (len(self.outputs) == 1) else tuple(y_pred_all)

    def _read_counters(self, ops, n_items, t, scale=1.0):
        for i in range(n_items):
            offset = n_items - i
            for layer in self.event_layers:
                ops_i = ops[layer.name]
                ops_i["overhead_math_ops"][-offset][t] += layer.overhead_math_ops[i] * scale
                ops_i["overhead_read_ops"][-offset][t] += layer.overhead_read_ops[i] * scale
                ops_i["overhead_write_ops"][-offset][t] += layer.overhead_write_ops[i] * scale
                ops_i["math_ops"][-offset][t] += layer.math_ops[i] * scale
                ops_i["read_ops"][-offset][t] += layer.read_ops[i] * scale
                if isinstance(layer, Gate):
                    ops_i["overhead_math_ops"][-offset][t] += (
                        layer.policy.overhead_math_ops[i] * scale
                    )
                    ops_i["overhead_read_ops"][-offset][t] += (
                        layer.policy.overhead_read_ops[i] * scale
                    )
                    ops_i["overhead_write_ops"][-offset][t] += (
                        layer.policy.overhead_write_ops[i] * scale
                    )
        self.reset_counters()

    # noinspection PyMethodMayBeStatic
    def _record_prediction(self, y_pred_all, y_pred_i_t, n_items, n_steps, t):
        for y_pred_all_o, y_pred_i_t_o in zip(y_pred_all, y_pred_i_t):
            if t == 0:
                dtype = y_pred_i_t_o.dtype.as_numpy_dtype()
                y_pred_all_o.append(np.empty((n_steps,) + y_pred_i_t_o.shape[1:], dtype=dtype))
            for i in range(n_items):
                offset = n_items - i
                y_pred_all_o[-offset][t] = y_pred_i_t_o[i]


def _apply_flat(variables, flat_value):
    i = 0
    for var in variables:
        size = var.numpy().size
        var.assign(flat_value[i : i + size].reshape(var.shape))
        i += size


def _concrete_shape(shape):
    return tuple((1 if dim is None else dim) for dim in shape)


def _copy_metrics(metrics, time_steps, temporal_axis, max_time):
    metric_copies = []
    for metric in metrics:
        config = metric.get_config()
        config.update({"name": None})
        if temporal_axis is None:
            # The config gets modified by from_config, so we need to
            # copy it on each use.
            metric_copies.append(
                [type(metric).from_config(copy(config)) for _ in range(time_steps)]
            )
        else:
            if max_time is None:
                metric_copies.append(type(metric).from_config(config))
            else:
                metric_copies.append(
                    [type(metric).from_config(copy(config)) for _ in range(max_time)]
                )
    return metric_copies


def _flat_size(variables):
    return sum([var.numpy().size for var in variables])


def _flattened(variables):
    array = np.empty(_flat_size(variables))
    i = 0
    for var in variables:
        size = var.numpy().size
        array[i : i + size] = var.numpy().flatten()
        i += size
    return array


def _defaulted_params(variables, params, default):
    array = np.full(_flat_size(variables), default)
    i = 0
    for var, param in zip(variables, params):
        size = var.numpy().size
        if param is not None:
            array[i : i + size] = np.full_like(var.numpy(), param).flatten()
        i += size
    return array


def _define_ragged_axes(item):
    return item.to_tensor() if isinstance(item, tf.RaggedTensor) else item


def _filter_items(items, n_items):
    if isinstance(items, (list, tuple)):
        return type(items)(x[:n_items] for x in items)
    else:
        return items[:n_items]


def _print_evaluation_status(metrics, metric_copies, temporal_axis, uniform_time):
    for i, metric in enumerate(metrics):
        if (temporal_axis is None) or (uniform_time is not None):
            print("  {}:".format(metric.name), file=sys.stderr, flush=True)
            for t, metric_t in enumerate(metric_copies[i]):
                print("    {}: {:.4g}".format(t, metric_t.result()), file=sys.stderr, flush=True)
        else:
            print(
                "  {}: {:.4g}".format(metric.name, metric_copies[i].result()),
                file=sys.stderr,
                flush=True,
            )


def _print_step_status(i, n_total):
    print("Item {} of {}".format(i + 1, n_total), file=sys.stderr, flush=True)


def _read_metrics(metrics, metric_copies, temporal_axis, uniform_time):
    results = {}
    for metric in metrics:
        if (temporal_axis is None) or (uniform_time is not None):
            results[metric.name] = np.array(
                [metric_t.result() for metric_t in metric_copies[len(results)]]
            )
        else:
            results[metric.name] = np.array(metric_copies[len(results)].result())
    return results


def _sum_types(ops, op_types):
    total = 0
    for op_type in op_types:
        total += ops[op_type]
    return total


def _t_slices(x_i, temporal_axis):
    multiple = isinstance(x_i, (list, tuple))
    if not multiple:
        x_i = (x_i,)
    x_i = tuple(_define_ragged_axes(x_i_j) for x_i_j in x_i)
    if temporal_axis is None:
        t_slices = tuple([x_i_j] for x_i_j in x_i)
    else:
        t_slices = tuple(tf.unstack(x_i_j, axis=temporal_axis) for x_i_j in x_i)
    return list(zip(*t_slices)) if multiple else t_slices[0]


def _update_metric(metric, y_true, y_pred, n_items):
    filtered_true = _filter_items(y_true, n_items)
    filtered_pred = _filter_items(y_pred, n_items)
    metric.update_state(filtered_true, filtered_pred)
