import unittest
from unittest import TestCase

import numpy as np
import tensorflow as tf

from eventnn.layers import Accumulator, Bias, Fuse, Gate, LayerWrapper, Mask, Unmask
from eventnn.policies import Threshold
from eventnn.schedules import Constant


class TestBias(TestCase):
    def setUp(self):
        self.layer = Bias()
        self.layer.build((None, 2, 3))
        self.x = tf.constant([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        self.layer.bias.assign(tf.constant([[1.0, 2.0, 3.0]]))

    def test_build(self):
        # self.layer.build already called in setUp.
        self.assertTrue(self.layer.built)
        self.assertEqual(self.layer.bias.shape, (1, 3))

    def test_call(self):
        actual = self.layer(self.x)
        expected = tf.constant([[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_compute_output_shape(self):
        shape = (1, 2, 3, 42, 43, 44)
        actual = self.layer.compute_output_shape(shape)
        self.assertEqual(actual, shape)


class TestFuse(TestCase):
    def setUp(self):
        self.layer = Fuse()
        self.layer.build((None, 3, 2))
        self.x = tf.constant([[[1.0, 2.0], [3.0, 2.0], [4.0, 6.0]]])

    def test_call(self):
        actual = self.layer(self.x)
        expected = tf.constant([[8.0, 10.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_compute_output_shape(self):
        actual = self.layer.compute_output_shape((1, 2, 3, 42, 43, 44))
        expected = (1, 3, 42, 43, 44)
        self.assertEqual(actual, expected)


# noinspection DuplicatedCode
class TestAccumulator(TestCase):
    def setUp(self):
        self.layer = Accumulator()
        self.layer.count_ops = True
        self.layer.build(((1, 2, 3),) * 2)
        self.x_1 = tf.constant([[[1.0, 0.0, 3.0], [4.0, 5.0, 6.0]]])
        self.x_2 = tf.constant([[[4.0, 5.0, 6.0], [1.0, 7.0, 3.0]]])
        self.mask_1 = tf.constant([[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
        self.mask_2 = tf.constant([[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]])

    def test_build(self):
        # self.layer.build already called in setUp.
        self.assertTrue(self.layer.built)
        self.assertEqual(self.layer.accumulator.shape, (1, 2, 3))

    def test_call_conventional(self):
        self.layer.reset()
        actual_1 = self.layer.call_conventional((self.x_1, self.mask_1))
        actual_2 = self.layer.call_conventional((self.x_2, self.mask_2))
        expected_1 = (self.x_1, tf.ones_like(self.mask_1))
        expected_2 = (self.x_2, tf.ones_like(self.mask_2))
        self.assertTrue(_tensor_tuples_equal(actual_1, expected_1))
        self.assertTrue(_tensor_tuples_equal(actual_2, expected_2))

    def test_call_event(self):
        self.layer.reset()
        actual_1 = self.layer.call_event((self.x_1, self.mask_1))
        actual_2 = self.layer.call_event((self.x_2, self.mask_2))
        expected_1 = (self.x_1 * self.mask_1, self.mask_1)
        expected_2 = (expected_1[0] + self.x_2 * self.mask_2, self.mask_2)
        self.assertTrue(_tensor_tuples_equal(actual_1, expected_1))
        self.assertTrue(_tensor_tuples_equal(actual_2, expected_2))

    def test_compute_output_shape(self):
        shape = ((5, 2, 3),) * 2
        actual = self.layer.compute_output_shape(shape)
        self.assertEqual(actual, shape)

    def test_compute_single_input_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_input_shape((shape,) * 2)
        self.assertEqual(actual, shape)

    def test_compute_single_output_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_output_shape((shape,) * 2)
        self.assertEqual(actual, shape)

    def test_count_ops_conventional(self):
        self.layer.reset()
        self.layer.call_conventional((self.x_1, self.mask_1))
        self.layer.call_conventional((self.x_2, self.mask_2))
        for counter in (
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(counter))

    def test_count_ops_event(self):
        self.layer.reset()
        self.layer.call_event((self.x_1, self.mask_1))
        self.layer.call_event((self.x_2, self.mask_2))
        for counter in self.layer.math_ops, self.layer.read_ops:
            self.assertTrue(_tensor_zero(counter))
        expected_overhead = tf.constant([3.0 + 4.0])
        for counter in (
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensors_equal(counter, expected_overhead))

    def test_reset(self):
        self.layer.call_event((self.x_1, self.mask_1))
        self.layer.reset()
        for tensor in (
            self.layer.accumulator,
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(tensor))

    def test_update_null_values(self):
        self.layer.reset()
        self.assertTrue(_tensor_zero(self.layer.accumulator))
        self.layer.update_null_values((self.x_1, self.mask_1))
        self.assertTrue(_tensors_equal(self.layer.null_input, self.x_1))
        self.assertTrue(_tensors_equal(self.layer.null_output, self.x_1))
        self.layer.reset()
        self.assertTrue(_tensors_equal(self.layer.accumulator, self.x_1))


# noinspection DuplicatedCode
class TestGate(TestCase):
    def setUp(self):
        self.layer = Gate()
        self.layer.count_ops = True
        self.layer.policy = Threshold(Constant())
        self.layer.build(((1, 2, 3),) * 2)
        self.layer.policy.schedule.scale.assign(0.2)
        self.x_1 = tf.constant([[[1.0, 0.0, 3.0], [0.1, 5.0, 6.0]]])
        self.x_2 = tf.constant([[[1.1, 5.0, 6.0], [1.0, 7.0, 3.0]]])
        self.mask_1 = tf.constant([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])
        self.mask_2 = tf.constant([[[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])

    def test_build(self):
        # self.layer.build already called in setUp.
        self.assertTrue(self.layer.built)
        for tensor in self.layer.best, self.layer.delta:
            self.assertEqual(tensor.shape, (1, 2, 3))

    def test_call_conventional(self):
        self.layer.reset()
        actual_1 = self.layer.call_conventional((self.x_1, self.mask_1))
        actual_2 = self.layer.call_conventional((self.x_2, self.mask_2))
        expected_1 = (self.x_1, tf.ones_like(self.mask_1))
        expected_2 = (self.x_2, tf.ones_like(self.mask_2))
        self.assertTrue(_tensor_tuples_equal(actual_1, expected_1))
        self.assertTrue(_tensor_tuples_equal(actual_2, expected_2))

    def test_call_event(self):
        self.layer.reset()
        actual_1 = self.layer.call_event((self.x_1, self.mask_1))
        actual_2 = self.layer.call_event((self.x_2, self.mask_2))
        expected_1 = (
            tf.constant([[[1.0, float("nan"), float("nan")], [float("nan"), 5.0, float("nan")]]]),
            tf.constant([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        )
        expected_2 = (
            tf.constant([[[float("nan"), float("nan"), float("nan")], [1.0, 2.0, 3.0]]]),
            tf.constant([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]),
        )
        self.assertTrue(_tensor_tuples_equal(actual_1, expected_1))
        self.assertTrue(_tensor_tuples_equal(actual_2, expected_2))

    def test_compute_output_shape(self):
        shape = ((5, 2, 3),) * 2
        actual = self.layer.compute_output_shape(shape)
        self.assertEqual(actual, shape)

    def test_compute_single_input_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_input_shape((shape,) * 2)
        self.assertEqual(actual, shape)

    def test_compute_single_output_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_output_shape((shape,) * 2)
        self.assertEqual(actual, shape)

    def test_count_ops_conventional(self):
        self.layer.reset()
        self.layer.call_conventional((self.x_1, self.mask_1))
        self.layer.call_conventional((self.x_2, self.mask_2))
        for counter in (
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(counter))

    def test_count_ops_event(self):
        self.layer.reset()
        self.layer.call_event((self.x_1, self.mask_1))
        self.layer.call_event((self.x_2, self.mask_2))
        expected_overhead_math = tf.constant([8.0 + 8.0])
        self.assertTrue(_tensors_equal(self.layer.overhead_math_ops, expected_overhead_math))
        expected_overhead_rw = tf.constant([8.0 + 8.0 + 2.0 + 3.0])
        self.assertTrue(_tensors_equal(self.layer.overhead_read_ops, expected_overhead_rw))
        self.assertTrue(_tensors_equal(self.layer.overhead_write_ops, expected_overhead_rw))
        for counter in self.layer.math_ops, self.layer.read_ops:
            self.assertTrue(_tensor_zero(counter))

    def test_reset(self):
        self.layer.call_event((self.x_1, self.mask_1))
        self.layer.reset()
        for tensor in (
            self.layer.best,
            self.layer.delta,
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(tensor))

    def test_update_null_values(self):
        self.layer.reset()
        self.assertTrue(_tensor_zero(self.layer.best))
        self.layer.update_null_values((self.x_1, self.mask_1))
        self.assertTrue(_tensors_equal(self.layer.null_input, self.x_1))
        self.assertTrue(_tensors_equal(self.layer.null_output, self.x_1))
        self.layer.reset()
        self.assertTrue(_tensors_equal(self.layer.best, self.x_1))


# noinspection DuplicatedCode
class TestLayerWrapper(TestCase):
    def setUp(self):
        self.layer = LayerWrapper(Bias(input_shape=(3,)), incremental=False, buffered=False)
        self.layer.count_ops = True
        self.layer.build(((1, 3),) * 2)
        self.layer.layer.set_weights([np.array([1.0, 2.0, 3.0])])
        self.x_1 = tf.constant([[1.0, 0.0, 3.0]])
        self.x_2 = tf.constant([[1.1, 5.0, 6.0]])
        self.mask_1 = tf.constant([[1.0, 1.0, 0.0]])
        self.mask_2 = tf.constant([[1.0, 0.0, 0.0]])

    def test_build(self):
        # self.layer.build already called in setUp.
        for item in self.layer, self.layer.layer, self.layer.math_counter, self.layer.read_counter:
            self.assertTrue(item.built)

    def test_call_conventional(self):
        self.layer.reset()
        actual_1 = self.layer.call_conventional((self.x_1, self.mask_1))
        actual_2 = self.layer.call_conventional((self.x_2, self.mask_2))
        expected_1 = (self.layer.layer(self.x_1), tf.ones_like(self.mask_1))
        expected_2 = (self.layer.layer(self.x_2), tf.ones_like(self.mask_2))
        self.assertTrue(_tensor_tuples_equal(actual_1, expected_1))
        self.assertTrue(_tensor_tuples_equal(actual_2, expected_2))

    def test_call_event(self):
        self.layer.reset()
        actual_1 = self.layer.call_event((self.x_1, self.mask_1))
        actual_2 = self.layer.call_event((self.x_2, self.mask_2))
        expected_1 = (
            tf.constant([[2.0, 2.0, 6.0]]),
            tf.constant([[1.0, 1.0, 0.0]]),
        )
        expected_2 = (
            tf.constant([[2.1, 7.0, 9.0]]),
            tf.constant([[1.0, 0.0, 0.0]]),
        )
        self.assertTrue(_tensor_tuples_equal(actual_1, expected_1))
        self.assertTrue(_tensor_tuples_equal(actual_2, expected_2))

    def test_compute_output_shape(self):
        shape = ((5, 2, 3),) * 2
        actual = self.layer.compute_output_shape(shape)
        expected = (self.layer.layer.compute_output_shape(shape[0]),) * 2
        self.assertEqual(actual, expected)

    def test_compute_single_input_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_input_shape((shape,) * 2)
        self.assertEqual(actual, shape)

    def test_compute_single_output_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_output_shape((shape,) * 2)
        self.assertEqual(actual, shape)

    def test_count_ops_conventional(self):
        self.layer.reset()
        self.layer.call_conventional((self.x_1, self.mask_1))
        self.layer.call_conventional((self.x_2, self.mask_2))
        expected_math = tf.constant([2.0 + 1.0])
        self.assertTrue(_tensors_equal(self.layer.math_ops, expected_math))
        expected_read = tf.constant([2.0 + 1.0])
        self.assertTrue(_tensors_equal(self.layer.math_ops, expected_read))
        for counter in (
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(counter))

    def test_count_ops_event(self):
        self.layer.reset()
        self.layer.call_event((self.x_1, self.mask_1))
        self.layer.call_event((self.x_2, self.mask_2))
        expected_math = tf.constant([2.0 + 1.0])
        self.assertTrue(_tensors_equal(self.layer.math_ops, expected_math))
        expected_read = tf.constant([2.0 + 1.0])
        self.assertTrue(_tensors_equal(self.layer.math_ops, expected_read))
        for counter in (
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(counter))

    def test_reset(self):
        self.layer.call_event((self.x_1, self.mask_1))
        self.layer.reset()
        for tensor in (
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(tensor))

    def test_update_null_values(self):
        self.layer.reset()
        self.layer.update_null_values((self.x_1, self.mask_1))
        self.assertTrue(_tensors_equal(self.layer.null_input, self.x_1))
        self.assertTrue(_tensors_equal(self.layer.null_output, self.layer.layer(self.x_1)))


# noinspection DuplicatedCode
class TestMask(TestCase):
    def setUp(self):
        self.layer = Mask()
        self.layer.count_ops = True
        self.layer.build((1, 3))
        self.x_1 = tf.constant([[1.0, 0.0, 3.0]])
        self.x_2 = tf.constant([[1.1, 5.0, 6.0]])

    def test_call_conventional(self):
        self.layer.reset()
        actual_1 = self.layer.call_conventional(self.x_1)
        actual_2 = self.layer.call_conventional(self.x_2)
        expected_1 = (self.x_1, tf.ones_like(self.x_1))
        expected_2 = (self.x_2, tf.ones_like(self.x_2))
        self.assertTrue(_tensor_tuples_equal(actual_1, expected_1))
        self.assertTrue(_tensor_tuples_equal(actual_2, expected_2))

    def test_call_event(self):
        self.layer.reset()
        actual_1 = self.layer.call_event(self.x_1)
        actual_2 = self.layer.call_event(self.x_2)
        expected_1 = (self.x_1, tf.ones_like(self.x_1))
        expected_2 = (self.x_2, tf.ones_like(self.x_2))
        self.assertTrue(_tensor_tuples_equal(actual_1, expected_1))
        self.assertTrue(_tensor_tuples_equal(actual_2, expected_2))

    def test_compute_output_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_output_shape(shape)
        self.assertEqual(actual, (shape,) * 2)

    def test_compute_single_input_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_input_shape(shape)
        self.assertEqual(actual, shape)

    def test_compute_single_output_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_output_shape(shape)
        self.assertEqual(actual, shape)

    def test_count_ops_conventional(self):
        self.layer.call_conventional(self.x_1)
        self.layer.call_conventional(self.x_2)
        for counter in (
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(counter))

    def test_count_ops_event(self):
        self.layer.call_event(self.x_1)
        self.layer.call_event(self.x_2)
        for counter in (
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(counter))

    def test_reset(self):
        self.layer.call_event(self.x_1)
        self.layer.reset()
        for tensor in (
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(tensor))

    def test_update_null_values(self):
        self.layer.reset()
        self.layer.update_null_values(self.x_1)
        self.assertTrue(_tensors_equal(self.layer.null_input, self.x_1))
        self.assertTrue(_tensors_equal(self.layer.null_output, self.x_1))


# noinspection DuplicatedCode
class TestUnmask(TestCase):
    def setUp(self):
        self.layer = Unmask()
        self.layer.count_ops = True
        self.layer.build(((1, 3),) * 2)
        self.x_1 = tf.constant([[1.0, 0.0, 3.0]])
        self.x_2 = tf.constant([[1.1, 5.0, 6.0]])
        self.mask_1 = tf.constant([[1.0, 1.0, 0.0]])
        self.mask_2 = tf.constant([[1.0, 0.0, 0.0]])

    def test_call_conventional(self):
        self.layer.reset()
        actual_1 = self.layer.call_conventional((self.x_1, self.mask_1))
        actual_2 = self.layer.call_conventional((self.x_2, self.mask_2))
        expected_1 = self.x_1
        expected_2 = self.x_2
        self.assertTrue(_tensors_equal(actual_1, expected_1))
        self.assertTrue(_tensors_equal(actual_2, expected_2))

    def test_call_event(self):
        self.layer.reset()
        actual_1 = self.layer.call_event((self.x_1, self.mask_1))
        actual_2 = self.layer.call_event((self.x_2, self.mask_2))
        expected_1 = self.x_1
        expected_2 = self.x_2
        self.assertTrue(_tensors_equal(actual_1, expected_1))
        self.assertTrue(_tensors_equal(actual_2, expected_2))

    def test_compute_output_shape(self):
        shape = ((5, 2, 3),) * 2
        actual = self.layer.compute_output_shape(shape)
        self.assertEqual(actual, shape[0])

    def test_compute_single_input_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_input_shape((shape,) * 2)
        self.assertEqual(actual, shape)

    def test_compute_single_output_shape(self):
        shape = (5, 2, 3)
        actual = self.layer.compute_single_output_shape((shape,) * 2)
        self.assertEqual(actual, shape)

    def test_count_ops_conventional(self):
        self.layer.call_conventional(self.x_1)
        self.layer.call_conventional(self.x_2)
        for counter in (
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(counter))

    def test_count_ops_event(self):
        self.layer.call_event(self.x_1)
        self.layer.call_event(self.x_2)
        for counter in (
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(counter))

    def test_reset(self):
        self.layer.call_event(self.x_1)
        self.layer.reset()
        for tensor in (
            self.layer.math_ops,
            self.layer.read_ops,
            self.layer.overhead_math_ops,
            self.layer.overhead_read_ops,
            self.layer.overhead_write_ops,
        ):
            self.assertTrue(_tensor_zero(tensor))

    def test_update_null_values(self):
        self.layer.reset()
        self.layer.update_null_values((self.x_1, self.mask_1))
        self.assertTrue(_tensors_equal(self.layer.null_input, self.x_1))
        self.assertTrue(_tensors_equal(self.layer.null_output, self.x_1))


def _tensors_equal(a, b):
    return np.all(np.abs(a - b) < 1e-6)


def _tensor_tuples_equal(a, b):
    return all(
        np.all(
            np.logical_or(np.abs(a_i - b_i) < 1e-6, np.logical_and(np.isnan(a_i), np.isnan(b_i)))
        )
        for (a_i, b_i) in zip(a, b)
    )


def _tensor_zero(a):
    return _tensors_equal(a, tf.zeros_like(a))


if __name__ == "__main__":
    unittest.main()
