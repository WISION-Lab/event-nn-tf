import unittest
from unittest import TestCase

import numpy as np
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
from eventnn.layers import Bias, Fuse


class TestBatchNormMathCounter(TestCase):
    def setUp(self):
        shape = (3,)
        self.layer = BatchNormalization(input_shape=shape)
        self.counter = BatchNormMathCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[0.0, 1.0, 2.0]])
        self.mask = tf.constant([[1.0, 1.0, 0.0]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[2.0, 2.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([[2.0, 2.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([[1.0, 2.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([[1.0, 2.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))


# Conv1D is easier to deal with than Conv2D, and the behavior should be
# basically identical.
class TestConv1DMathCounter(TestCase):
    def setUp(self):
        shape = (4, 2)
        self.layer = Conv1D(
            filters=2, kernel_size=3, padding="same", use_bias=False, input_shape=shape
        )
        self.counter = LinearMathCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[[0.0, -0.5], [2.0, 0.0], [5.0, 0.0], [1.0, 0.1]]])
        self.mask = tf.constant([[[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[[4.0, 4.0], [6.0, 6.0], [6.0, 6.0], [0.0, 0.0]]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([[[3.0, 3.0], [3.0, 3.0], [1.0, 1.0], [0.0, 0.0]]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([[[2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([[[2.0, 2.0], [2.0, 2.0], [1.0, 1.0], [0.0, 0.0]]])
        self.assertTrue(_tensors_equal(actual, expected))


class TestDenseMathCounter(TestCase):
    def setUp(self):
        shape = (4,)
        self.layer = Dense(units=2, use_bias=False, input_shape=shape)
        self.counter = LinearMathCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[0.0, -0.5, 2.0, 0.0]])
        self.mask = tf.constant([[1.0, 1.0, 0.0, 1.0]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[4.0, 4.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([[3.0, 3.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([[2.0, 2.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([[1.0, 1.0]])
        self.assertTrue(_tensors_equal(actual, expected))


class TestFuseMathCounter(TestCase):
    def setUp(self):
        shape = (2, 3)
        self.layer = Fuse(input_shape=shape)
        self.counter = LinearMathCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[[0.0, 1.0, 2.0], [-0.5, 5.0, 0.0]]])
        self.mask = tf.constant([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[2.0, 2.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([[2.0, 1.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([[1.0, 2.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([[1.0, 1.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))


# MaxPooling1D is easier to deal with than MaxPooling2D, and the
# behavior should be basically identical.
class TestMaxPooling1DMathCounter(TestCase):
    # noinspection DuplicatedCode
    def setUp(self):
        shape = (4, 2)
        self.layer = MaxPooling1D(pool_size=2, padding="valid", input_shape=shape)
        self.counter = MaxPoolingMathCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[[0.0, -0.5], [2.0, 0.0], [5.0, 0.0], [1.0, 0.1]]])
        self.mask = tf.constant([[[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[[2.0, 2.0], [0.0, 0.0]]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[[2.0, 2.0], [0.0, 0.0]]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        # sparse=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[[2.0, 2.0], [0.0, 0.0]]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        # incremental=True and sparse=True have no effect.
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[[2.0, 2.0], [0.0, 0.0]]])
        self.assertTrue(_tensors_equal(actual, expected))


class TestPointwiseMathCounter(TestCase):
    def setUp(self):
        shape = (3,)
        self.layer = Bias(input_shape=shape)
        self.counter = PointwiseMathCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[0.0, 1.0, 2.0]])
        self.mask = tf.constant([[1.0, 1.0, 0.0]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[1.0, 1.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([[1.0, 1.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        # sparse=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([[1.0, 1.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        # incremental=True and sparse=True have no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([[1.0, 1.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))


class TestBatchNormReadCounter(TestCase):
    def setUp(self):
        shape = (2, 3)
        self.layer = BatchNormalization(input_shape=shape)
        self.counter = BatchNormReadCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])
        self.mask = tf.constant([[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([4.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([4.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([2.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([2.0])
        self.assertTrue(_tensors_equal(actual, expected))


class TestBiasReadCounter(TestCase):
    def setUp(self):
        shape = (3,)
        self.layer = Bias(input_shape=shape)
        self.counter = BiasReadCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[0.0, 1.0, 2.0]])
        self.mask = tf.constant([[1.0, 1.0, 0.0]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([2.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([2.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([1.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([1.0])
        self.assertTrue(_tensors_equal(actual, expected))


# Conv1D is easier to deal with than Conv2D, and the behavior should be
# basically identical.
class TestConvReadCounter(TestCase):
    def setUp(self):
        shape = (2, 3)
        self.layer = Conv1D(
            filters=2, kernel_size=3, padding="same", use_bias=False, input_shape=shape
        )
        self.counter = ConvReadCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[[0.0, -0.5, 2.0], [0.0, 5.0, 0.0]]])
        self.mask = tf.constant([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([18.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([12.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([18.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([6.0])
        self.assertTrue(_tensors_equal(actual, expected))


class TestDenseReadCounter(TestCase):
    def setUp(self):
        shape = (4,)
        self.layer = Dense(units=2, use_bias=False, input_shape=shape)
        self.counter = DenseReadCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[0.0, -0.5, 2.0, 0.0], [0.0, 3.0, 5.0, 0.0]])
        self.mask = tf.constant([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([[8.0, 8.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([[6.0, 4.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([[8.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([[2.0, 0.0]])
        self.assertTrue(_tensors_equal(actual, expected))


class TestLeakyReLUReadCounter(TestCase):
    # noinspection DuplicatedCode
    def setUp(self):
        shape = (4,)
        self.layer = LeakyReLU(input_shape=shape)
        self.counter = LeakyReLUReadCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[0.0, 1.0, -2.0], [0.0, 1.0, -5.0]])
        self.mask = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([1.0, 0.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([1.0, 0.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        # sparse=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([1.0, 0.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        # incremental=True and sparse=True have no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([1.0, 0.0])
        self.assertTrue(_tensors_equal(actual, expected))


class TestNoneReadCounter(TestCase):
    # noinspection DuplicatedCode
    def setUp(self):
        shape = (4,)
        self.layer = Fuse(input_shape=shape)
        self.counter = NoneReadCounter(self.layer)
        self.counter.build((None,) + shape)
        self.x = tf.constant([[0.0, 1.0, -2.0], [0.0, 1.0, -5.0]])
        self.mask = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    def test_call(self):
        actual = self.counter(self.x, self.mask, incremental=False, sparse=False)
        expected = tf.constant([0.0, 0.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_incremental(self):
        # incremental=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=False)
        expected = tf.constant([0.0, 0.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse(self):
        # sparse=True has no effect.
        actual = self.counter(self.x, self.mask, incremental=False, sparse=True)
        expected = tf.constant([0.0, 0.0])
        self.assertTrue(_tensors_equal(actual, expected))

    def test_call_sparse_incremental(self):
        # incremental=True and sparse=True have no effect.
        actual = self.counter(self.x, self.mask, incremental=True, sparse=True)
        expected = tf.constant([0.0, 0.0])
        self.assertTrue(_tensors_equal(actual, expected))


def _tensors_equal(a, b):
    return np.all(np.abs(a - b) < 1e-6)


if __name__ == "__main__":
    unittest.main()
