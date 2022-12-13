from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Schedule(Layer):
    def reset(self):
        pass

    def get_trace_id(self):
        return str(id(self))

    @abstractmethod
    def step(self):
        pass


class Constant(Schedule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = tf.Variable(tf.constant(0, dtype=self.dtype), trainable=False, name="scale")

    def step(self):
        return self.scale.value()


class PeriodicReset(Schedule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = tf.Variable(tf.constant(0, dtype=self.dtype), trainable=False, name="scale")
        self.t = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, name="t")
        self.period = tf.Variable(
            tf.constant(0, dtype=tf.int64), trainable=False, name="period"
        )

    def reset(self):
        super().reset()
        self.t.assign(0)

    def step(self):
        if self.t % self.period == 0:
            output = tf.constant(0, dtype=self.dtype)
        else:
            output = self.scale.value()
        self.t.assign_add(1)
        return output


class LinearDecay(Schedule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = tf.Variable(tf.constant(0, dtype=self.dtype), trainable=False, name="scale")
        self.slope = tf.Variable(tf.constant(0, dtype=self.dtype), trainable=False, name="slope")
        self.t = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, name="t")

    def reset(self):
        super().reset()
        self.t.assign(0)

    def step(self):
        output = self.scale * tf.maximum(1.0 - self.slope * self.t, 0.0)
        self.t.assign_add(1)
        return output


class ExponentialDecay(Schedule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = tf.Variable(tf.constant(0, dtype=self.dtype), trainable=False, name="scale")
        self.decay = tf.Variable(tf.constant(0, dtype=self.dtype), trainable=False, name="decay")
        self.t = tf.Variable(0)

    def reset(self):
        super().reset()
        self.t.assign(0)

    def step(self):
        output = self.scale * tf.exp(-self.decay * self.t)
        self.t.assign_add(1)
        return output
