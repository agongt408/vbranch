from .core import Layer

import tensorflow as tf

class Add(Layer):
    def __init__(self, name):
        super().__init__(name)

    @Layer.call
    def __call__(self, x):
        # x: list of tensors
        return tf.add_n(x, name='output')

class Concatenate(Layer):
    def __init__(self, name, axis=-1):
        super().__init__(name)
        self.axis = axis

    @Layer.call
    def __call__(self, x):
        # x: list of tensors
        return tf.concat(x, axis=self.axis, name='output')
