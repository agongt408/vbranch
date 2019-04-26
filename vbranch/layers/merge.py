from .core import Layer

import tensorflow as tf

class Add(Layer):
    def __init__(self, name):
        super().__init__(name)

    @Layer.call
    def __call__(self, x):
        # x: list of tensors
        return tf.add_n(x, name=self.name)
