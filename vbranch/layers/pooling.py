from .core import Layer

import tensorflow as tf

class AveragePooling2D(Layer):
    def __init__(self, pool_size, name, strides=None, padding='valid'):
        super().__init__(name)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    @Layer.call
    def __call__(self, x):
        shape_in = x.get_shape().as_list()

        ksize = (1, self.pool_size[0], self.pool_size[1], 1)

        if self.strides is None:
            strides = ksize
        else:
            strides = (1, self.strides[0], self.strides[1], 1)

        output = tf.nn.avg_pool(x, ksize, strides, self.padding.upper(),
            name='output')

        return output

    def get_config(self):
        config = {'name':self.name, 'pool_size':self.pool_size,
            'strides':self.strides, 'padding':self.padding,
            'output_shape':self.output_shape}
        return config

class GlobalAveragePooling2D(Layer):
    def __init__(self, name):
        super().__init__(name)

    @Layer.call
    def __call__(self, x):
        output = tf.reduce_mean(x, axis=[1, 2], name='output')
        return output
