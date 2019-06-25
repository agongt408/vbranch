from .core import Layer
from ..utils.generic import eval_params, EmptyOutput

import tensorflow as tf

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, name, strides=1, padding='valid',
            use_bias=True):
        super().__init__(name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.f = []
        self.b = []

    @Layer.call
    def __call__(self, x):
        # Return empty output for empty layer
        if self.filters == 0:
            return EmptyOutput()

        shape_in = x.get_shape().as_list()
        channels_in = shape_in[-1]

        self.f = tf.get_variable('filter', shape=[self.kernel_size,
            self.kernel_size, channels_in, self.filters])

        strides = (1, self.strides, self.strides, 1)

        if self.use_bias:
            output = tf.nn.conv2d(x, self.f, strides, self.padding.upper())
            self.b = tf.get_variable('bias', initializer=tf.zeros([self.filters]))

            b = tf.reshape(self.b, [-1, 1, 1, self.filters])
            output = tf.add(output, b, name='output')
        else:
            output = tf.nn.conv2d(x, self.f, strides, self.padding.upper(),
                name='output')

        return output

    def get_config(self, sess=None):
        config = {'name':self.name, 'filters':self.filters,
            'kernel_size':self.kernel_size, 'strides':self.strides,
            'padding':self.padding, 'use_bias':self.use_bias,
            'output_shape':self.output_shape, 'weights':self.get_weights(sess)}
        return config

class Conv1D(Layer):
    def __init__(self, filters, kernel_size, name, strides=1, padding='valid',
            use_bias=True):
        super().__init__(name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

    @Layer.call
    def __call__(self, x):
        shape_in = x.get_shape().as_list()
        channels_in = shape_in[-1]

        self.f = tf.get_variable('filter', shape=[self.kernel_size,
            channels_in, self.filters])

        if self.use_bias:
            output = tf.nn.conv1d(x,self.f,self.strides,self.padding.upper())
            self.b = tf.get_variable('bias', initializer=tf.zeros([self.filters]))

            b = tf.reshape(self.b, [-1, 1, self.filters])
            output = tf.add(output, b, name='output')
        else:
            output = tf.nn.conv1d(x, self.f, strides, self.padding.upper(),
                name='output')

        return output

    def get_config(self):
        config = {'name':self.name, 'filters':self.filters,
            'kernel_size':self.kernel_size, 'strides':self.strides,
            'padding':self.padding, 'use_bias':self.use_bias,
            'output_shape':self.output_shape, 'weights':self.get_weights()}
        return config
