from .core import Layer, eval_params, EmptyOutput

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

        self.f = tf.get_variable(self.name + '_f', shape=[self.kernel_size,
            self.kernel_size, channels_in, self.filters])

        strides = (1, self.strides, self.strides, 1)

        if self.use_bias:
            output = tf.nn.conv2d(x, self.f, strides, self.padding.upper())
            self.b = tf.get_variable(self.name + '_b',
                initializer=tf.zeros([self.filters]))
            b = tf.reshape(self.b, [-1, 1, 1, self.filters])
            output = tf.add(output, b, name=self.name)
        else:
            output = tf.nn.conv2d(x, self.f, strides, self.padding.upper(),
                name=self.name)

        return output

    def get_config(self, eval_weights=False):
        config = {'name':self.name, 'filters':self.filters,
            'kernel_size':self.kernel_size, 'strides':self.strides,
            'padding':self.padding, 'use_bias':self.use_bias,
            'output_shape':self.output_shape,
            'weights':self.get_weights(eval_weights)}
        return config

    @eval_params
    def get_weights(self, eval_weights=True):
        return self.f, self.b

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

        self.f = tf.get_variable(self.name + '_f', shape=[self.kernel_size,
            channels_in, self.filters])

        if self.use_bias:
            output = tf.nn.conv1d(x,self.f,self.strides,self.padding.upper())
            self.b = tf.get_variable(self.name + '_b',
                initializer=tf.zeros([self.filters]))
            b = tf.reshape(self.b, [-1, 1, self.filters])
            output = tf.add(output, b, name=self.name)
        else:
            output = tf.nn.conv1d(x, self.f, strides, self.padding.upper(),
                name=self.name)

        return output

    def get_config(self):
        config = {'name':self.name, 'filters':self.filters,
            'kernel_size':self.kernel_size, 'strides':self.strides,
            'padding':self.padding, 'use_bias':self.use_bias,
            'output_shape':self.output_shape, 'weights':self.get_weights()}
        return config

    @eval_params
    def get_weights(self):
        return self.f, self.b
