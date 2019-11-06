from .core import Layer
from ..utils.layer import *
from ..initializers import glorot_uniform, rectifier_init

import tensorflow as tf

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, name, strides=1, padding='valid',
            use_bias=True, fan_in=None, fan_out=None):
        super().__init__(name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.f = []
        self.b = []
        self.fan_in = fan_in
        self.fan_out = fan_out

    @Layer.call
    def __call__(self, x):
        # Return empty output for empty layer
        if self.filters == 0:
            return EmptyOutput()

        shape_in = x.get_shape().as_list()
        channels_in = shape_in[-1]
        kernel_size = check_2d_param(self.kernel_size)

        if self.fan_in is None or self.fan_out is None:
            # Calculate `fan_in` for weight initialization
            if type(self.kernel_size) is int:
                receptive_field_size = self.kernel_size**2
            else:
                receptive_field_size = self.kernel_size[0] * self.kernel_size[1]

            self.fan_in = get_fan_in(x) * receptive_field_size
            self.fan_out = self.filters * receptive_field_size

        # print(self.name, self.fan_in, self.fan_out)
        self.f = tf.get_variable('filter', initializer=\
            # glorot_uniform(kernel_size + [channels_in, self.filters],
            #     self.fan_in, self.fan_out))
            rectifier_init(kernel_size+[channels_in,self.filters], self.fan_in))

        strides = (1, *check_2d_param(self.strides), 1)

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

class ZeroPadding2D(Layer):
    def __init__(self, name, padding=(1,1)):
        super().__init__(name)
        self.padding = padding

    @Layer.call
    def __call__(self, x):
        dim_pad = check_2d_param(self.padding)
        padding_list = [[0,0], dim_pad, dim_pad, [0,0]]
        output = tf.pad(x, padding_list, mode='constant', name='output')
        return output
