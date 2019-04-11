# Build layers

import tensorflow as tf

def eval_params(func):
    """
    Decorator to evaluate the parameters returned by get_weights method
    using a tf session. Initializes variables if needed."""

    def inner(layer, eval_vars):
        variables = func(layer)

        if eval_vars:
            with tf.Session() as sess:
                try:
                    weights = sess.run(variables)
                except tf.errors.FailedPreconditionError:
                    sess.run(tf.global_variables_initializer())
                    weights = sess.run(variables)
        else:
            weights = variables

        return weights
    return inner

class Layer(object):
    def __init__(self, name):
        self.name = name
        self.output_shape = None

    def set_output_shape(func):
        def call(self, x):
            output = func(self, x)
            if output == []:
                self.output_shape = []
            else:
                self.output_shape = output.get_shape().as_list()
            return output
        return call

    # By default, return empty list for weights
    def get_weights(self):
        return []

    def get_config(self):
        config = {'name':self.name, 'output_shape':self.output_shape}
        return config

    def catch_empty(func):
        def call(self, x):
            if x == []:
                return []
            return func(self, x)
        return call

class Dense(Layer):
    def __init__(self, units, name, use_bias=True):
        super().__init__(name)
        self.units = units
        self.use_bias = use_bias
        self.w = []
        self.b = []

    @Layer.set_output_shape
    @Layer.catch_empty
    def __call__(self, x):
        # Return empty output for empty layer
        if self.units == 0:
            return []

        n_in = x.get_shape().as_list()[-1]
        self.w = tf.get_variable(self.name + '_w', shape=[n_in, self.units])

        if self.use_bias:
            self.b = tf.get_variable(self.name + '_b', shape=[self.units])
            output = tf.nn.xw_plus_b(x, self.w, self.b, name=self.name)
        else:
            output = tf.matmul(x, self.w, name=self.name)

        return output

    def get_config(self, eval_weights=False):
        config = {'name':self.name, 'units':self.units,
            'use_bias':self.use_bias, 'output_shape':self.output_shape,
            'weights':self.get_weights(eval_weights)}
        return config

    @eval_params
    def get_weights(self, eval_weights=True):
        return self.w, self.b

class BatchNormalization(Layer):
    def __init__(self, name, epsilon=1e-8):
        super().__init__(name)
        self.epsilon = epsilon
        self.beta = []
        self.scale = []

    @Layer.set_output_shape
    @Layer.catch_empty
    def __call__(self, x):
        n_out = x.get_shape().as_list()[-1]

        batch_mean, batch_var = tf.nn.moments(x, [0])
        self.scale = tf.get_variable(self.name+'_scale', initializer=tf.ones([n_out]))
        self.beta = tf.get_variable(self.name+'_beta', initializer=tf.zeros([n_out]))
        output = tf.nn.batch_normalization(x, batch_mean, batch_var,
            self.beta, self.scale, self.epsilon, name=self.name)

        return output

    def get_config(self, eval_weights=False):
        config = {'name':self.name, 'epsilon':self.epsilon,
            'output_shape':self.output_shape,
            'weights':self.get_weights(eval_weights)}
        return config

    @eval_params
    def get_weights(self, eval_weights=True):
        return self.beta, self.scale

class Activation(Layer):
    def __init__(self, activation, name):
        super().__init__(name)

        assert activation in ['linear', 'softmax', 'relu'], \
            'activation {} not suppoted'.format(activation)
        self.activation = activation

    @Layer.set_output_shape
    @Layer.catch_empty
    def __call__(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'relu':
            return tf.nn.relu(x, name=self.name)
        else:
            return None

    def get_config(self):
        config = {'name' : self.name, 'activation' : self.activation,
            'output_shape':self.output_shape}
        return config

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, name, strides=1, padding='valid', use_bias=True):
        super().__init__(name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.f = []
        self.b = []

    @Layer.set_output_shape
    @Layer.catch_empty
    def __call__(self, x):
        # Return empty output for empty layer
        if self.filters == 0:
            return []

        shape_in = x.get_shape().as_list()
        channels_in = shape_in[-1]

        self.f = tf.get_variable(self.name + '_f', shape=[self.kernel_size,
            self.kernel_size, channels_in, self.filters])

        strides = (1, self.strides, self.strides, 1)

        if self.use_bias:
            output = tf.nn.conv2d(x, self.f, strides, self.padding.upper())
            self.b = tf.get_variable(self.name + '_b', initializer=tf.zeros([self.filters]))
            b = tf.reshape(self.b, [-1, 1, 1, self.filters])
            output = tf.add(output, b, name=self.name)
        else:
            output = tf.nn.conv2d(x, self.f, strides, self.padding.upper(), name=self.name)

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

class AveragePooling2D(Layer):
    def __init__(self, pool_size, name, strides=None, padding='valid'):
        super().__init__(name)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    @Layer.set_output_shape
    @Layer.catch_empty
    def __call__(self, x):
        shape_in = x.get_shape().as_list()
        ksize = (1, self.pool_size[0], self.pool_size[1], 1)

        if self.strides is None:
            strides = ksize
        else:
            strides = (1, self.strides[0], self.strides[1], 1)

        output = tf.nn.avg_pool(x, ksize, strides, self.padding.upper(), name=self.name)
        self.output_shape = output.get_shape().as_list()
        return output

    def get_config(self):
        config = {'name':self.name, 'pool_size':self.pool_size, 'strides':self.strides,
            'padding':self.padding, 'output_shape':self.output_shape}
        return config

class GlobalAveragePooling2D(Layer):
    def __init__(self, name):
        super().__init__(name)

    @Layer.set_output_shape
    @Layer.catch_empty
    def __call__(self, x):
        output = tf.reduce_mean(x, axis=[1, 2], name=self.name)
        self.output_shape = output.get_shape().as_list()
        return output
