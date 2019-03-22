# Build layers

import tensorflow as tf

class Dense(object):
    def __init__(self, units, name, use_bias=True):
        self.units = units
        self.name = name
        self.use_bias = use_bias

    def __call__(self, x):
        n_in = x.get_shape().as_list()[-1]
        self.w = tf.get_variable(self.name + '_w', shape=[n_in, self.units])

        if self.use_bias:
            self.b = tf.get_variable(self.name + '_b', shape=[self.units])
            output = tf.nn.xw_plus_b(x, self.w, self.b)
        else:
            output = tf.matmul(x, self.w)

        return output

class BatchNormalization(object):
    def __init__(self, name, epsilon=1e-8):
        self.name = name
        self.epsilon = epsilon

    def __call__(self, x):
        n_out = x.get_shape().as_list()[-1]

        batch_mean, batch_var = tf.nn.moments(x, [0])
        self.scale = tf.get_variable(self.name+'_scale', initializer=tf.ones([n_out]))
        self.beta = tf.get_variable(self.name+'_beta', initializer=tf.zeros([n_out]))
        output = tf.nn.batch_normalization(x, batch_mean, batch_var,
            self.beta, self.scale, self.epsilon)

        return output

class Activation(object):
    def __init__(self, activation, name):
        assert activation in ['linear', 'softmax', 'relu'], \
            'activation {} not suppoted'.format(activation)

        self.activation = activation
        self.name = name

    def __call__(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'softmax':
            return tf.nn.softmax(x, name=self.name)
        elif self.activation == 'relu':
            return tf.nn.relu(x, name=self.name)
        else:
            return None
