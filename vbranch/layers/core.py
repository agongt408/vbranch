from ..utils.generic_utils import eval_params, EmptyOutput

import tensorflow as tf
from os.path import join

class Layer(object):
    def __init__(self, name):
        self.name = name
        self.output_shape = None

    @eval_params
    def get_weights(self, eval_weights=True):
        scope = join(self.model_scope, self.name) + '/'
        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return weights

    def get_config(self):
        config = {'name':self.name, 'output_shape':self.output_shape}
        return config

    # Decorator before each call to catch empty inputs and set output shape
    @staticmethod
    def call(func):
        def new_func(layer, x):
            # Store model scope
            layer.model_scope = tf.get_variable_scope().name

            # Set inbound nodes
            if type(x) is list:
                layer._inbound_tensors = x
            else:
                layer._inbound_tensors = [x]

            if isinstance(x, EmptyOutput):
                output = EmptyOutput()
                layer.output_shape = []
            else:
                with tf.variable_scope(layer.name):
                    output = func(layer, x)

                # Catch empty output
                if isinstance(output, EmptyOutput):
                    layer.output_shape = []
                else:
                    layer.output_shape = output.get_shape().as_list()

            # Add vbranch history here to output tensor
            # Similar to Keras Functional API
            # construct model later from input/output tensors only
            setattr(output, '_vb_history', layer)

            return output
        return new_func

class Dense(Layer):
    def __init__(self, units, name, use_bias=True):
        super().__init__(name)
        self.units = units
        self.use_bias = use_bias
        self.w = []
        self.b = []

    @Layer.call
    def __call__(self, x):
        # Return empty output for empty layer
        if self.units == 0:
            return EmptyOutput()

        n_in = x.get_shape().as_list()[-1]
        self.w = tf.get_variable('weight', shape=[n_in, self.units])

        if self.use_bias:
            self.b = tf.get_variable('bias', shape=[self.units])
            output = tf.nn.xw_plus_b(x, self.w, self.b, name='output')
        else:
            output = tf.matmul(x, self.w, name='output')

        return output

    def get_config(self, eval_weights=False):
        config = {'name':self.name, 'units':self.units,
            'use_bias':self.use_bias, 'output_shape':self.output_shape,
            'weights':self.get_weights(eval_weights)}
        return config

class BatchNormalization(Layer):
    def __init__(self, name, epsilon=1e-8):
        super().__init__(name)
        self.epsilon = epsilon
        self.beta = []
        self.scale = []

    @Layer.call
    def __call__(self, x):
        n_out = x.get_shape().as_list()[-1]

        batch_mean, batch_var = tf.nn.moments(x, [0])
        self.scale = tf.get_variable('scale', initializer=tf.ones([n_out]))
        self.beta = tf.get_variable('beta', initializer=tf.zeros([n_out]))

        output = tf.nn.batch_normalization(x, batch_mean, batch_var,
            self.beta, self.scale, self.epsilon, name='output')

        return output

    def get_config(self, eval_weights=False):
        config = {'name':self.name, 'epsilon':self.epsilon,
            'output_shape':self.output_shape,
            'weights':self.get_weights(eval_weights)}
        return config

class Activation(Layer):
    def __init__(self, activation, name):
        super().__init__(name)

        assert activation in ['linear', 'softmax', 'relu'], \
            'activation {} not suppoted'.format(activation)
        self.activation = activation

    @Layer.call
    def __call__(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'relu':
            return tf.nn.relu(x, name='output')
        else:
            return None

    def get_config(self):
        config = {'name' : self.name, 'activation' : self.activation,
            'output_shape':self.output_shape}
        return config

class Flatten(Layer):
    def __init__(self, name):
        super().__init__(name)

    @Layer.call
    def __call__(self, x):
        shape_in = x.get_shape().as_list()
        dim = np.prod(shape_in[1:])
        output = tf.reshape(x, [-1, dim], name='output')
        return output

class InputLayer(Layer):
    def __init__(self, name):
        super().__init__(name)
        self._inbound_tensors = []

def Input(input_tensor):
    """Instantiate Input layer"""
    # Declare layer
    input_layer = InputLayer(input_tensor.name)

    # Attach InputLayer as attribute to tensor
    setattr(input_tensor, '_vb_history', input_layer)
    return input_tensor
