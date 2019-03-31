# Build Virtual Branching layers

from . import layers as L

import tensorflow as tf
import collections

CrossWeights = collections.namedtuple('CrossWeights',
    ['shared_to_unique', 'unique_to_shared', 'unique_to_unique'])

class Layer(object):
    def __init__(self, name, n_branches):
        self.name = name
        self.n_branches = n_branches
        self.output_shapes = []

    def expand_input(call_func):
        def inner(layer, x):
            if type(x) is list:
                assert len(x) == layer.n_branches, 'len(x) != n_branches'
                x_list = x
            else:
                x_list = [x] * layer.n_branches

            return call_func(layer, x_list)
        return inner

    def set_output_shapes(func):
        def call(self, x_list):
            output = func(self, x_list)

            if not type(output) is list:
                output_list = [output]
            else:
                output_list = output

            self.output_shapes = [o.get_shape().as_list() for o in output_list]
            return output

        return call

    def get_weights(self):
        return []

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

class Dense(Layer):
    def __init__(self, units_list, n_branches, name, shared_units=0):
        super().__init__(name, n_branches)

        assert n_branches == len(units_list), 'n_branches != len(units_list)'
        self.units_list = units_list
        self.shared_units = shared_units

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        self.branches = []
        output_list = []

        if self.shared_units == 0:
            for i in range(self.n_branches):
                layer = L.Dense(self.units_list[i],self.name+'_vb'+str(i+1))
                x_out = layer(x_list[i])
                self.branches.append(layer)
                output_list.append(x_out)

            return output_list

        # For efficiency, only apply computation to shared_in ONCE
        self.shared_branch = L.Dense(self.shared_units, self.name+'_vb0')

        for i in range(self.n_branches):
            assert self.units_list[i] > self.shared_units, 'units <= shared_units'
            unique_units = self.units_list[i] - self.shared_units

            # Operations to build the rest of the layer
            shared_to_unique = L.Dense(unique_units,
                self.name+'_vb'+str(i+1)+'_shared_to_unique')
            unique_to_shared = L.Dense(self.shared_units,
                self.name+'_vb'+str(i+1)+'_unique_to_shared')
            unique_to_unique = L.Dense(unique_units,
                self.name+'_vb'+str(i+1)+'_unique_to_unique')

            shared_in = x_list[i][:, :self.shared_units]
            unique_in = x_list[i][:, self.shared_units:]
            shared_out = self.shared_branch(shared_in) + unique_to_shared(unique_in)
            unique_out = shared_to_unique(shared_in) + unique_to_unique(unique_in)

            cross_weights = CrossWeights(shared_to_unique=shared_to_unique,
                unique_to_shared=unique_to_shared,
                unique_to_unique=unique_to_unique)
            output = tf.concat([shared_out, unique_out], -1, name=self.name+'_vb'+str(i+1))

            self.branches.append(cross_weights)
            output_list.append(output)

        return output_list

    @L.eval_params
    def get_weights(self):
        try:
            if self.shared_units == 0:
                weights = []
                for layer in self.branches:
                    weights += [layer.w, layer.b]
            else:
                weights = [self.shared_branch.w, self.shared_branch.b]
                for layer in self.branches:
                    weights += [layer.shared_to_unique.w, layer.shared_to_unique.b,
                        layer.unique_to_shared.w, layer.unique_to_shared.b,
                        layer.unique_to_unique.w, layer.unique_to_unique.b]
        except AttributeError:
            weights = []

        return weights

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'shared_units':self.shared_units, 'output_shapes':self.output_shapes,
            'units_list':self.units_list, 'weights':self.get_weights()}
        return config

class BatchNormalization(Layer):
    def __init__(self, n_branches, name, epsilon=1e-8):
        super().__init__(name, n_branches)
        self.epsilon = epsilon

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        self.branches = []
        output_list = []

        for i in range(self.n_branches):
            layer = L.BatchNormalization(self.name+'_vb'+str(i+1), self.epsilon)
            x_out = layer(x_list[i])
            self.branches.append(layer)
            output_list.append(x_out)

        return output_list

    @L.eval_params
    def get_weights(self):
        weights = []
        for layer in self.branches:
            weights += [layer.beta, layer.scale]
        return weights

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'epsilon':self.epsilon, 'output_shapes':self.output_shapes,
            'weights':self.get_weights()}
        return config

class Activation(Layer):
    def __init__(self, activation, n_branches, name):
        super().__init__(name, n_branches)
        self.activation = activation

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        self.branches = []
        output_list = []

        for i in range(self.n_branches):
            layer = L.Activation(self.activation, self.name+'_vb'+str(i+1))
            x_out = layer(x_list[i])
            self.branches.append(layer)
            output_list.append(x_out)

        return output_list

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes, 'activation':self.activation}
        return config

class Add(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        output = tf.reduce_sum(x_list, [0], name=self.name)
        return output

class Average(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        output = tf.reduce_mean(x_list, [0], name=self.name)
        return output

class Concatenate(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        output = tf.concat(x_list, [-1], name=self.name)
        return output
