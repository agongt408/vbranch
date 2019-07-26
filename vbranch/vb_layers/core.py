# Virtual branching version of layers

from .. import layers as L
from ..utils.layer import *

import tensorflow as tf
import collections
from os.path import join
import numpy as np

CrossWeights = collections.namedtuple('CrossWeights',
    ['shared_to_unique', 'unique_to_shared', 'unique_to_unique'])

class VBOutput(object):
    """
    Class to store output of virtual branching layer
    Enables adding of _vb_history attribute to object"""

    def __init__(self, output):
        assert type(output) is list, \
            'output must be a list created by virtual branching layer'
        self.content = output

    def __len__(self):
        return len(self.content)

    def __getitem__(self, i):
        return self.content[i]

    def to_list(self):
        return self.content

    def __repr__(self):
        return str(self.content)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.content):
            result = self.content[self.i]
            self.i += 1
            return result
        else:
            raise StopIteration

class Layer(object):
    def __init__(self, name, n_branches, merge=False):
        self.name = name
        self.n_branches = n_branches
        self.output_shapes = []
        self.merge = merge

    # Decorator for exanding input for branches and
    # setting output shape after calling the layer
    @staticmethod
    def call(func):
        def call(layer, x):
            """All x must either be a single tensor, a VBOutput object,
            or a list of VBOutput objects."""
            # Store model scope
            layer.model_scope = tf.get_variable_scope().name
            # print(layer.name, x)

            # Set inbound nodes
            if type(x) is list:
                if all([isinstance(x_, tf.Tensor) for x_ in x]):
                    # List of tensors (i.e., inputs)
                    assert len(x) == layer.n_branches, \
                        'invalid number of tensors in list'

                    # Collate tensors into VBOutput object
                    x = VBOutput(x)
                    layer._inbound_tensors = [x]
                else:
                    # List of VBOutput objects (i.e., for merge layer)
                    # Validate input
                    for x_ in x:
                        assert isinstance(x_, VBOutput), \
                            'invalid input, not VBOutput object'
                    layer._inbound_tensors = x
            elif isinstance(x, VBOutput):
                # `x` is already VBOutput object
                layer._inbound_tensors = [x]
            else:
                # `x` is a single tensor
                # Expand input to match number of branches
                x = VBOutput([x] * layer.n_branches)
                layer._inbound_tensors = [x]

            with tf.variable_scope(layer.name):
                if layer.merge:
                    output_list = []
                    for i, output in enumerate(func(layer, x)):
                        if type(output) is list:
                            with tf.variable_scope('vb'+str(i+1)):
                                vb_output = smart_concat(output, name='output')
                            output_list.append(vb_output)
                        else:
                            output_list.append(output)
                else:
                    output_list = func(layer, x)

            layer.output_shapes = []

            for output in output_list:
                if type(output) is list:
                    # print(output)
                    out_shape = [get_shape(output[0]), get_shape(output[1])]
                else:
                    out_shape = [get_shape(output)]
                layer.output_shapes.append(out_shape)

            # Set vb history
            # First convert output to Python object
            vb_out = VBOutput(output_list)
            setattr(vb_out, '_vb_history', layer)

            setattr(layer, 'output', vb_out)

            return vb_out

        return call

    @eval_params
    def get_weights(self, sess=None):
        scope = join(self.model_scope, self.name)
        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return weights

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

class Dense(Layer):
    """
    Shared weights depends on size of shared output from previous layer and
    number of `shared units`. All other weights are unique per branch.
    """

    def __init__(self, units_list, n_branches, name, shared_units=0, merge=False):
        """
        Args:
            - units_list: number of units per branch
            - shared_units: number of shared units per branch (aka size of
            shared output)
        """
        super().__init__(name, n_branches, merge)

        assert n_branches == len(units_list), 'n_branches != len(units_list)'
        self.units_list = units_list
        self.shared_units = shared_units
        self.shared_branch = None

    @Layer.call
    def __call__(self, x):
        self.branches = []
        output_list = []

        fan_in = get_fan_in(x[0])
        fan_out = int(np.mean(self.units_list))
        assert all([fan_out == units for units in self.units_list])

        if self.shared_units > 0:
            # For efficiency, only apply computation to shared_in ONCE
            self.shared_branch = L.Dense(self.shared_units, 'shared_to_shared',
                fan_in=fan_in, fan_out=fan_out)

        for i in range(self.n_branches):
            if self.shared_units == 0:
                layer = L.Dense(self.units_list[i], 'vb'+str(i+1))
                x_out = layer(smart_concat(x[i], -1))
                self.branches.append(layer)
                output_list.append([EmptyOutput(), x_out])
            elif self.shared_units == fan_out:
                x_out = self.shared_branch(smart_concat(x[i], -1))
                output_list.append([x_out, EmptyOutput()])
            else:
                # Build the rest of the layer
                unique_units = self.units_list[i] - self.shared_units
                shared_to_unique = L.Dense(unique_units,
                    'vb'+str(i+1)+'_shared_to_unique',
                    fan_in=fan_in, fan_out=fan_out)
                unique_to_shared = L.Dense(self.shared_units,
                    'vb'+str(i+1)+'_unique_to_shared',
                    fan_in=fan_in, fan_out=fan_out)
                unique_to_unique = L.Dense(unique_units,
                    'vb'+str(i+1)+'_unique_to_unique',
                    fan_in=fan_in, fan_out=fan_out)

                if type(x[i]) is list:
                    shared_in = x[i][0]
                    unique_in = x[i][1]

                    shared_out = smart_add(self.shared_branch(shared_in),
                        unique_to_shared(unique_in))
                    unique_out = smart_add(shared_to_unique(shared_in),
                        unique_to_unique(unique_in))
                else:
                    shared_out = self.shared_branch(x[i])
                    unique_out = shared_to_unique(x[i])

                cross_weights = CrossWeights(
                    shared_to_unique=shared_to_unique,
                    unique_to_shared=unique_to_shared,
                    unique_to_unique=unique_to_unique
                )

                self.branches.append(cross_weights)
                output_list.append([shared_out, unique_out])

        return output_list

    def get_config(self, sess=None):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'shared_units':self.shared_units, 'output_shapes':self.output_shapes,
            'units_list':self.units_list, 'weights':self.get_weights(sess)}
        return config

class BatchNormalization(Layer):
    """
    Shared weights and unique weights. Size of shared dimension depends on
    size of shared output from previous layer.
    """

    def __init__(self, n_branches, name, epsilon=1e-8, merge=False):
        super().__init__(name, n_branches, merge)
        self.epsilon = epsilon
        self.shared_branch = None

    @Layer.call
    def __call__(self, x):
        self.branches = []
        output_list = []

        assert all([type(x[i]) is list for i in range(self.n_branches)]) or \
            all([not type(x[i]) is list for i in range(self.n_branches)])

        # For efficiency, only apply computation to shared_in ONCE
        if type(x[0]) is list:
            self.shared_branch = L.BatchNormalization('shared_to_shared',
                self.epsilon)

        for i in range(self.n_branches):
            # Operations to build the rest of the layer
            unique_to_unique = L.BatchNormalization('vb'+str(i+1)+\
                '_unique_to_unique', self.epsilon)

            if type(x[i]) is list:
                shared_out = self.shared_branch(x[i][0])
                unique_out = unique_to_unique(x[i][1])
                output_list.append([shared_out, unique_out])
            else:
                output_list.append(unique_to_unique(x[i]))

            self.branches.append(unique_to_unique)

        return output_list

    def get_config(self, sess=None):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'epsilon':self.epsilon, 'output_shapes':self.output_shapes,
            'weights':self.get_weights(sess)}
        return config

class Activation(Layer):
    def __init__(self, activation, n_branches, name, merge=False):
        super().__init__(name, n_branches, merge)
        self.activation = activation

    @Layer.call
    def __call__(self, x):
        self.branches = []
        output_list = []

        for i in range(self.n_branches):
            layer = L.Activation(self.activation, 'vb'+str(i+1))

            if type(x[i]) is list:
                shared_out = layer(x[i][0])
                unique_out = layer(x[i][1])
                output_list.append([shared_out, unique_out])
            else:
                output_list.append(layer(x[i]))

            self.branches.append(layer)

        return output_list

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes, 'activation':self.activation}
        return config

class InputLayer(Layer):
    # This layer does not actually show up in the functional Model
    # so we do not need to provide n_branches attribute
    def __init__(self, name):
        self.name = name
        self._inbound_tensors = []

def Input(inputs, num_branches, name='input'):
    """
    Instantiate Input layer
    Args:
        - inputs: single tensor or list of tensors
    """

    # Expand inputs if single tensor
    if not type(inputs) is list:
        inputs = [inputs] * num_branches

    output = VBOutput(inputs)
    setattr(output, '_vb_history', InputLayer(name))

    return output
