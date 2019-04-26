# Virtual branching version of layers

from .. import layers as L

import tensorflow as tf
import collections

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
    def __init__(self, name, n_branches):
        self.name = name
        self.n_branches = n_branches
        self.output_shapes = []

    # Decorator for exanding input for branches and
    # setting output shape after calling the layer
    @staticmethod
    def call(func):
        def get_shape(x):
            if x == []:
                return []
            else:
                return x.get_shape().as_list()

        def call(layer, x):
            """All x must either be a single tensor, a VBOutput object,
            or a list of VBOutput objects."""

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
                layer._inbound_tensors = [x]
            else:
                # `x` is a single tensor
                # Expand input to match number of branches
                x = VBOutput([x] * layer.n_branches)
                layer._inbound_tensors = [x]

            output = func(layer, x)

            if not type(output) is list:
                output_list = [output]
            else:
                output_list = output

            layer.output_shapes = []
            for o in output_list:
                if type(o) is list:
                    out_shape = [get_shape(o[0]), get_shape(o[1])]
                else:
                    out_shape = [get_shape(o)]
                layer.output_shapes.append(out_shape)

            # Set vb history
            # First convert output to Python object
            vb_out = VBOutput(output)
            setattr(vb_out, '_vb_history', layer)

            return vb_out

        return call

    def get_weights(self):
        return []

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

def smart_add(x, y):
    # Intelligently add x and y to avoid error when adding empty list
    if y == []:
        return x
    else:
        return x + y

def smart_concat(xs, axis=-1, name='concat'):
    # Intelligently concat x and y to avoid error when concating empty list
    x_concat = []
    for x in xs:
        if x != []:
            x_concat.append(x)
    return tf.concat(x_concat, axis=axis, name=name)

class Dense(Layer):
    """
    Shared weights depends on size of shared output from previous layer and
    number of `shared units`. All other weights are unique per branch.
    """

    def __init__(self, units_list, n_branches, name, shared_units=0):
        """
        Args:
            - units_list: number of units per branch
            - shared_units: number of shared units per branch (aka size of
            shared output)
        """

        super().__init__(name, n_branches)

        assert n_branches == len(units_list), 'n_branches != len(units_list)'
        self.units_list = units_list
        self.shared_units = shared_units
        self.shared_branch = None

    @Layer.call
    def __call__(self, x):
        self.branches = []
        output_list = []

        if self.shared_units == 0:
            for i in range(self.n_branches):
                layer = L.Dense(self.units_list[i],self.name+'_vb'+str(i+1))

                if type(x[i]) is list:
                    input_ = smart_concat(x[i], -1)
                else:
                    input_ = x[i]

                x_out = layer(input_)
                self.branches.append(layer)
                output_list.append(x_out)

            return output_list

        # For efficiency, only apply computation to shared_in ONCE
        self.shared_branch = L.Dense(self.shared_units,self.name+'_shared_to_shared')

        for i in range(self.n_branches):
            assert self.units_list[i] >= self.shared_units, 'units < shared_units'
            unique_units = self.units_list[i] - self.shared_units

            # Operations to build the rest of the layer
            shared_to_unique = L.Dense(unique_units,
                self.name+'_vb'+str(i+1)+'_shared_to_unique')
            unique_to_shared = L.Dense(self.shared_units,
                self.name+'_vb'+str(i+1)+'_unique_to_shared')
            unique_to_unique = L.Dense(unique_units,
                self.name+'_vb'+str(i+1)+'_unique_to_unique')

            if type(x[i]) is list:
                shared_in = x[i][0]
                unique_in = x[i][1]

                shared_out = smart_add(self.shared_branch(shared_in),
                    unique_to_shared(unique_in))
                unique_out = shared_to_unique(shared_in) + \
                    unique_to_unique(unique_in)
            else:
                shared_out = self.shared_branch(x[i])
                unique_out = shared_to_unique(x[i])

            cross_weights = CrossWeights(shared_to_unique=shared_to_unique,
                unique_to_shared=unique_to_shared,
                unique_to_unique=unique_to_unique)

            self.branches.append(cross_weights)
            output_list.append([shared_out, unique_out])

        return output_list

    @L.eval_params
    def get_weights(self, eval_weights=True):
        # Get weights for shared branch
        if self.shared_branch is None:
            weights = [[], []]
        else:
            weights = [self.shared_branch.w, self.shared_branch.b]

        # Get unique weights
        if self.shared_units == 0:
            for layer in self.branches:
                weights += [layer.w, layer.b]
        else:
            for layer in self.branches:
                weights += [layer.shared_to_unique.w, layer.shared_to_unique.b,
                    layer.unique_to_shared.w, layer.unique_to_shared.b,
                    layer.unique_to_unique.w, layer.unique_to_unique.b]

        return weights

    def get_config(self, eval_weights=False):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'shared_units':self.shared_units, 'output_shapes':self.output_shapes,
            'units_list':self.units_list, 'weights':self.get_weights(eval_weights)}
        return config

class BatchNormalization(Layer):
    """
    Shared weights and unique weights. Size of shared dimension depends on
    size of shared output from previous layer.
    """

    def __init__(self, n_branches, name, epsilon=1e-8):
        super().__init__(name, n_branches)
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
            self.shared_branch = L.BatchNormalization(self.name+'_shared_to_shared',
                self.epsilon)

        for i in range(self.n_branches):
            # Operations to build the rest of the layer
            unique_to_unique = L.BatchNormalization(self.name+'_vb'+str(i+1)+\
                '_unique_to_unique', self.epsilon)

            if type(x[i]) is list:
                shared_out = self.shared_branch(x[i][0])
                unique_out = unique_to_unique(x[i][1])
                output_list.append([shared_out, unique_out])
            else:
                output_list.append(unique_to_unique(x[i]))

            self.branches.append(unique_to_unique)

        return output_list

    @L.eval_params
    def get_weights(self, eval_weights=True):
        # Get weights for shared branch
        if self.shared_branch is None:
            weights = [[], []]
        else:
            weights = [self.shared_branch.beta, self.shared_branch.scale]

        # Get unique weights
        for layer in self.branches:
            weights += [layer.beta, layer.scale]

        return weights

    def get_config(self, eval_weights=False):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'epsilon':self.epsilon, 'output_shapes':self.output_shapes,
            'weights':self.get_weights(eval_weights)}
        return config

class Activation(Layer):
    def __init__(self, activation, n_branches, name):
        super().__init__(name, n_branches)
        self.activation = activation

    @Layer.call
    def __call__(self, x):
        self.branches = []
        output_list = []

        for i in range(self.n_branches):
            layer = L.Activation(self.activation, self.name+'_vb'+str(i+1))

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

def Input(inputs, name='input'):
    """
    Instantiate Input layer
    Args:
        - inputs: single tensor or list of tensors
    """

    if not type(inputs) is list:
        inputs = [inputs]

    output = VBOutput(inputs)
    setattr(output, '_vb_history', InputLayer(name))

    return output
