# Virtual branching version of layers

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

    # Decorator for formatting inputs to fit the right number of branches
    def expand_input(call_func):
        def inner(layer, x):
            if type(x) is list:
                assert len(x) == layer.n_branches, \
                    '{} != {}'.format(len(x), layer.n_branches)
                x_list = x
            else:
                x_list = [x] * layer.n_branches

            return call_func(layer, x_list)
        return inner

    # Decorator for setting output shape after calling the layer
    def set_output_shapes(func):
        def get_shape(x):
            if x == []:
                return []
            else:
                return x.get_shape().as_list()

        def call(self, x_list):
            output = func(self, x_list)

            if not type(output) is list:
                output_list = [output]
            else:
                output_list = output

            self.output_shapes = []
            for o in output_list:
                if type(o) is list:
                    out_shape = [get_shape(o[0]), get_shape(o[1])]
                else:
                    out_shape = [get_shape(o)]
                self.output_shapes.append(out_shape)

            return output

        return call

    def get_weights(self):
        return []

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

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

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        self.branches = []
        output_list = []

        if self.shared_units == 0:
            for i in range(self.n_branches):
                layer = L.Dense(self.units_list[i],self.name+'_vb'+str(i+1))

                if type(x_list[i]) is list:
                    input_ = smart_concat(x_list[i], -1)
                else:
                    input_ = x_list[i]

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

            if type(x_list[i]) is list:
                shared_in = x_list[i][0]
                unique_in = x_list[i][1]

                shared_out = smart_add(self.shared_branch(shared_in),
                    unique_to_shared(unique_in))
                unique_out = shared_to_unique(shared_in) + \
                    unique_to_unique(unique_in)
            else:
                shared_out = self.shared_branch(x_list[i])
                unique_out = shared_to_unique(x_list[i])

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

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        self.branches = []
        output_list = []

        assert all([type(x_list[i]) is list for i in range(self.n_branches)]) or \
            all([not type(x_list[i]) is list for i in range(self.n_branches)])

        # For efficiency, only apply computation to shared_in ONCE
        if type(x_list[0]) is list:
            self.shared_branch = L.BatchNormalization(self.name+'_shared_to_shared',
                self.epsilon)

        for i in range(self.n_branches):
            # Operations to build the rest of the layer
            unique_to_unique = L.BatchNormalization(self.name+'_vb'+str(i+1)+\
                '_unique_to_unique', self.epsilon)

            if type(x_list[i]) is list:
                shared_out = self.shared_branch(x_list[i][0])
                unique_out = unique_to_unique(x_list[i][1])
                output_list.append([shared_out, unique_out])
            else:
                output_list.append(unique_to_unique(x_list[i]))

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

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        self.branches = []
        output_list = []

        for i in range(self.n_branches):
            layer = L.Activation(self.activation, self.name+'_vb'+str(i+1))

            if type(x_list[i]) is list:
                shared_out = layer(x_list[i][0])
                unique_out = layer(x_list[i][1])
                output_list.append([shared_out, unique_out])
            else:
                output_list.append(layer(x_list[i]))

            self.branches.append(layer)

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
        if type(x_list[0]) is list:
            input_ = smart_concat(x_list, -1)
        else:
            input_ = x_list

        output = tf.reduce_sum(input_, [0], name=self.name)
        return output

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

class Average(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        if type(x_list[0]) is list:
            input_ = smart_concat(x_list, -1)
        else:
            input_ = x_list

        output = tf.reduce_mean(input_, [0], name=self.name)
        return output

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

class Concatenate(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        if type(x_list[0]) is list:
            input_ = smart_concat(x_list, -1)
        else:
            input_ = x_list

        output = tf.concat(input_, -1, name=self.name)
        return output

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

class MergeSharedUnique(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        output = []
        for i in range(self.n_branches):
            output.append(smart_concat(x_list[i], -1, self.name+'_'+str(i+1)))
        return output

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

class Conv2D(Layer):
    def __init__(self, filters_list, kernel_size, n_branches, name,
            shared_filters=0, strides=1, padding='valid'):
        super().__init__(name, n_branches)

        assert n_branches == len(filters_list), 'n_branches != len(filters_list)'
        self.filters_list = filters_list
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.shared_filters = shared_filters
        self.shared_branch = None

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        self.branches = []
        output_list = []

        if self.shared_filters == 0:
            for i in range(self.n_branches):
                layer = L.Conv2D(self.filters_list[i], self.kernel_size,
                    self.name+'_vb'+str(i+1), strides=self.strides,
                    padding=self.padding)

                if type(x_list[i]) is list:
                    input_ = smart_concat(x_list[i], -1)
                else:
                    input_ = x_list[i]

                x_out = layer(input_)
                self.branches.append(layer)
                output_list.append(x_out)

            return output_list

        # For efficiency, only apply computation to shared_in ONCE
        self.shared_branch = L.Conv2D(self.shared_filters, self.kernel_size,
            self.name+'_shared_to_shared',strides=self.strides,padding=self.padding)

        for i in range(self.n_branches):
            assert self.filters_list[i] >= self.shared_filters, 'filters < shared_filters'
            unique_filters = self.filters_list[i] - self.shared_filters

            # Operations to build the rest of the layer
            shared_to_unique = L.Conv2D(unique_filters, self.kernel_size,
                self.name+'_vb'+str(i+1)+'_shared_to_unique',strides=self.strides,
                padding=self.padding)
            unique_to_shared = L.Conv2D(self.shared_filters, self.kernel_size,
                self.name+'_vb'+str(i+1)+'_unique_to_shared',strides=self.strides,
                padding=self.padding)
            unique_to_unique = L.Conv2D(unique_filters, self.kernel_size,
                self.name+'_vb'+str(i+1)+'_unique_to_unique',strides=self.strides,
                padding=self.padding)

            if type(x_list[i]) is list:
                shared_in = x_list[i][0]
                unique_in = x_list[i][1]

                shared_out = smart_add(self.shared_branch(shared_in),
                    unique_to_shared(unique_in))
                unique_out = shared_to_unique(shared_in)+unique_to_unique(unique_in)
            else:
                shared_out = self.shared_branch(x_list[i])
                unique_out = shared_to_unique(x_list[i])

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
            weights = [self.shared_branch.f, self.shared_branch.b]

        # Get unique weights
        if self.shared_filters == 0:
            for layer in self.branches:
                weights += [layer.f, layer.b]
        else:
            for layer in self.branches:
                weights += [layer.shared_to_unique.f, layer.shared_to_unique.b,
                    layer.unique_to_shared.f, layer.unique_to_shared.b,
                    layer.unique_to_unique.f, layer.unique_to_unique.b]

        return weights

    def get_config(self, eval_weights=False):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'shared_filters':self.shared_filters, 'output_shapes':self.output_shapes,
            'filters_list':self.filters_list, 'weights':self.get_weights(eval_weights)}
        return config

class Pooling(Layer):
    def __init__(self, name, n_branches, layer):
        super().__init__(name, n_branches)
        self.layer = layer

    @Layer.set_output_shapes
    @Layer.expand_input
    def __call__(self, x_list):
        output_list = []

        for i in range(self.n_branches):
            if type(x_list[i]) is list:
                shared_out = self.layer(x_list[i][0])
                unique_out = self.layer(x_list[i][1])
                output_list.append([shared_out, unique_out])
            else:
                output_list.append(self.layer(x_list[i]))

        return output_list

class AveragePooling2D(Pooling):
    def __init__(self, pool_size, n_branches, name, strides=None, padding='valid'):
        layer = L.AveragePooling2D(pool_size, name, strides, padding)

        super().__init__(name, n_branches, layer)

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes, 'pool_size':self.pool_size,
            'strides':self.strides, 'padding':self.padding}
        return config

class GlobalAveragePooling2D(Pooling):
    def __init__(self, n_branches, name):
        layer = L.GlobalAveragePooling2D(name)
        super().__init__(name, n_branches, layer)

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
