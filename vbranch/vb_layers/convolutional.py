from .. import layers as L
from .core import Layer, CrossWeights, EmptyOutput
from ..utils.generic import smart_add, smart_concat

import numpy as np

class Conv2D(Layer):
    def __init__(self, filters_list, kernel_size, n_branches, name,
            shared_filters=0, strides=1, padding='valid', merge=False):
        super().__init__(name, n_branches, merge)

        assert n_branches == len(filters_list),'n_branches != len(filters_list)'
        self.filters_list = filters_list
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.shared_filters = shared_filters
        self.shared_branch = None

    @Layer.call
    def __call__(self, x):
        self.branches = []
        output_list = []

        if self.shared_filters == 0:
            for i in range(self.n_branches):
                layer = L.Conv2D(self.filters_list[i], self.kernel_size,
                    'vb'+str(i+1), strides=self.strides, padding=self.padding)

                if type(x[i]) is list:
                    input_ = smart_concat(x[i], -1)
                else:
                    input_ = x[i]

                x_out = layer(input_)
                self.branches.append(layer)
                output_list.append(x_out)

            return output_list

        # Calculate `fan_in` for weight initialization
        if type(x[0]) is list:
            fan_in_list = []
            for x_ in x[0]:
                if not isinstance(x_, EmptyOutput):
                    fan_in_list.append(x_.get_shape().as_list()[-1])
            fan_in = np.sum(fan_in_list)
        else:
            fan_in = x[0].get_shape().as_list()[-1]

        fan_out = int(np.mean(self.filters_list))
        assert all([fan_out == units for units in self.filters_list])

        # For efficiency, only apply computation to shared_in ONCE
        self.shared_branch = L.Conv2D(self.shared_filters, self.kernel_size,
            'shared_to_shared', strides=self.strides, padding=self.padding,
            fan_in=fan_in, fan_out=fan_out)

        for i in range(self.n_branches):
            assert self.filters_list[i] >= self.shared_filters, \
                'filters < shared_filters'
            unique_filters = self.filters_list[i] - self.shared_filters

            # Operations to build the rest of the layer
            shared_to_unique = L.Conv2D(unique_filters, self.kernel_size,
                'vb'+str(i+1)+'_shared_to_unique', strides=self.strides,
                padding=self.padding, fan_in=fan_in, fan_out=fan_out)
            unique_to_shared = L.Conv2D(self.shared_filters, self.kernel_size,
                'vb'+str(i+1)+ '_unique_to_shared', strides=self.strides,
                padding=self.padding, fan_in=fan_in, fan_out=fan_out)
            unique_to_unique = L.Conv2D(unique_filters, self.kernel_size,
                'vb'+str(i+1) + '_unique_to_unique', strides=self.strides,
                padding=self.padding, fan_in=fan_in, fan_out=fan_out)

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

            cross_weights = CrossWeights(shared_to_unique=shared_to_unique,
                unique_to_shared=unique_to_shared,
                unique_to_unique=unique_to_unique)

            self.branches.append(cross_weights)
            output_list.append([shared_out, unique_out])

        return output_list

    def get_config(self, sess=None):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'shared_filters':self.shared_filters,
            'output_shapes':self.output_shapes,
            'filters_list':self.filters_list,
            'weights':self.get_weights(sess)}
        return config
