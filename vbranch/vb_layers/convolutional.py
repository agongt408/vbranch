from .. import layers as L
from .core import Layer, CrossWeights, EmptyOutput
from ..utils.layer import *

import numpy as np

class Conv2D(Layer):
    def __init__(self, filters_list, kernel_size, n_branches, name,
            shared_filters=0, strides=1, padding='valid', use_bias=True, merge=False):
        super().__init__(name, n_branches, merge)

        assert n_branches == len(filters_list),'n_branches != len(filters_list)'
        assert type(kernel_size) in [int, list, tuple], kernel_size
        self.filters_list = filters_list
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.shared_filters = shared_filters
        self.shared_branch = None
        self.use_bias = use_bias

    @Layer.call
    def __call__(self, x):
        self.branches = []
        output_list = []

        # Calculate `fan_in` for weight initialization
        if type(self.kernel_size) is int:
            receptive_field_size = self.kernel_size**2
        else:
            receptive_field_size = self.kernel_size[0] * self.kernel_size[1]

        fan_in = get_fan_in(x[0]) * receptive_field_size
        fan_out = int(np.mean(self.filters_list)) * receptive_field_size
        assert all([fan_out == units for units in self.filters_list])

        if self.shared_filters > 0:
            # For efficiency, only apply computation to shared_in ONCE
            self.shared_branch = L.Conv2D(self.shared_filters, self.kernel_size,
                'shared_to_shared', strides=self.strides, padding=self.padding,
                fan_in=fan_in, fan_out=fan_out, use_bias=self.use_bias)

        for i in range(self.n_branches):
            if self.shared_filters == 0:
                layer = L.Conv2D(self.filters_list[i], self.kernel_size,
                    'vb'+str(i+1), strides=self.strides, padding=self.padding,
                    fan_in=fan_in, fan_out=fan_out, use_bias=self.use_bias)

                x_out = layer(smart_concat(x[i], -1))
                self.branches.append(layer)
                output_list.append([EmptyOutput(), x_out])

            elif self.shared_filters == fan_out:
                x_out = self.shared_branch(smart_concat(x[i], -1))
                output_list.append([x_out, EmptyOutput()])

            else:
                # Operations to build the rest of the layer
                unique_filters = self.filters_list[i] - self.shared_filters
                shared_to_unique = L.Conv2D(unique_filters, self.kernel_size,
                    'vb'+str(i+1)+'_shared_to_unique', strides=self.strides,
                    padding=self.padding, fan_in=fan_in, fan_out=fan_out,
                    use_bias=self.use_bias)
                unique_to_shared = L.Conv2D(self.shared_filters, self.kernel_size,
                    'vb'+str(i+1)+ '_unique_to_shared', strides=self.strides,
                    padding=self.padding, fan_in=fan_in, fan_out=fan_out,
                    use_bias=self.use_bias)
                unique_to_unique = L.Conv2D(unique_filters, self.kernel_size,
                    'vb'+str(i+1) + '_unique_to_unique', strides=self.strides,
                    padding=self.padding, fan_in=fan_in, fan_out=fan_out,
                    use_bias=self.use_bias)

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
            'shared_filters':self.shared_filters,
            'output_shapes':self.output_shapes,
            'filters_list':self.filters_list,
            'weights':self.get_weights(sess)}
        return config

class ZeroPadding2D(Layer):
    def __init__(self, padding, n_branches, name, merge=False):
        super().__init__(name, n_branches, merge)
        self.padding = padding

    @Layer.call
    def __call__(self, x):
        pad_layer = L.ZeroPadding2D(self.name, self.padding)
        output_list = []

        for i in range(self.n_branches):
            if type(x[i]) is list:
                shared_out = pad_layer(x[i][0])
                unique_out = pad_layer(x[i][1])
                output_list.append([shared_out, unique_out])
            else:
                output_list.append(pad_layer(x[i]))

        return output_list
