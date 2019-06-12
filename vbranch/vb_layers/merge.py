from .. import layers as L
from .core import Layer, smart_add, smart_concat

import tensorflow as tf

class Add(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.call
    def __call__(self, x):
        # x: list of VBOutput objects
        assert type(x) is list, 'x is not a list of VBOutput objects'

        self.branches = []
        output_list = []

        for b in range(self.branches):
            input_list = []
            layer = L.Add('vb'+str(i+1))

            for i in range(len(x)):
                if type(x[i][b]) is list:
                    input_ = smart_concat(x[i][b], -1)
                else:
                    input_ = x[i][b]

                input_list.append(input_)

            x_out = layer(input_list)
            self.branches(layer)
            output_list.append(x_out)

        return output_list
