# Merge branches (specific to vbranch)

from .core import Layer, smart_concat

import tensorflow as tf

class Add(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.call
    def __call__(self, x):
        if type(x[0]) is list:
            input_ = smart_concat(x, -1)
        else:
            input_ = x

        output = tf.reduce_sum(input_, [0], name=self.name)
        return output

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

class Average(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.call
    def __call__(self, x):
        if type(x[0]) is list:
            input_ = smart_concat(x, -1)
        else:
            input_ = x

        output = tf.reduce_mean(input_, [0], name=self.name)
        return output

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

class Concatenate(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.call
    def __call__(self, x):
        if type(x[0]) is list:
            input_ = smart_concat(x, -1)
        else:
            input_ = x

        output = tf.concat(input_, -1, name=self.name)
        return output

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config

class MergeSharedUnique(Layer):
    def __init__(self, n_branches, name):
        super().__init__(name, n_branches)

    @Layer.call
    def __call__(self, x):
        output = []
        for i in range(self.n_branches):
            output.append(smart_concat(x[i], -1, self.name+'_'+str(i+1)))
        return output

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes}
        return config
