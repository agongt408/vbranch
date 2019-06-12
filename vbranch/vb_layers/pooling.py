from .core import Layer
from .. import layers as L

class Pooling(Layer):
    def __init__(self, name, n_branches, layer, merge=False):
        super().__init__(name, n_branches, merge)
        self.layer = layer

    @Layer.call
    def __call__(self, x):
        output_list = []

        for i in range(self.n_branches):
            if type(x[i]) is list:
                shared_out = self.layer(x[i][0])
                unique_out = self.layer(x[i][1])
                output_list.append([shared_out, unique_out])
            else:
                output_list.append(self.layer(x[i]))

        return output_list

class AveragePooling2D(Pooling):
    def __init__(self,pool_size,n_branches,name,strides=None,padding='valid', 
            merge=False):

        layer = L.AveragePooling2D(pool_size, name, strides, padding)

        super().__init__(name, n_branches, layer, merge)

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def get_config(self):
        config = {'name':self.name, 'n_branches':self.n_branches,
            'output_shapes':self.output_shapes, 'pool_size':self.pool_size,
            'strides':self.strides, 'padding':self.padding}
        return config

class GlobalAveragePooling2D(Pooling):
    def __init__(self, n_branches, name, merge=False):
        layer = L.GlobalAveragePooling2D(name)
        super().__init__(name, n_branches, layer, merge)
