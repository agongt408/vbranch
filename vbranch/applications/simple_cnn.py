from ..slim import *

import tensorflow as tf

def default(input_tensor, num_classes, *layers_spec, name=None):
    # NOTE: input_tensor is a single Tensor object

    ip = L.Input(input_tensor)

    x = ip
    for i, filters in enumerate(layers_spec):
        for l in range(2):
            x = Conv2D(x, filters, 3, name='conv2d_%d_%d' % (i+1, l+1))
            x = BatchNormalization(x, name='bn_%d_%d' % (i + 1, l+1))
            x = Activation(x, 'relu', name='relu_%d_%d' % (i+1, l+1))

        if i < len(layers_spec) - 1:
            x = AveragePooling2D(x, (2,2), name='avg_pool2d_'+str(i + 1))
        else:
            x = GlobalAveragePooling2D(x, name='global_avg_pool2d')

            # Embedding layers
            x = Dense(x, layers_spec[-1], name='fc1')
            x = BatchNormalization(x, name='bn_fc1')
            x = Activation(x, 'relu', name='relu_fc1')
            x = Dense(x, num_classes, name='output')

    return Model(input_tensor, x, name=name)

def vbranch_default(inputs, final_spec, *layers_spec, name=None):
    """
    Args:
        - inputs: list of Tensors
        - final_spec: tuple of (num_classes, shared_units)
        - layers_spec: tuple(s) of (filters_list, shared_filters)
        - branches: number of branches
    """
    assert type(inputs) is list

    ip = Input(inputs)

    x = ip
    for i, (filters, shared) in enumerate(layers_spec):
        for l in range(2):
            x = Conv2D(x, filters, 3, name='conv2d_%d_%d'%(i+1,l+1), shared=shared)
            x = BatchNormalization(x, name='bn_%d_%d' % (i+1, l+1))
            x = Activation(x, 'relu', name='relu_%d_%d' % (i+1, l+1))

        if i < len(layers_spec) - 1:
            x = AveragePooling2D(x, (2,2),name='avg_pool2d_'+str(i+1))
        else:
            x = GlobalAveragePooling2D(x, name='global_avg_pool2d')

            # Embedding layers
            x = Dense(x, layers_spec[-1][0], name='fc1', shared=layers_spec[-1][1])
            x = BatchNormalization(x, name='bn_fc1')
            x = Activation(x, 'relu', name='relu_fc1')

            x = Dense(x,final_spec[0],shared=final_spec[1],name='output',merge=True)

    return ModelVB(ip, x, name=name)
