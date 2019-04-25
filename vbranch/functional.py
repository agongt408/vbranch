# Declare models using input and output tensors
# Utilizes _vb_history attribute

from . import layers as L
from . import vb_layers as VBL
from . import utils

import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, inputs, outputs):
        self.input = inputs
        self.output = outputs

        if type(inputs) is not list:
            inputs = [inputs]
        if type(outputs) is not list:
            outputs = [outputs]

        self.layers = _map_graph(inputs, outputs)

    def summary(self):
        model_summary = utils.Summary('i','Layer name','Output shape',
            'Parameters','Num param')

        total_num_params = 0

        # Input spec
        model_summary.add('','Input',utils.get_shape_as_str(self.input),'','')

        for i, l in enumerate(self.layers):
            config = l.get_config()
            name = config['name']
            output_shape = utils.shape_to_str(config['output_shape'])

            num_params = 0
            param_shapes = ''
            if 'weights' in config.keys():
                for weight in config['weights']:
                    num_params += utils.get_num_params(weight)
                    param_shapes += utils.get_shape_as_str(weight) + ' '

            model_summary.add(i, name, output_shape, param_shapes, num_params)
            total_num_params += num_params

        model_summary.show()

        print('Total parameters: {:d}'.format(total_num_params))

def _map_graph(inputs, outputs):
    """
    Validates a network's topology and gather its layers and nodes.
    Args:
        - inputs: List of input tensors
        - outputs: List of outputs tensors
    """

    def build_map(tensor, layers, inputs):
        if tensor in inputs:
            # End recursion if reached inputs
            return

        assert hasattr(tensor, '_vb_history'), \
            AttributeError('tensor {} was not created by custom layer')

        layer = tensor._vb_history

        for x in layer._inbound_tensors:
            build_map(x, layers, inputs)

        # Add layer of tensor to layers list (only if unique)
        if not layer in layers:
            layers.append(layer)

    layers = []
    for x in outputs:
        build_map(x, layers, inputs)

    return layers

def simple_fcn(input_tensor, *layers_spec):
    # NOTE: input_tensor is a single Tensor object

    x = input_tensor

    for i, units in enumerate(layers_spec):
        x = L.Dense(units, 'fc'+str(i + 1))(x)
        x = L.BatchNormalization('bn'+str(i + 1))(x)

        activation_name = 'relu'+str(i+1) if i<len(layers_spec)-1 else 'output'
        x = L.Activation('relu', activation_name)(x)

    model = Model(input_tensor, x)

    return model

def simple_cnn(input_tensor, num_classes, *layers_spec):
    # NOTE: input_tensor is a single Tensor object

    x = input_tensor

    for i, filters in enumerate(layers_spec):
        for l in range(2):
            x = L.Conv2D(filters, 3, 'conv2d_%d_%d' % (i+1, l+1))(x)
            x = L.BatchNormalization('bn_%d_%d' % (i + 1, l+1))(x)
            x = L.Activation('relu', 'relu_%d_%d' % (i+1, l+1))(x)

        if i < len(layers_spec) - 1:
            x = L.AveragePooling2D((2,2), 'avg_pool2d_'+str(i + 1))(x)
        else:
            x = L.GlobalAveragePooling2D('global_avg_pool2d')(x)

            # Embedding layers
            x = L.Dense(layers_spec[-1], 'fc1')(x)
            x = L.BatchNormalization('bn_fc1')(x)
            x = L.Activation('relu', 'relu_fc1')(x)
            x = L.Dense(num_classes, 'output')(x)

    model = Model(input_tensor, x)

    return model
