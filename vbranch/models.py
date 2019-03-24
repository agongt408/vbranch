# Declare models using tensorflow

from .layers import *

import tensorflow as tf
import numpy as np

class Sequential(object):
    def __init__(self, input_tensor):
        self.input = input_tensor
        self.layers = []
        self.output = input_tensor

    def add(self, layer):
        self.layers.append(layer)
        self.output = layer(self.output)

    def summary(self):
        def print_line(s1, s2, s3, s4):
            print('{:<4}{:<20}{:>20}{:>20}'.format(str(s1),str(s2),str(s3),str(s4)))
            print('-' * 64)

        print_line('i', 'Layer name', 'Output shape', 'Parameters')
        total_num_params = 0

        # Input spec
        print_line('', 'Input', self.input.get_shape().as_list(), '')

        for i, l in enumerate(self.layers):
            config = l.get_config()
            name = config['name']
            output_shape = config['output_shape']

            num_params = 0
            if 'weights' in config.keys():
                for weight in config['weights']:
                    num_params += np.prod(weight.shape)

            print_line(i, name, output_shape, num_params)
            total_num_params += num_params

        print('Parameters: {:d}'.format(total_num_params))

def simple_fcn(input_tensor, *layers_spec):
    model = Sequential(input_tensor)

    for i, units in enumerate(layers_spec):
        model.add(Dense(units, 'fc'+str(i + 1)))
        model.add(BatchNormalization('bn'+str(i + 1)))
        model.add(Activation('relu', 'relu'+str(i + 1)))

    return model

def simple_cnn(input_tensor, num_classes, *layers_spec):
    model = Sequential(input_tensor)

    for i, filters in enumerate(layers_spec):
        for l in range(2):
            model.add(Conv2D(filters, 3, 'conv2d_%d_%d' % (i+1, l+1)))
            model.add(BatchNormalization('bn_%d_%d' % (i + 1, l+1)))
            model.add(Activation('relu', 'relu_%d_%d' % (i+1, l+1)))

        if i < len(layers_spec) - 1:
            model.add(AveragePooling2D((2,2), 'avg_pool2d_'+str(i + 1)))
        else:
            model.add(GlobalAveragePooling2D('global_avg_pool2d'))
            model.add(Dense(num_classes, 'pred'))

    return model
