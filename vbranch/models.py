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
        def print_line(i, name, output_shape, param_shapes, num_params):
            str_f = '{:<4}'.format(str(i))
            str_f += '{:<20}'.format(str(name))
            str_f += '{:<20}'.format(str(output_shape))
            str_f += '{:<30}'.format(str(param_shapes))
            str_f += '{:<10}'.format(str(num_params))

            print(str_f)
            print('-' * len(str_f))

        print_line('i','Layer name','Output shape','Parameters','Num param')
        total_num_params = 0

        # Input spec
        print_line('', 'Input', str(self.input.get_shape().as_list()).\
            replace(' ', ''), '', '')

        for i, l in enumerate(self.layers):
            config = l.get_config()
            name = config['name']
            output_shape = str(config['output_shape']).replace(' ', '')

            num_params = 0
            param_shapes = ''
            if 'weights' in config.keys():
                for weight in config['weights']:
                    num_params += np.prod(weight.shape)
                    param_shapes += str(weight.shape).replace(' ', '') + ' '

            print_line(i, name, output_shape, param_shapes, num_params)
            total_num_params += num_params

        print('Total parameters: {:d}'.format(total_num_params))

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
            model.add(Dense(num_classes, 'output'))

    return model
