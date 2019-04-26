# Declare models sequentially

from . import utils

import tensorflow as tf
import numpy as np

class Sequential(object):
    def __init__(self, input_tensor):
        """
        Args:
            - input_tensor: single Tensor"""

        self.input = input_tensor
        self.layers = []
        self.output = input_tensor

    def add(self, layer):
        self.layers.append(layer)
        self.output = layer(self.output)

    def summary(self):
        model_summary = utils.Summary('i', 'Layer name', 'Output shape',
            'Parameters', 'Num param')

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

class SequentialVB(object):
    def __init__(self, inputs):
        """
        Args:
            - inputs: single Tensor or list of Tensors"""

        self.input = inputs
        self.layers = []
        self.output = inputs

    def add(self, layer):
        self.layers.append(layer)
        self.output = layer(self.output)

    def summary(self):
        model_summary = utils.Summary('i', 'Layer name', 'Output shapes',
            'Num param')

        total_num_params = 0

        # Input spec
        if type(self.input) is list:
            for ip in self.input:
                model_summary.add('', 'Input', utils.get_shape_as_str(ip), '')
        else:
            model_summary.add('','Input',utils.get_shape_as_str(self.input),'')

        for i, l in enumerate(self.layers):
            config = l.get_config()
            name = config['name']

            num_params = 0
            if 'weights' in config.keys():
                for weight in config['weights']:
                    if weight != []:
                        num_params += utils.get_num_params(weight)
            total_num_params += num_params

            num_outputs = len(config['output_shapes'])
            for b in range(num_outputs):
                output_shape = ''
                for shape in config['output_shapes'][b]:
                    output_shape += utils.shape_to_str(shape) + ' '

                if b == 0:
                    model_summary.add(i, name, output_shape, num_params,
                        show_line=b==num_outputs-1)
                else:
                    model_summary.add('', '', output_shape, '',
                        show_line=b==num_outputs-1)

        model_summary.show()

        print('Total parameters: {:d}'.format(total_num_params))