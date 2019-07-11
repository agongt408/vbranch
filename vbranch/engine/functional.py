# Declare models using input and output tensors
# Utilizes _vb_history attribute

from ..utils.generic import Summary, get_shape_as_str, shape_to_str, get_num_params
from ..vb_layers import VBOutput

import tensorflow as tf
import numpy as np

class Network(object):
    def __init__(self, inputs, outputs, name=None):
        self.input = inputs
        self.output = outputs
        self.name = name

        if type(inputs) is not list:
            inputs = [inputs]
        if type(outputs) is not list:
            outputs = [outputs]

        self.layers = _map_graph(inputs, outputs)

    def summary(self):
        model_summary = Summary('i','Layer name','Output shape','Parameters',
            'Num param', 'Inbound')

        total_num_params = 0

        # Input spec
        input_shape = get_shape_as_str(self.input)
        model_summary.add('', 'Input', input_shape, '', '', '')

        for i, l in enumerate(self.layers):
            config = l.get_config()

            name = '{} ({})'.format(config['name'], l.__class__.__name__)
            # Limit name length
            name = name[:30]

            output_shape = shape_to_str(config['output_shape'])
            inbound_names = [ip._vb_history.name for ip in l._inbound_tensors]

            num_params = 0
            param_shapes = ''
            if 'weights' in config.keys():
                for weight in config['weights']:
                    num_params += get_num_params(weight)
                    param_shapes += get_shape_as_str(weight) + ' '
                param_shapes = param_shapes.strip()

            for n, in_name in enumerate(inbound_names):
                if n == 0:
                    model_summary.add(i,name,output_shape,param_shapes,
                        num_params,in_name,show_line=(n==len(inbound_names)-1))
                else:
                    model_summary.add('', '', '', '', '', in_name,
                        show_line=(n==len(inbound_names)-1))

            total_num_params += num_params

        model_summary.show()

        print('Total parameters: {:d}'.format(total_num_params))

    def count_parameters(self):
        total_num_params = 0

        for i, l in enumerate(self.layers):
            config = l.get_config()
            num_params = 0
            if 'weights' in config.keys():
                for weight in config['weights']:
                    num_params += get_num_params(weight)

            total_num_params += num_params

        return total_num_params

class NetworkVB(object):
    def __init__(self, inputs, outputs, name=None):
        self.input = inputs
        self.output = outputs
        self.name = name

        if type(inputs) is not list:
            inputs = [inputs]
        if type(outputs) is not list:
            outputs = [outputs]

        self.layers = _map_graph(inputs, outputs)

        # Calculate number of branches
        self.n_branches = self.layers[0].n_branches

        for l in self.layers:
            # Verify number of branches
            assert l.n_branches == self.n_branches, \
                'model has {} branches, but layer {} has {} branches'.\
                    format(self.n_branches, l.name, l.n_branches)

    def summary(self):
        model_summary = Summary('i','Layer name','Output shape','Num param',
            'Inbound')

        total_num_params = 0

        # Input spec
        if type(self.input) is list or isinstance(self.input, VBOutput):
            for ip in self.input:
                input_shape = get_shape_as_str(ip)
                model_summary.add('', 'Input', input_shape, '', '')
        else:
            input_shape = get_shape_as_str(self.input)
            model_summary.add('', 'Input', input_shape, '', '')

        for i, l in enumerate(self.layers):
            config = l.get_config()
            name = '{} ({})'.format(config['name'], l.__class__.__name__)
            # Limit name length
            name = name[:30]

            num_params = 0
            if 'weights' in config.keys():
                for weight in config['weights']:
                    if weight != []:
                        num_params += get_num_params(weight)
            total_num_params += num_params

            inbound_names = [ip._vb_history.name for ip in l._inbound_tensors]

            num_outputs = len(config['output_shapes'])
            for b in range(num_outputs):
                output_shape = ''
                for shape in config['output_shapes'][b]:
                    output_shape += shape_to_str(shape) + ' '
                output_shape = output_shape.strip()

                if b < len(inbound_names):
                    in_name = inbound_names[b]
                else:
                    in_name = ''

                if b == 0:
                    model_summary.add(i, name, output_shape,num_params,in_name,
                        show_line=b==num_outputs-1)
                else:
                    model_summary.add('', '', output_shape, '', in_name,
                        show_line=b==num_outputs-1)

        model_summary.show()

        print('Total parameters: {:d}'.format(total_num_params))

    def count_parameters(self):
        total_num_params = 0

        for i, l in enumerate(self.layers):
            config = l.get_config()
            name = config['name']

            num_params = 0
            if 'weights' in config.keys():
                for weight in config['weights']:
                    if weight != []:
                        num_params += get_num_params(weight)
            total_num_params += num_params

        return total_num_params

def _map_graph(inputs, outputs):
    """
    Validates a network's topology and gather its layers and nodes.
    Args:
        - inputs: List of input tensors
        - outputs: List of outputs tensors
    """
    layers = []
    def build_map(tensor, inputs):
        # End recursion if reached inputs
        if tensor in inputs:
            return

        # If VBOutput is from collated input tensors
        if isinstance(tensor, VBOutput):
            if all([(x in inputs) for x in tensor]):
                return

        assert hasattr(tensor, '_vb_history'), \
            AttributeError('tensor {} was not created by custom layer'.\
                format(tensor))

        l = tensor._vb_history
        if l not in layers:
            for x in l._inbound_tensors:
                build_map(x, inputs)
            layers.append(l)

    for x in outputs:
        build_map(x, inputs)

    return layers
