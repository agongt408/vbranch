from keras.layers import *
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model

import numpy as np

class ModelConfig:
    '''Model must be sequential'''

    def __init__(self, layers=[], input_layers=[], output_layers=[],
                name='Model', weights=[]):
        self.layers = layers
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.name = name
        self.init_weights = weights

    def from_model(self, model):
        self.layers = self.get_layers(model)
        self.input_layers = self.get_input_layers(model)
        self.output_layers = self.get_output_layers(model)
        self.init_weights = model.get_weights()

    def get_input_layers(self, model):
        inputs = []
        for ip in model.inputs:
            name = ip.name
            inputs.append(name[:name.index(':')])
        return inputs

    def get_output_layers(self, model):
        outputs = []
        for ip in model.outputs:
            name = ip.name
            outputs.append(name[:name.index(':')])
        return outputs

    def get_layers(self, model):
        layers = []

        for l in model.get_config()['layers']:
            l_class = l['class_name']
            config = l['config']

            try:
                inbound_nodes = np.array([[[ip[0], 0] for ip in ib_n] \
                    for ib_n in l['inbound_nodes']])
            except:
                inbound_nodes = l['inbound_nodes']

            name = l['name']

            op = eval(l_class).from_config(config)

            layers.append(Layer(l_class, op, inbound_nodes, name))

        return layers

    def set_input(self, op, name='input'):
        self.input_layers = [name]
        prev_input = self.input_layers[0]

        if self.layers[0].class_name == 'InputLayer':
            self.layers[0] = Layer('InputLayer', op, [[]], name)

            for l in range(len(self.layers)):
                for ip in range(len(self.layers[l].inbound_nodes)):
                    if self.layers[l].inbound_nodes[ip] == prev_input:
                        self.layers[l].inbound_nodes[ip] = [[name, 0]]
        else:
            self.layers.insert(0, Layer('InputLayer', op, [[]], name))

    def add_input(self, op, n, name=None):
        if n > len(self.layers) - 1:
            raise ValueError, "n must be less than number of layers"

        if name is None:
            name = 'ip_' + str(n + 1)

        layer = Layer('Input', op, [[[]]], name)
        self.layers.insert(n + 1, layer)

        self.input_layers.append(name)

    def set_output(self, n):
        '''sets the nth layer as the output, discard all following layers'''
        for i in range(len(self.layers) - 1 - n):
            self.layers.pop()
        self.output_layers = [self.layers[-1]]

    def add_layer(self, class_name, op, n, inbound_nodes=None,
                    outbound_nodes=None, name=None):
        '''Adds layer after nth layer in model'''
        if n > len(self.layers) - 1:
            raise ValueError, "n must be less than number of layers"

        if inbound_nodes is None:
            prev_layer = self.layers[n]
        if outbound_nodes is None:
            next_layer = self.layers[n + 1] \
                if n < len(self.layers) - 1 else None

        if name is None:
            name = 'l_' + str(n + 1)

        if inbound_nodes:
            layer = Layer(class_name, op, np.array(inbound_nodes), name)
        else:
            layer = Layer(class_name, op,
                np.array([[[self.layers[n].name, 0]]]), name)
        self.layers.insert(n + 1, layer)

        if outbound_nodes:
            # Loop through all layers
            for l in range(len(self.layers)):
                # If found layer in outbound_nodes
                if self.layers[l].name in outbound_nodes:
                    # Loop through inbound nodes of added layer
                    for i in range(len(inbound_nodes)):
                        for j in range(len(inbound_nodes[i])):
                            # Loop through inbound nodes of
                            # matching layer in outbound nodes
                            if (inbound_nodes[i][j] == \
                                self.layers[l].inbound_nodes).all(1).any():
                                for ib_n in range(len(
                                    self.layers[l].inbound_nodes)):
                                    for ip in range(len(self.layers[l].\
                                        inbound_nodes[ib_n])):
                                        if inbound_nodes[i][j][0] == \
                                            self.layers[l].inbound_nodes[ib_n][ip][0]:
                                            self.layers[l].inbound_nodes[ib_n][ip] \
                                                = np.array([name, 0])
                            else:
                                l_ib_n = self.layers[l].inbound_nodes.tolist()
                                l_ib_n.append(np.array([[name, 0]]))
                                self.layers[l].inbound_nodes = np.array(l_ib_n)

        if outbound_nodes is None and next_layer is None:
            self.output_layers = [name]

    def reconstruct_model(self, name='Model', init_weights=True,
                            output_idx=None):
        x_list = [{'tensor' : [self.layers[0].op.get_output_at(0)],
                    'name' : self.layers[0].name}]

        for layer in self.layers[1:]:
            # Initialize inbound_tensor_list to have correct size
            inbound_tensor_list = [[] for i in layer.inbound_nodes]

            if layer.class_name != 'Input':
                # Loop through all inbound nodes
                for ib_n in range(len(layer.inbound_nodes)):
                    for ip in range(len(layer.inbound_nodes[ib_n])):
                        # Find matching input
                        for l in range(len(self.layers)):
                            if layer.inbound_nodes[ib_n][ip][0] == \
                                self.layers[l].name:
                                # If inbound node from InputLayer
                                if layer.inbound_nodes[0][0][0].\
                                    find('input') > -1:
                                    inbound_tensor_list[ib_n].\
                                        append(x_list[0]['tensor'][ib_n])

                                elif layer.inbound_nodes[0][0][1] > -1:
                                    inbound_tensor_list[ib_n].\
                                        append(x_list[l]['tensor']\
                                    [int(layer.inbound_nodes[ib_n][ip][1])])

            print layer.name, inbound_tensor_list
            # print 'here'

            if len(inbound_tensor_list) == 1:
                if len(inbound_tensor_list[0]) == 0:
                    x_list.append({'tensor' : [layer.op],
                                    'name' : layer.name})
                elif len(inbound_tensor_list[0]) == 1:
                    x_list.append({'tensor' : [layer.op(
                                        inbound_tensor_list[0][0])],
                                    'name' : layer.name})
                else:
                    x_list.append({'tensor' : [layer.op(
                                        inbound_tensor_list[0])],
                                    'name' : layer.name})
            else:
                if len(inbound_tensor_list[0]) == 1:
                    # print layer.name, inbound_tensor_list
                    x_list.append(
                        {'tensor' : [layer.op(inbound_tensor_list[b][0]) \
                            for b in range(len(inbound_tensor_list))],
                        'name' : layer.name})
                else:
                    # print layer.name, inbound_tensor_list
                    x_list.append(
                        {'tensor' : [layer.op(inbound_tensor_list[b]) \
                            for b in range(len(inbound_tensor_list))],
                        'name' : layer.name})

        if len(self.input_layers) == 1:
            if output_idx:
                output_arr = []
                for idx in output_idx:
                    output_arr.append(x_list[idx]['tensor'][0])
                # print output_arr
                model = Model(x_list[0]['tensor'], output_arr, name=name)
            else:
                model = Model(x_list[0]['tensor'],
                    x_list[-1]['tensor'], name=name)
        else:
            input_arr = []
            for ip in range(len(self.input_layers)):
                for l in range(len(x_list)):
                    if x_list[l]['name'] == self.input_layers[ip]:
                        input_arr.append(x_list[l]['tensor'][0])

            if output_idx:
                output_arr = []
                for idx in output_idx:
                    output_arr.append(x_list[idx]['tensor'][0])
                # print output_arr
                model = Model(input_arr, output_arr, name=name)
            else:
                model = Model(input_arr, x_list[-1]['tensor'], name=name)

        if init_weights:
            model.set_weights(self.init_weights)

        return model, x_list


class Layer:
    def __init__(self, class_name, op, inbound_nodes, name):
        self.class_name = class_name
        self.op = op
        self.inbound_nodes = inbound_nodes
        self.name = name
