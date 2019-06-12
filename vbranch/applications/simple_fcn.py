from ..slim import *

def default(input_tensor, *layers_spec, name=None):
    # NOTE: input_tensor is a single Tensor object

    # Wrap input tensor in Input layer in order to retrieve inbound name
    # when printing model summary
    ip = Input(input_tensor)

    x = ip
    for i, units in enumerate(layers_spec[:-1]):
        x = Dense(x, units, name='fc'+str(i+1))
        x = BatchNormalization(x, name='bn'+str(i+1))
        x = Activation(x, 'relu', name='relu'+str(i+1))

    x = Dense(x, layers_spec[-1], name='output')

    return Model(ip, x, name=name)

def vbranch_default(inputs, *layers_spec, name=None):
    # NOTE: inputs is a list of Tensors
    assert type(inputs) is list

    ip = Input(inputs)

    x = ip
    for i, (units, shared) in enumerate(layers_spec[:-1]):
        x = Dense(x, units, shared=shared, name='fc'+str(i+1))
        print(x)
        x = BatchNormalization(x, name='bn'+str(i+1))
        x = Activation(x, 'relu', name='relu'+str(i+1))

    x = Dense(x, layers_spec[-1][0], shared=layers_spec[-1][1],
            name='output', merge=True)

    return ModelVB(ip, x, name=name)
