from .. import layers as L
from .. import vb_layers as VBL

from ..engine import Sequential, SequentialVB, Model, ModelVB

def default(input_tensor, *layers_spec):
    # Model created using Functional approach
    # NOTE: input_tensor is a single Tensor object

    # Wrap input tensor in Input layer in order to retrieve inbound name
    # when printing model summary
    ip = L.Input(input_tensor)
    x = ip

    for i, units in enumerate(layers_spec):
        x = L.Dense(units, 'fc'+str(i + 1))(x)
        x = L.BatchNormalization('bn'+str(i + 1))(x)

        activation_name = 'relu'+str(i+1) if i<len(layers_spec)-1 else 'output'
        x = L.Activation('relu', activation_name)(x)

    model = Model(ip, x)

    return model

def vbranch_default(inputs, *layers_spec, branches=1):
    # NOTE: inputs can be single Tensor or list of Tensors

    ip = VBL.Input(inputs)
    x = ip

    for i, (units_list, shared_units) in enumerate(layers_spec):
        x = VBL.Dense(units_list,branches,'fc'+str(i + 1), shared_units)(x)
        x = VBL.BatchNormalization(branches, 'bn'+str(i + 1))(x)

        activation_name = 'relu'+str(i + 1) if (i < len(layers_spec) - 1 or \
            layers_spec[-1][-1] > 0) else 'output'
        x = VBL.Activation('relu', branches, activation_name)(x)

    # If using shared params
    if layers_spec[-1][-1] > 0:
        x = VBL.MergeSharedUnique(branches, 'output')(x)

    model = ModelVB(ip, x)

    return model

def seq_version(input_tensor, *layers_spec):
    # NOTE: input_tensor is a single Tensor object

    model = Sequential(input_tensor)

    for i, units in enumerate(layers_spec):
        model.add(L.Dense(units, 'fc'+str(i + 1)))
        model.add(L.BatchNormalization('bn'+str(i + 1)))

        activation_name = 'relu'+str(i+1) if i<len(layers_spec)-1 else 'output'
        model.add(L.Activation('relu', activation_name))

    return model

def vbranch_seq(inputs, *layers_spec, branches=1):
    # NOTE: inputs can be single Tensor or list of Tensors
    model = SequentialVB(inputs)

    for i, (units_list, shared_units) in enumerate(layers_spec):
        model.add(VBL.Dense(units_list,branches,'fc'+str(i + 1), shared_units))
        model.add(VBL.BatchNormalization(branches, 'bn'+str(i + 1)))

        activation_name = 'relu'+str(i + 1) if (i < len(layers_spec) - 1 or \
            layers_spec[-1][-1] > 0) else 'output'
        model.add(VBL.Activation('relu', branches, activation_name))

    # If using shared params
    if layers_spec[-1][-1] > 0:
        model.add(VBL.MergeSharedUnique(branches, 'output'))

    return model
