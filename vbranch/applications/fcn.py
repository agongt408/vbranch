from ..slim import *
from tensorflow import Tensor

def base(input_, *layers_spec, name=None, shared_frac=None):
    assert isinstance(input_, Tensor) or type(input_) is list
    vb_mode = (type(input_) is list)
    if vb_mode:
        assert shared_frac is not None
        assert shared_frac >= 0 and shared_frac <= 1
        if shared_frac > 0:
            assert type(shared_frac) is float

    ip = Input(input_)

    x = ip
    for i, spec in enumerate(layers_spec[:-1]):
        if type(spec) is int:
            units = spec
            shared = shared_frac
        elif type(spec) is tuple:
            units = spec[0]
            shared = spec[1]
        else:
            raise ValueError('invalid layers spec:', spec)

        x = Dense(x, units, shared=shared, name='fc'+str(i+1))
        x = BatchNormalization(x, name='bn'+str(i+1))
        x = Activation(x, 'relu', name='relu'+str(i+1))

    # if type(layers_spec[-1]) is int:
    #     x = Dense(x, layers_spec[-1], name='output')
    # elif type(layers_spec[-1]) is tuple:
    #     x = Dense(x, layers_spec[-1][0], shared=layers_spec[-1][1],
    #             name='output', merge=True)
    # else:
    #     raise ValueError('invalid final_spec:', layers_spec[-1])

    if vb_mode:
        if type(layers_spec[-1]) is int:
            final_units = layers_spec[-1]
            shared = shared_frac
        elif type(layers_spec[-1]) is tuple:
            final_units, shared = layers_spec[-1]
        else:
            raise ValueError('invalid final_spec:', layers_spec[-1])

        x = Dense(x,final_units,shared=shared,merge=True, name='output')
    else:
        x = Dense(x, layers_spec[-1], name='output')

    if type(input_) is list:
        return ModelVB(ip, x, name=name)

    return Model(ip, x, name=name)

def SimpleFCNv1(inputs, classes, name=None, shared_frac=None):
    return base(inputs, 512, classes, name=name, shared_frac=shared_frac)

def SimpleFCNv2(inputs, classes, name=None, shared_frac=None):
    return base(inputs, 512, 256, classes, name=name, shared_frac=shared_frac)

def SimpleFCNv3(inputs, classes, name=None, shared_frac=None):
    return base(inputs, 512, 512, classes, name=name,shared_frac=shared_frac)

def SimpleFCNv4(inputs, classes, name=None, shared_frac=None):
    return base(inputs, 512, 512, 512, classes, name=name,shared_frac=shared_frac)
