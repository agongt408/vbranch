from ..slim import *
from tensorflow import Tensor

def base(input_, *layers_spec, name=None):
    assert isinstance(input_, Tensor) or type(input_) is list

    ip = Input(input_)

    x = ip
    for i, spec in enumerate(layers_spec[:-1]):
        if type(spec) is int:
            units = spec
            shared = None
        elif type(spec) is tuple:
            units = spec[0]
            shared = spec[1]
        else:
            raise ValueError('invalid layers spec:', spec)

        x = Dense(x, units, shared=shared, name='fc'+str(i+1))
        x = BatchNormalization(x, name='bn'+str(i+1))
        x = Activation(x, 'relu', name='relu'+str(i+1))

    if type(layers_spec[-1]) is int:
        x = Dense(x, layers_spec[-1], name='output')
    elif type(layers_spec[-1]) is tuple:
        x = Dense(x, layers_spec[-1][0], shared=layers_spec[-1][1],
                name='output', merge=True)
    else:
        raise ValueError('invalid final_spec:', layers_spec[-1])

    if type(input_) is list:
        return ModelVB(ip, x, name=name)

    return Model(ip, x, name=name)
