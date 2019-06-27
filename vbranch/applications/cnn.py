from ..slim import *
from tensorflow import Tensor

def SimpleCNNSmall(inputs, classes, name=None, shared_frac=None):
    return base(inputs, (classes, 0), 16, 32, name=name, shared_frac=shared_frac)

def SimpleCNNLarge(inputs, classes, name=None, shared_frac=None):
    return base(inputs, (classes, 0), 32, 64, 128, 256, name=name,
        shared_frac=shared_frac)

def base(input_, final_spec, *layers_spec, name=None, shared_frac=None):
    """
    Create SimpleCNN model; dynamically determine what type of model to use
    (i.e., Model or ModelVB)
    Args:
        - final_spec: scalar of number of units; or tuple of (units, shared);
        if scalar, default to `shared_frac`
        - layers_spec: list of layer sizes of list of (layer size, shared)
        tuples; shared can be either float (fraction) or int (units); if list
        of scalars, each layer will default to `shared_frac`
        - name: model name
        - shared_frac: fraction of each layer's parameters to share; only
        used if creating ModelVB
    Returns:
        - Model or ModelVB instance
    """

    assert isinstance(input_, Tensor) or type(input_) is list
    vb_mode = (type(input_) is list)
    if vb_mode:
        assert shared_frac is not None
        assert shared_frac >= 0 and shared_frac <= 1

    ip = Input(input_)

    x = ip
    for i, spec in enumerate(layers_spec):
        if type(spec) is int:
            filters = spec
            shared = shared_frac
        elif type(spec) is tuple:
            filters = spec[0]
            shared = spec[1]
        else:
            raise ValueError('invalid layers spec:', spec)

        for l in range(2):
            x = Conv2D(x, filters, 3, name='conv2d_%d_%d'%(i+1,l+1),
                    shared=shared)
            x = BatchNormalization(x, name='bn_%d_%d' % (i+1, l+1))
            x = Activation(x, 'relu', name='relu_%d_%d' % (i+1, l+1))

        if i < len(layers_spec) - 1:
            x = AveragePooling2D(x, (2,2),name='avg_pool2d_'+str(i+1))
        else:
            x = GlobalAveragePooling2D(x, name='global_avg_pool2d')
            x = Dense(x, filters, name='fc1', shared=shared)
            x = BatchNormalization(x, name='bn_fc1')
            x = Activation(x, 'relu', name='relu_fc1')

            if type(final_spec) is int:
                final_units = final_spec
                shared = shared_frac
            elif type(final_spec) is tuple:
                final_units, shared = final_spec
            else:
                raise ValueError('invalid final_spec:', final_spec)

            if vb_mode:
                x = Dense(x,final_units,shared=shared,merge=True, name='output')
            else:
                x = Dense(x, final_units, name='output')

    if type(input_) is list:
        return ModelVB(ip, x, name=name)

    return Model(ip, x, name=name)
