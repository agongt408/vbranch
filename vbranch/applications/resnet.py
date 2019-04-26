from .. import layers as L
from .. import vb_layers as VBL

from ..engine import Model, ModelVB

def res_cnn(input_tensor, num_classes, *layers_spec):
    def _res_block(x, n_layers, n_filters, kernel, name, reduce=True):
        if reduce:
            strides = 2
        else:
            strides = 1

        x = _res_layer(x, n_filters, kernel, strides, name+'_1', shortcut=True)

        for i in range(n_layers - 1):
            x = _res_layer(x, n_filters, kernel, 1, name+'_'+str(i+2),
                shortcut=False)

        return x

    def _res_layer(x, n_filters, kernel, strides, name, shortcut=False):
        # Pre-activation residual layer
        pre = x
        x = L.BatchNormalization(name + '_bn')(x)
        x = L.Activation('relu', name + '_relu')(x)
        x = L.Conv2D(n_filters, kernel, name+'_conv', strides=strides,
            padding='same')(x)

        # Add shortcut if channels do not match
        if shortcut:
            pre = L.Conv2D(n_filters,1,name+'_conv_short', strides=strides,
                padding="same")(pre)

        output = L.Add(name + '_add')([pre, x])

        return output

    n_layers = 3
    kernel = 3

    # Wrap input tensor in Input layer in order to retrieve inbound name
    # when printing model summary
    ip = L.Input(input_tensor)

    # Pre-convolutional layer
    x = L.Conv2D(16, 5, 'conv2d_pre', padding='same')(ip)

    for i, n_filters in enumerate(layers_spec):
        reduce = (i != 0)
        x = _res_block(x, n_layers, n_filters, kernel, 'res_%d'%(i+1), reduce)

    x = L.GlobalAveragePooling2D('global_avg_pool2d')(x)

    x = L.Dense(layers_spec[-1], 'fc1')(x)
    x = L.BatchNormalization('bn_fc1')(x)
    x = L.Activation('relu', 'relu_fc1')(x)
    x = L.Dense(num_classes, 'output')(x)

    model = Model(ip, x)

    return model
