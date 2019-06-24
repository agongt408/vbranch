from ..slim import *
from ..utils import TFSessionGrow
from .. import layers

from tensorflow import Tensor
import pickle
import pkgutil

def base(input_, classes, layer_spec, kernel_spec, filter_spec, name=None,
        subsample_initial_block=True, shared_frac=None):
    """
    Args:
        - inputs: Tensor object
        - classes: number of classes or units in embedding
        - layer_spec: tuple of n_layers per block
        - kernel_spec: tuple of kernel sizes per layer
        - filter_spec: list of tuples of filters per layer per block
        - name: model name
        - subsample_initial_block: if true, apply strides=2 and downsampling
    """

    def conv_block(inputs, kernels, filters, stage, block, strides,
            shortcut, shared):

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        i = 0
        x = inputs
        for kernel, filter in zip(kernels, filters):
            if i == 0:
                s = strides
            else:
                s = 1
            label = '2'+chr(97+i)

            x = Conv2D(x, filter, kernel, strides=s, padding='same',
                    name=conv_name_base + label, shared=shared)
            x = BatchNormalization(x, name=bn_name_base + label)

            if i < len(kernels) - 1:
                x = Activation(x, 'relu')

            i += 1

        if shortcut:
            short = Conv2D(inputs, filters[-1], 1, strides=strides,
                        padding='same', name=conv_name_base+'1', shared=shared)
            short = BatchNormalization(short, name=bn_name_base+'1')
            x = Add([x, short])
        else:
            x = Add([x, inputs])

        x = Activation(x, 'relu')
        return x

    # Create model
    assert isinstance(input_, Tensor) or type(input_) is list
    vb_mode = (type(input_) is list)
    if vb_mode:
        assert shared_frac is not None
        assert shared_frac >= 0 and shared_frac <= 1

    ip = Input(input_)

    # Initial convolution
    if subsample_initial_block:
        initial_kernel = 7
        initial_strides = 2
    else:
        initial_kernel = 3
        initial_strides = 1
    initial_filters = filter_spec[0][0]

    x = Conv2D(ip, initial_filters, initial_kernel, strides=initial_strides,
            name='conv1', padding='same', shared=shared_frac)

    if subsample_initial_block:
        x = BatchNormalization(x, name='bn_conv1')
        x = Activation(x, 'relu')
        x = MaxPooling2D(x, (3, 3), strides=(2, 2), padding='same')

    for i, n_layers in enumerate(layer_spec):
        if i == 0:
            strides = 1
        else:
            strides = 2

        for l in range(n_layers):
            if l == 0 and i > 0:
                strides = 2
            else:
                strides = 1

            x = conv_block(x, kernel_spec, filter_spec[i], stage=i+2,
                    block=chr(97+l), strides=strides, shortcut=l==0,
                    shared=shared_frac)

    x = GlobalAveragePooling2D(x, name='avg_pool')
    # x = Dense(x, layer_spec[-1][0], name='fc1', shared=shared_frac)
    # x = BatchNormalization(x, name='bn_fc1')
    # x = Activation(x, 'relu', name='relu_fc1')
    # Don't share parameters for last layers
    x = Dense(x, classes, name='output')

    if type(input_) is list:
        return ModelVB(ip, x, name=name)

    return Model(ip, x, name=name)

def ResNet18(inputs, classes, name=None, shared_frac=None):
    layer_spec = (2, 2, 2, 2)
    kernel_spec = (3, 3)
    filter_spec = [(64, 64), (128, 128), (256, 256), (512, 512)]
    return base(inputs, classes, layer_spec, kernel_spec, filter_spec,
        name=name, shared_frac=shared_frac)

def ResNet34(inputs, classes, name=None, shared_frac=None):
    layer_spec = (3, 4, 6, 3)
    kernel_spec = (3, 3)
    filter_spec = [(64, 64), (128, 128), (256, 256), (512, 512)]
    return base(inputs, classes, layer_spec, kernel_spec, filter_spec,
        name=name, shared_frac=shared_frac)

def ResNet50(inputs, classes, name=None, shared_frac=None, weights=None):
    layer_spec = (3, 4, 6, 3)
    kernel_spec = (1, 3, 1)
    filter_spec = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    model = base(inputs, classes, layer_spec, kernel_spec, filter_spec,
        name=name, shared_frac=shared_frac)

    if weights == 'imagenet' and isinstance(inputs, Tensor):
        print('Loading weights for ResNet50...')

        # Load weights
        with open('weights/resnet50.pickle', 'rb') as pickle_in:
            weights = pickle.load(pickle_in)

        assign_ops = []
        for layer in model.layers:
            name = layer.name
            if isinstance(layer, layers.Conv2D):
                assign_ops.append(tf.assign(layer.f, weights[name]['filter']))
                assign_ops.append(tf.assign(layer.b, weights[name]['bias']))
            elif isinstance(layer, layers.BatchNormalization):
                assign_ops.append(tf.assign(layer.scale, weights[name]['scale']))
                assign_ops.append(tf.assign(layer.beta, weights[name]['beta']))

        return model, assign_ops

    return model

def ResNet101(inputs, classes, name=None, shared_frac=None):
    layers_spec = (3, 4, 23, 3)
    kernel_spec = (1, 3, 1)
    filter_spec = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    return base(inputs, classes, layer_spec, kernel_spec, filter_spec,
        name=name, shared_frac=shared_frac)

def ResNet152(inputs, classes, name=None, shared_frac=None):
    layers_spec = (3, 8, 36, 3)
    kernel_spec = (1, 3, 1)
    filter_spec = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    return base(inputs, classes, layer_spec, kernel_spec, filter_spec,
        name=name, shared_frac=shared_frac)
