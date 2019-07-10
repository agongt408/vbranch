from ..slim import *
from ..utils import TFSessionGrow
from .weight_utils import load_weights

from tensorflow import Tensor
from numpy import ndarray
import pickle


def ResNet50(inputs, classes, name=None, shared_frac=None, weights=None):
    """
    Construct ResNet50 model with optional weights
    Args (see `base`):
        - weights: if imagenet, use pretrained weights, else use randomly
        initialized weights"""

    layer_spec = (3, 4, 6, 3)
    kernel_spec = (1, 3, 1)
    filter_spec = [
        (64, 64, 256), (128, 128, 512),
        (256, 256, 1024), (512, 512, 2048)
    ]

    model = base(inputs, classes, layer_spec, kernel_spec, filter_spec,
        name=name, shared_frac=shared_frac)

    if weights == 'imagenet':
        print('Loading weights for ResNet50...')
        with open('weights/resnet50.pickle', 'rb') as pickle_in:
            weights = pickle.load(pickle_in)
        assign_ops = load_weights(model, weights)
        return model, assign_ops

    return model

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

def base(input_, classes, layer_spec, kernel_spec, filter_spec, name=None,
        shared_frac=None):
    """
    Construct ResNet model with additional FC layers
    Args:
        - inputs: Tensor object
        - classes: number of classes or units in embedding
        - layer_spec: tuple of n_layers per block
        - kernel_spec: tuple of kernel sizes per layer
        - filter_spec: list of tuples of filters per layer per block
        - name: model name
    Returns:
        - Model or ModelVB instance
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
    x = ZeroPadding2D(ip, padding=(3,3), name='conv1_pad')
    x = Conv2D(x, 64, (7,7), strides=(2,2), name='conv1', padding='valid',
            shared=shared_frac)
    x = BatchNormalization(x, name='bn_conv1')
    x = Activation(x, 'relu')
    x = ZeroPadding2D(x, padding=(1,1), name='pool1_pad')
    x = MaxPooling2D(x, (3, 3), strides=(2, 2))

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
    x = Dense(x, filter_spec[-1][-1] // 2, shared=shared_frac)
    x = BatchNormalization(x)
    x = Activation(x, 'relu')
    x = Dense(x, classes, name='output')

    # x = GlobalAveragePooling2D(x, name='output')
    if type(input_) is list:
        return ModelVB(ip, x, name=name)

    return Model(ip, x, name=name)
