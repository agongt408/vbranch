from ..slim import *
from ..utils.layer import get_shape
from ..engine import Model, ModelVB
from .weight_utils import load_weights_densenet
from ..vb_layers.core import VBOutput

from tensorflow import Tensor
import pickle

def dense_block(x, blocks, name, shared, growth_rate=32):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name+'_block'+str(i+1), shared=shared)
    return x

def transition_block(x, reduction, name, shared):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    if isinstance(x, VBOutput):
        if type(x[0]) is list:
            channels = 0
            for x_ in x[0]:
                shape = get_shape(x_)
                if shape != []:
                    channels += shape[-1]
        else:
            channels = get_shape(x[0])[-1]
    else:
        channels = get_shape(x)[-1]

    x = BatchNormalization(x, epsilon=1.001e-5, name=name + '_bn')
    x = Activation(x, 'relu', name=name + '_relu')
    x = Conv2D(x, int(channels * reduction), 1, use_bias=False,
            name=name + '_conv', shared=shared)
    x = AveragePooling2D(x, 2, strides=2, name=name + '_pool')
    return x

def conv_block(x, growth_rate, name, shared):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    x1 = BatchNormalization(x, epsilon=1.001e-5, name=name + '_0_bn')
    x1 = Activation(x1, 'relu', name=name + '_0_relu')
    x1 = Conv2D(x1, 4 * growth_rate, 1, use_bias=False,
            name=name + '_1_conv', shared=shared)
    x1 = BatchNormalization(x1, epsilon=1.001e-5, name=name + '_1_bn')
    x1 = Activation(x1, 'relu', name=name + '_1_relu')
    x1 = Conv2D(x1, growth_rate, 3, padding='same', use_bias=False,
               name=name + '_2_conv', shared=shared)
    x = Concatenate([x, x1], name=name + '_concat')
    return x

def DenseNet4(blocks, inputs,
        weights='imagenet', classes=1000,
        shared_frac=None, name=None, # subsample=True,
        shared_frac_blocks=None):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `Input()`)
            to use as image input for the model
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified."""

    # Create model
    # assert isinstance(inputs, Tensor) or type(inputs) in [list, tuple]
    if type(inputs) in [list, tuple] and shared_frac_blocks is None:
        assert shared_frac is not None
        assert shared_frac >= 0 and shared_frac <= 1
        shared_frac = float(shared_frac)

    if shared_frac_blocks is None:
        shared_frac_blocks = [shared_frac] * 4

    img_input = Input(inputs)

    # if subsample:
    x = ZeroPadding2D(img_input, padding=(3, 3))
    x = Conv2D(x, 64, 7, strides=2, use_bias=False, name='conv1/conv',
            # shared=shared_frac)
            shared=shared_frac_blocks[0])
    x = BatchNormalization(x, epsilon=1.001e-5, name='conv1/bn')
    x = Activation(x, 'relu', name='conv1/relu')
    x = ZeroPadding2D(x, padding=(1, 1))
    x = MaxPooling2D(x, 3, strides=2, name='pool1')
    # else:
    #     x = ZeroPadding2D(img_input, padding=(1, 1))
    #     x = Conv2D(x, 64, 3, use_bias=False, name='conv1/conv',
    #             # shared=shared_frac)
    #             shared=shared_frac_blocks[0])

    x = dense_block(x, blocks[0], name='conv2', shared=shared_frac_blocks[0])
    x = transition_block(x, 0.5, name='pool2', shared=shared_frac_blocks[0])
    x = dense_block(x, blocks[1], name='conv3', shared=shared_frac_blocks[1])
    x = transition_block(x, 0.5, name='pool3', shared=shared_frac_blocks[1])
    x = dense_block(x, blocks[2], name='conv4', shared=shared_frac_blocks[2])
    x = transition_block(x, 0.5, name='pool4', shared=shared_frac_blocks[2])
    x = dense_block(x, blocks[3], name='conv5', shared=shared_frac_blocks[3])

    x = BatchNormalization(x, epsilon=1.001e-5, name='bn')
    x = Activation(x, 'relu', name='relu')

    with tf.variable_scope('top'):
        x = GlobalAveragePooling2D(x, name='avg_pool')
        # x = Dense(x, 1024, shared=shared_frac)
        # x = BatchNormalization(x)
        # x = Activation(x, 'relu')
        x = Dense(x, classes, name='output', merge=True)

    if type(inputs) is list:
        model = ModelVB(img_input, x, name=name)
    else:
        model = Model(img_input, x, name=name)

    # Load weights.
    if weights == 'imagenet':
        print('Loading weights for DenseNet121...')
        with open('weights/densenet121.pickle', 'rb') as pickle_in:
            weights = pickle.load(pickle_in)
        assign_ops = load_weights_densenet(model, weights)
        return model, assign_ops

    return model

def DenseNet121(inputs, classes,
        weights=None, shared_frac=None,
        name=None, subsample=True, shared_frac_blocks=None):
    return DenseNet4([6, 12, 24, 16], inputs, weights, classes,
        shared_frac, name, shared_frac_blocks)

# def DenseNet169(inputs, classes, weights=None, shared_frac=None, name=None):
#     return DenseNet([6, 12, 32, 32], inputs, weights, classes, shared_frac, name)
#
# def DenseNet201(inputs, classes, weights=None, shared_frac=None, name=None):
#     return DenseNet([6, 12, 48, 32], inputs, weights, classes, shared_frac, name)

def DenseNet3(depth, growth_rate, inputs,
        classes=10, shared_frac=None, name=None,
        shared_frac_blocks=None):
    """Instantiates the DenseNet architecture for Cifar10 with BC.
    # Arguments
        input_tensor: optional Keras tensor
            (i.e. output of `Input()`)
            to use as image input for the model
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified."""

    # Create model
    # assert isinstance(inputs, Tensor) or type(inputs) in [list, tuple]
    if type(inputs) in [list, tuple] and shared_frac_blocks is None:
        assert shared_frac is not None
        assert shared_frac >= 0 and shared_frac <= 1
        shared_frac = float(shared_frac)

    if shared_frac_blocks is None:
        shared_frac_blocks = [shared_frac] * 3

    # No BC -> divide by 3
    # BC -> divide by 6
    n_layers = (depth - 4) // 6

    img_input = Input(inputs)

    x = ZeroPadding2D(img_input, padding=(1, 1))
    x = Conv2D(x, growth_rate*2, 3, use_bias=False, name='conv1/conv',
            shared=shared_frac_blocks[0])

    x = dense_block(x, n_layers, 'conv2', shared_frac_blocks[0], growth_rate)
    x = transition_block(x, 0.5, 'pool2', shared_frac_blocks[0])
    x = dense_block(x, n_layers, 'conv3', shared_frac_blocks[1], growth_rate)
    x = transition_block(x, 0.5, 'pool3', shared_frac_blocks[1])
    x = dense_block(x, n_layers, 'conv4', shared_frac_blocks[2], growth_rate)

    x = BatchNormalization(x, epsilon=1.001e-5, name='bn')
    x = Activation(x, 'relu', name='relu')

    with tf.variable_scope('top'):
        x = GlobalAveragePooling2D(x, name='avg_pool')
        x = Dense(x, classes, name='output', merge=True, shared=shared_frac_blocks[-1])

    if type(inputs) is list:
        model = ModelVB(img_input, x, name=name)
    else:
        model = Model(img_input, x, name=name)

    return model
