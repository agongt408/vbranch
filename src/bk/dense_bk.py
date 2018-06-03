'''DenseNet models for Keras.
# Reference
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation]
    (https://arxiv.org/pdf/1611.09326.pdf)
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, Lambda
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.layer_utils import convert_dense_weights_data_format
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
import keras.backend as K

import tensorflow as tf
import numpy as np
import os

from subpixel import SubPixelUpscaling

DENSENET_121_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/' + \
                            'releases/download/v3.0/DenseNet-BC-121-32.h5'

WEIGHTS_ROOT = '/home/albert/github/tensorflow/src/weights/'
if not os.path.exists(WEIGHTS_ROOT):
    WEIGHTS_ROOT = '/home/ubuntu/albert/src/weights/'

def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017 # scale values

    return x


def DenseNet(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12,
                nb_filter=-1, nb_layers_per_block=-1, bottleneck=False,
                reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
                include_top=True, weights=None, input_tensor=None,
                output_dim=128, blocks=[], cam_dim=None, fc1=1024,
                diagnostic=False):
    '''
    Instantiate the DenseNet architecture, optionally loading weights
    pre-trainedon CIFAR-10.

    Args:
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(32, 32, 3)` (with `channels_last` dim ordering)
            or `(3, 32, 32)` (with `channels_first` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 8.
            E.g. `(200, 200, 3)` would be one valid value.
        depth: number or layers in the DenseNet
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. -1 indicates initial
            number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the network depth.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        bottleneck: flag to add bottleneck blocks in between dense blocks
        reduction: reduction factor of transition blocks.
            Note : reduction value is inverted to compute compression.
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization) or
            'imagenet' (pre-training on ImageNet)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        output_dim: length of final embedding
        blocks: list such that all blocks after the smallest number in
            'blocks' is included in the final model; if -1 in 'blocks',
            subsample block included
        cam_dim: tuple representing dimensions of CAM with shape
            (rows, columns)
        fc1: dim of first FC layer

    Returns:
        A Keras model instance.
    '''

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    input_var, pred_arr = __create_dense_net(output_dim, img_input,
        depth, nb_dense_block, growth_rate, nb_filter, nb_layers_per_block,
        bottleneck, reduction, dropout_rate, weight_decay, blocks,
        cam_dim, weights, include_top, fc1, diagnostic)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    '''
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input'''
    # Create model.
    model = Model(input_var, pred_arr, name='densenet')

    return model

def DenseNetBlockImageNet121(input_shape=None, input_tensor=None,
                            blocks=[-1,0,1,2,3],
                            cam_dim=None, weights='imagenet',
                            include_top=True, output_dim=128,
                            dropout_rate=0.0, fc1=1024,
                            diagnostic=False):
    return DenseNet(input_shape=input_shape, depth=121, nb_dense_block=4,
                    growth_rate=32, nb_filter=64,
                    nb_layers_per_block=[6, 12, 24, 16], bottleneck=True,
                    reduction=0.5, dropout_rate=dropout_rate,
                    weight_decay=1e-4, include_top=include_top,
                    weights=weights, input_tensor=input_tensor,
                    blocks=blocks, cam_dim=cam_dim,
                    output_dim=output_dim, fc1=fc1,
                    diagnostic=diagnostic)

def DenseNetBlockImageNet161(input_shape=None, input_tensor=None,
                            blocks=[-1,0,1,2,3],
                            cam_dim=None, weights='imagenet',
                            include_top=True, output_dim=128,
                            dropout_rate=0.0, fc1=1024,
                            diagnostic=False):
    return DenseNet(input_shape=input_shape, depth=161, nb_dense_block=4,
                    growth_rate=48, nb_filter=96,
                    nb_layers_per_block=[6, 12, 36, 24], bottleneck=True,
                    reduction=0.5, dropout_rate=dropout_rate,
                    weight_decay=1e-4, include_top=include_top,
                    weights=weights, input_tensor=input_tensor,
                    blocks=blocks, cam_dim=cam_dim,
                    output_dim=output_dim, fc1=fc1, diagnostic=diagnostic)


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None,
                weight_decay=1e-4, reg=None, batch_norm=None, diagnostic=False):
    '''
    Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout

    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns:
        keras tensor with batch_norm, relu and convolution2d added
        (optional bottleneck)
    '''

    # print('conv: ' + str(mask))

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = batch_norm[0](ip) if batch_norm else \
            BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                diagnostic=diagnostic)(ip)
    x = Activation('relu')(x)

    if bottleneck:
        # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
        inter_channel = nb_filter * 4

        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal',
                    padding='same', use_bias=False,
                    kernel_regularizer=(reg if reg else l2(weight_decay)))(x)
        x = batch_norm[1](x) if batch_norm else \
                BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                    diagnostic=diagnostic)(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal',
                padding='same', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False,
                    dropout_rate=None, weight_decay=1e-4,
                    grow_nb_filters=True, regularizers=None,
                    batch_norm_list=None, diagnostic=False):
    '''
    Build a dense_block where the output of each conv_block is fed to
    subsequent ones.

    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to
            the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along
            with the actual output

    Returns:
        keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]
    mask_input_list = []

    if regularizers:
        assert len(regularizers) == nb_layers , \
            'length of regularizers list must be same as nb_layers'

    # 2 batch norm layers per conv block
    if batch_norm_list:
        assert len(batch_norm_list) == 2 * nb_layers, \
            'length of batch_norm_list must be same as 2*nb_layers = ' + \
            str(2*nb_layers) + ' but is %d' % len(batch_norm_list)

    for i in range(nb_layers):
        if batch_norm_list:
            init_batch_norm = [batch_norm_list[2*i], batch_norm_list[2*i+1]]

        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate,
                            weight_decay,
                            regularizers[i] if regularizers else None,
                            init_batch_norm if batch_norm_list else None,
                            diagnostic=diagnostic)
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0,
                        weight_decay=1e-4, reg=None, diagnostic=False):
    '''
    Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout
    and Maxpooling2D.

    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns:
        keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    # print('transition: ' + str(mask))

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                            diagnostic=diagnostic)(ip)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * compression), (1, 1),
                kernel_initializer='he_normal', padding='same',
                use_bias=False,
                kernel_regularizer=(reg if reg else l2(weight_decay)))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def __create_dense_net(output_dim, img_input, depth=40, nb_dense_block=3,
                        growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False,
                       reduction=0.0, dropout_rate=None,
                       weight_decay=1e-4, blocks=[], cam_dim=None,
                       weights='imagenet', include_top=True,
                       fc1=1024, diagnostic=False):
    '''
    Build the DenseNet model

    Args:
        output_dim: length of final embedding layer, e.g., 128
        img_input: tuple of shape (rows, columns, channels)
        depth: number or layers
        nb_dense_block: number of dense blocks (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates
            initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the depth of the network.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks.
            Note: reduction value is inverted to compute compression
        dropout_rate: dropout rate, applied to last conv layer in each
            conv block
        weight_decay: weight decay rate, used for kernel regularizer
        blocks: list such that all blocks after the smallest number in
            'blocks' is included in the final model; if -1 in 'blocks',
            subsample block included
        cam_dim: tuple representing dimensions of CAM with shape
            (rows, columns)
        weights: None or 'imagenet'
        include_top: if True, include FC layers
        fc1: dim of first FC layer

    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, \
            'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block), \
            'If list, nb_layer is used as provided. ' \
            'Note that list size must be (nb_dense_block)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            assert (depth - 4) % 3 == 0, \
                'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            count = int((depth - 4) / 3)
            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    initial_filter = nb_filter

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    input_set = True
    input_var = img_input

    initial_kernel = (7, 7)
    initial_strides = (2, 2)

    x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal',
                padding='same', strides=initial_strides, use_bias=False,
                kernel_regularizer=l2(weight_decay))(img_input)

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                            diagnostic=diagnostic)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    y = x

    if -1 not in blocks:
        input_set = False
        y = None
        input_var = None

    print(blocks)

    blocks_exist = False
    for block_idx in range(nb_dense_block):
        if block_idx in blocks:
            blocks_exist = True
            break
    assert blocks_exist == True, 'must include at least one block'

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        if (block_idx in blocks) and (input_set == False):
            y = Input(shape=tuple(x.get_shape().as_list()[1:]))
            input_var = y
            input_set = True

        if input_set:
            y, nb_filter = __dense_block(y, nb_layers[block_idx],
                nb_filter, growth_rate, bottleneck=bottleneck,
                dropout_rate=dropout_rate, weight_decay=weight_decay,
                diagnostic=diagnostic)
            # add transition_block
            y = __transition_block(y, nb_filter, compression=compression,
                                    weight_decay=weight_decay,
                                    diagnostic=diagnostic)
            nb_filter = int(nb_filter * compression)
        else:
            x, nb_filter = __dense_block(x, nb_layers[block_idx],
                nb_filter, growth_rate, bottleneck=bottleneck,
                dropout_rate=dropout_rate, weight_decay=weight_decay,
                diagnostic=diagnostic)
            # add transition_block
            x = __transition_block(x, nb_filter, compression=compression,
                                    weight_decay=weight_decay,
                                    diagnostic=diagnostic)
            nb_filter = int(nb_filter * compression)

    if y is None:
        y = Input(shape=tuple(x.get_shape().as_list()[1:]))
        input_var = y
        input_set = True

    # The last dense_block does not have a transition_block
    y, nb_filter = __dense_block(y, final_nb_layer, nb_filter,
                                growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate,
                                 weight_decay=weight_decay,
                                 diagnostic=diagnostic)

    y = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                            diagnostic=diagnostic)(y)
    act = Activation('relu')(y)
    pred = GlobalAveragePooling2D()(act)

    model = Model(input_var, pred)

    # load weights
    if weights == 'imagenet':
        if (depth == 121) and (nb_dense_block == 4) and \
                (growth_rate == 32) and (initial_filter == 64) and \
                (bottleneck is True) and (reduction == 0.5):
            imagenet_weights = np.load(
                os.path.join(WEIGHTS_ROOT, 'DenseNetImageNet121-no-top.npy'))
            model.set_weights(imagenet_weights[-len(model.get_weights()):])

            print("Weights for the model were loaded successfully")

        if (depth == 161) and (nb_dense_block == 4) and \
                (growth_rate == 48) and (initial_filter == 96) and \
                (bottleneck is True) and (reduction == 0.5):
            imagenet_weights = np.load(
                os.path.join(WEIGHTS_ROOT, 'DenseNetImageNet161-no-top.npy'))
            model.set_weights(imagenet_weights[-len(model.get_weights()):])

            print("Weights for the model were loaded successfully")

    if include_top:
        pred = Dense(fc1)(model.output)
        pred = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                    diagnostic=diagnostic)(pred)
        pred = Activation('relu')(pred)

        pred = Dense(output_dim)(pred)

        if cam_dim is None:
            return input_var, pred
        else:
            cam_output = Lambda(cam, arguments={'cam_dim' : cam_dim})(act)
            return input_var, [pred, cam_output]
    else:
        return input_var, pred

def cam(im, cam_dim=(16,8)):
    cam_mean = K.mean(tf.image.resize_images(im, cam_dim), axis=3)
    cam_min = tf.tile(tf.reshape(K.min(
        cam_mean, axis=(1,2)), (-1, 1, 1)), (1,cam_dim[0],cam_dim[1]))
    cam_mean = cam_mean - cam_min
    cam_max = tf.tile(tf.reshape(K.max(
        cam_mean, axis=(1,2)), (-1, 1, 1)), (1,cam_dim[0],cam_dim[1]))
    cam_norm = tf.div(cam_mean, cam_max)
    return tf.reshape(cam_norm, (-1,) + cam_dim + (1,))
