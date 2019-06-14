from ..slim import *
from tensorflow import Tensor

def base(input_, classes, *layers_spec, kernel_spec=[3,3], name=None,
        subsample_initial_block=True, shared_frac=None):
    """
    Args:
        - input_tensor: Tensor object
        - classes: number of classes or units in embedding
        - layers_spec: list of (n_filters, n_layers) for residual blocks
        - kernel_spec: list of kernel sizes for each residual layer
        - name: model name
        - subsample_initial_block: if true, apply strides=2 and downsampling
    """
    # Residual block (followed by pooling layer in model)
    def _res_block(x, n_layers, n_filters, kernel_spec, name, shared):
        strides = 1
        x = _res_layer(x, n_filters, kernel_spec, strides, name+'_1',
                shortcut=True, shared=shared)
        for i in range(n_layers - 1):
            x = _res_layer(x, n_filters, kernel_spec, 1, name+'_'+str(i+2),
                    shortcut=False, shared=shared)
        return x

    # Pre-activation residual layer
    def _res_layer(x, n_filters, kernel_spec, strides, name, shortcut=False,
            shared=None):
        pre = x
        for i, kernel in enumerate(kernel_spec):
            x = BatchNormalization(x, name=name + '_bn_'+str(i+1))
            x = Activation(x, 'relu', name=name + '_relu_'+str(i+1))
            x = Conv2D(x, n_filters, kernel, strides=strides, padding='same',
                name=name+'_conv_'+str(i+1), shared=shared)

        # Add shortcut if channels do not match
        if shortcut:
            pre = Conv2D(pre, n_filters,1, strides=strides,
                padding='same', name=name+'_conv_short', shared=shared)

        return Add([pre, x], name=name + '_add')

    # Create model
    assert isinstance(input_, Tensor) or type(input_) is list
    vb_mode = (type(input_) is list)
    if vb_mode:
        assert shared_frac is not None
        assert shared_frac >= 0 and shared_frac <= 1
        if shared_frac > 0:
            assert type(shared_frac) is float

    ip = Input(input_)

    # Initial convolution
    if subsample_initial_block:
        initial_kernel = 7
        initial_strides = 2
    else:
        initial_kernel = 3
        initial_strides = 1
    initial_filters = layers_spec[0][0]

    x = Conv2D(ip, initial_filters, initial_kernel, strides=initial_strides,
            name='pre_conv', padding='same', shared=shared_frac)

    if subsample_initial_block:
        x = BatchNormalization(x, name='pre_bn')
        x = Activation(x, 'relu', name='pre_relu')
        x = MaxPooling2D(x, (3, 3), strides=(2, 2), padding='same',
            name='pre_max_pool2d')

    for i, (n_filters, n_layers) in enumerate(layers_spec):
        x = _res_block(x, n_layers, n_filters, kernel_spec, 'res_%d'%(i+1),
                shared_frac)

        if i < len(layers_spec) - 1:
            x = AveragePooling2D(x, (2,2), name='avg_pool2d_'+str(i + 1))

    x = GlobalAveragePooling2D(x, name='global_pool2d')
    x = Dense(x, layers_spec[-1][0], name='fc1', shared=shared_frac)
    x = BatchNormalization(x, name='bn_fc1')
    x = Activation(x, 'relu', name='relu_fc1')
    # Don't share parameters for last layers
    x = Dense(x, classes, name='output')

    if type(input_) is list:
        return ModelVB(ip, x, name=name)

    return Model(ip, x, name=name)

def ResNet18(input_tensor, classes, name=None, shared_frac=None):
    layers_spec = [(64, 2), (128, 2), (256, 2), (512, 2)]
    kernel_spec = [3, 3]
    return base(input_tensor, classes, *layers_spec, kernel_spec=kernel_spec,
        name=name, shared_frac=shared_frac)

def ResNet34(input_tensor, classes, name=None, shared_frac=None):
    layers_spec = [(64, 3), (128, 4), (256, 6), (512, 3)]
    kernel_spec = [3, 3]
    return base(input_tensor, classes, *layers_spec, kernel_spec=kernel_spec,
        name=name, shared_frac=shared_frac)

def ResNet50(input_tensor, classes, name=None, shared_frac=None):
    layers_spec = [(64, 3), (128, 4), (256, 6), (512, 3)]
    kernel_spec = [1, 3, 1]
    return base(input_tensor, classes, *layers_spec, kernel_spec=kernel_spec,
        name=name, shared_frac=shared_frac)

def ResNet101(input_tensor, classes, name=None, shared_frac=None):
    layers_spec = [(64, 3), (128, 4), (256, 23), (512, 3)]
    kernel_spec = [1, 3, 1]
    return base(input_tensor, classes, *layers_spec, kernel_spec=kernel_spec,
        name=name, shared_frac=shared_frac)

def ResNet152(input_tensor, classes, name=None, shared_frac=None):
    layers_spec = [(64, 3), (128, 8), (256, 36), (512, 3)]
    kernel_spec = [1, 3, 1]
    return base(input_tensor, classes, *layers_spec, kernel_spec=kernel_spec,
        name=name, shared_frac=shared_frac)

# https://keras.io/applications/#resnet
# keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.resnet.ResNet101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.resnet.ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
