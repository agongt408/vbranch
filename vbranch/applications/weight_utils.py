from .. import layers
from .. import vb_layers
from ..engine.training import Model

import tensorflow as tf

def load_weights(model, weights):
    """
    Assign weights to designated model layer tensors for ResNet
    Args:
        - model: pre-constructed Model or ModelVB instance
        - weights: dict of layer names and weight values
            e.g., weights[name] = {'filter': np.ndarray, 'bias': np.ndarray}
    Returns:
        - list of assign ops
    """

    assign_ops = []

    if isinstance(model, Model):
        for layer in model.layers:
            # Only load weights up to the GlobalAveragePooling2D layer
            if isinstance(layer, layers.GlobalAveragePooling2D):
                break

            if isinstance(layer, layers.Conv2D):
                assign_ops.append(tf.assign(layer.f,
                    weights[layer.name]['filter']))
                if layer.use_bias:
                    assign_ops.append(tf.assign(layer.b,
                        weights[layer.name]['bias']))
            elif isinstance(layer, layers.BatchNormalization):
                assign_ops.append(tf.assign(layer.scale,
                    weights[layer.name]['scale']))
                assign_ops.append(tf.assign(layer.beta,
                    weights[layer.name]['beta']))
    else:
        for layer in model.layers:
            # Only load weights up to the GlobalAveragePooling2D layer
            if isinstance(layer, vb_layers.GlobalAveragePooling2D):
                break

            if isinstance(layer, vb_layers.Conv2D):
                assign_ops.append(get_vb_assign_conv(layer,
                    weights[layer.name]['filter'],
                    weights[layer.name]['bias'] if layer.use_bias else None))
            elif isinstance(layer, vb_layers.BatchNormalization):
                assign_ops.append(get_vb_assign_bn(layer,
                    weights[layer.name]['scale'],
                    weights[layer.name]['beta']))

    return assign_ops

def get_weight_partition_conv(name, filter_ref, filter_value,
        bias_ref=None, bias_value=None):
    """
    Args:
        - filter_ref, bias_ref: Tensors
        - filter_value, bias_value: np arrays
    """
    # Catch uncalled layers (no params)
    if filter_ref == [] or bias_ref == []:
        return []

    f_ref_shape = filter_ref.get_shape().as_list()
    f_value_shape = filter_value.shape

    if bias_ref is not None:
        b_ref_shape = bias_ref.get_shape().as_list()
        b_value_shape = bias_value.shape

    if name == 'shared_to_shared':
        filter_slice = filter_value[:, :, :f_ref_shape[-2], :f_ref_shape[-1]]
        if bias_ref is not None:
            bias_slice = bias_value[:b_ref_shape[0]]
    elif name == 'shared_to_unique':
        filter_slice = filter_value[:, :, :f_ref_shape[-2], -f_ref_shape[-1]:]
        if bias_ref is not None:
            bias_slice = bias_value[-b_ref_shape[0]:]
    elif name == 'unique_to_shared':
        filter_slice = filter_value[:, :, -f_ref_shape[-2]:, :f_ref_shape[-1]]
        if bias_ref is not None:
            bias_slice = bias_value[:b_ref_shape[0]]
    elif name == 'unique_to_unique':
        filter_slice = filter_value[:, :, -f_ref_shape[-2]:, -f_ref_shape[-1]:]
        if bias_ref is not None:
            bias_slice = bias_value[-b_ref_shape[0]:]
    else:
        raise ValueError('invalid name', name)

    if bias_ref is None:
        return tf.assign(filter_ref, filter_slice)

    return tf.assign(filter_ref, filter_slice), tf.assign(bias_ref, bias_slice)

def get_weight_partition_bn(name, scale_ref, beta_ref, scale_value, beta_value):
    # Catch uncalled layers (no params)
    if scale_ref == [] or beta_ref == []:
        return []

    s_ref_shape = scale_ref.get_shape().as_list()
    b_ref_shape = beta_ref.get_shape().as_list()

    if name == 'shared_to_shared':
        scale_slice = scale_value[:s_ref_shape[0]]
        beta_slice = beta_value[:b_ref_shape[0]]
    elif name == 'unique_to_unique':
        scale_slice = scale_value[-s_ref_shape[0]:]
        beta_slice = beta_value[-b_ref_shape[0]:]
    else:
        raise ValueError('invalid name', name)

    scale_assign = tf.assign(scale_ref, scale_slice)
    beta_assign = tf.assign(beta_ref, beta_slice)
    return scale_assign, beta_assign

def get_vb_assign_conv(layer, filter_value, bias_value=None):
    assign_ops = []

    if layer.shared_branch is None:
        for sub_layer in layer.branches:
            assign_ops.append(tf.assign(sub_layer.f, filter_value))

            if layer.use_bias:
                assign_ops.append(tf.assign(sub_layer.b, bias_value))
    else:
        if layer.use_bias:
            assign_ops.append(get_weight_partition_conv('shared_to_shared',
                layer.shared_branch.f, filter_value,
                layer.shared_branch.b, bias_value))
        else:
            assign_ops.append(get_weight_partition_conv('shared_to_shared',
                layer.shared_branch.f, filter_value))

        for branch in layer.branches:
            # Extract CrossWeights namedtuple
            for name, sub_layer in branch._asdict().items():
                if layer.use_bias:
                    assign_ops.append(get_weight_partition_conv(name,
                        sub_layer.f, filter_value,
                        sub_layer.b, bias_value))
                else:
                    assign_ops.append(get_weight_partition_conv(name,
                        sub_layer.f, filter_value))

    return assign_ops

def get_vb_assign_bn(layer, scale_value, beta_value):
    assign_ops = []

    if layer.shared_branch is None:
        for sub_layer in layer.branches:
            assign_ops.append(tf.assign(sub_layer.scale, scale_value))
            assign_ops.append(tf.assign(sub_layer.beta, beta_value))
    else:
        assign_ops.append(get_weight_partition_bn('shared_to_shared',
            layer.shared_branch.scale, layer.shared_branch.beta,
            scale_value, beta_value))

        for sub_layer in layer.branches:
            assign_ops.append(get_weight_partition_bn('unique_to_unique',
                sub_layer.scale, sub_layer.beta, scale_value, beta_value))

    return assign_ops