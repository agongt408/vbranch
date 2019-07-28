from .. import layers
from .. import vb_layers
from ..engine.training import Model
from ..utils import get_fan_in

import tensorflow as tf
import numpy as np

def load_weights_resnet(model, weights):
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
                assign_ops.append(get_vb_assign_conv(layer, weights[layer.name]))
            elif isinstance(layer, vb_layers.BatchNormalization):
                assign_ops.append(get_vb_assign_bn(layer, weights[layer.name]))

    return assign_ops

def load_weights_densenet(model, weights, growth_rate=32, init_dim=64):
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
        concat_layer = None

        for layer in model.layers:
            # Only load weights up to the GlobalAveragePooling2D layer
            if isinstance(layer, vb_layers.GlobalAveragePooling2D):
                break

            if isinstance(layer, vb_layers.Concatenate):
                concat_layer = layer

            if isinstance(layer, vb_layers.AveragePooling2D):
                init_dim = get_fan_in(layer._inbound_tensors[0][0])

            if concat_layer is None:
                if isinstance(layer, vb_layers.Conv2D):
                    assign_ops.append(get_vb_assign_conv(layer, weights[layer.name]))
                elif isinstance(layer, vb_layers.BatchNormalization):
                    assign_ops.append(get_vb_assign_bn(layer, weights[layer.name]))
            else:
                if isinstance(layer, vb_layers.Conv2D):
                    # # Estimate shared_frac
                    shared_frac = get_fan_in(layer._inbound_tensors[0][0][0]) / \
                        get_fan_in(layer._inbound_tensors[0][0])

                    assign_ops.append(get_vb_assign_conv_concat(layer,
                        weights[layer.name], growth_rate, init_dim, shared_frac))

                    # Reset concat layer only after reaching first conv layer
                    concat_layer = None
                elif isinstance(layer, vb_layers.BatchNormalization):
                    # # Estimate shared_frac
                    shared_frac = get_fan_in(layer._inbound_tensors[0][0][0]) / \
                        get_fan_in(layer._inbound_tensors[0][0])

                    assign_ops.append(get_vb_assign_bn_concat(layer,
                        weights[layer.name], growth_rate, init_dim, shared_frac))

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

def get_vb_assign_conv(layer, weights):
    assign_ops = []

    if layer.shared_branch is None:
        for sub_layer in layer.branches:
            assign_ops.append(tf.assign(sub_layer.f, weights['filter']))

            if 'bias' in weights.keys():
                assign_ops.append(tf.assign(sub_layer.b, weights['bias']))
    else:
        if 'bias' in weights.keys():
            assign_ops.append(get_weight_partition_conv('shared_to_shared',
                layer.shared_branch.f, weights['filter'],
                layer.shared_branch.b, weights['bias']))
        else:
            assign_ops.append(get_weight_partition_conv('shared_to_shared',
                layer.shared_branch.f, weights['filter']))

        for branch in layer.branches:
            # Extract CrossWeights namedtuple
            for name, sub_layer in branch._asdict().items():
                if 'bias' in weights.keys():
                    assign_ops.append(get_weight_partition_conv(name,
                        sub_layer.f, weights['filter'],
                        sub_layer.b, weights['bias']))
                else:
                    assign_ops.append(get_weight_partition_conv(name,
                        sub_layer.f, weights['filter']))

    return assign_ops

def get_vb_assign_bn(layer, weights):
    assign_ops = []

    if layer.shared_branch is None:
        for sub_layer in layer.branches:
            assign_ops.append(tf.assign(sub_layer.scale, weights['scale']))
            assign_ops.append(tf.assign(sub_layer.beta, weights['beta']))
    else:
        assign_ops.append(get_weight_partition_bn('shared_to_shared',
            layer.shared_branch.scale, layer.shared_branch.beta,
            weights['scale'], weights['beta']))

        for sub_layer in layer.branches:
            assign_ops.append(get_weight_partition_bn('unique_to_unique',
                sub_layer.scale, sub_layer.beta, weights['scale'], weights['beta']))

    return assign_ops

def get_vb_assign_conv_concat(layer, weight, growth_rate, init_dim, shared_frac):
    shared, unique = rearrange_4d(weight['filter'], growth_rate,
        init_dim, shared_frac)
    filter_value = np.concatenate([shared, unique], axis=-2)
    return get_vb_assign_conv(layer, filter_value)

def get_vb_assign_bn_concat(layer, weight, growth_rate, init_dim, shared_frac):
    scale_shared, scale_unique = rearrange_1d(weight['scale'],
        growth_rate, init_dim, shared_frac)
    scale_value = np.concatenate([scale_shared, scale_unique])

    beta_shared, beta_unique = rearrange_1d(weight['beta'],
        growth_rate, init_dim, shared_frac)
    beta_value = np.concatenate([beta_shared, beta_unique])

    return get_vb_assign_bn(layer, scale_value, beta_value)

def rearrange_4d(weight, growth_rate, init_dim, shared_frac):
    fan_in = weight.shape[-2]
    fan_out = weight.shape[-1]

    if fan_in == init_dim:
        shared = weight[..., :int(shared_frac*fan_in), :]
        unique = weight[..., int(shared_frac*fan_in):, :]
        return shared, unique

    shared_1, unique_1 = rearrange_4d(weight[..., :-growth_rate, :],
        growth_rate, init_dim, shared_frac)

    part2 = weight[..., -growth_rate:, :]
    shared_2 = part2[..., :int(shared_frac*growth_rate), :]
    unique_2 = part2[..., int(shared_frac*growth_rate):, :]

    shared = np.concatenate([shared_1, shared_2], axis=-2)
    unique = np.concatenate([unique_1, unique_2], axis=-2)
    return shared, unique

def rearrange_1d(weight, growth_rate, init_dim, shared_frac):
    fan_in = weight.shape[0]

    if fan_in == init_dim:
        shared = weight[:int(shared_frac*fan_in)]
        unique = weight[int(shared_frac*fan_in):]
        return shared, unique

    shared_1, unique_1 = rearrange_1d(weight[:-growth_rate],
        growth_rate, init_dim, shared_frac)

    part2 = weight[-growth_rate:]
    shared_2 = part2[:int(shared_frac*growth_rate)]
    unique_2 = part2[int(shared_frac*growth_rate):]

    shared = np.concatenate([shared_1, shared_2])
    unique = np.concatenate([unique_1, unique_2])
    return shared, unique
