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
                assign_ops.append(get_vb_assign_conv(layer,
                    weights[layer.name]['filter'],
                    weights[layer.name]['bias'] if layer.use_bias else None))
            elif isinstance(layer, vb_layers.BatchNormalization):
                assign_ops.append(get_vb_assign_bn(layer,
                    weights[layer.name]['scale'],
                    weights[layer.name]['beta']))

    return assign_ops

def load_weights_densenet(model, weights):
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

            if concat_layer is None:
                if isinstance(layer, vb_layers.Conv2D):
                    assign_ops.append(get_vb_assign_conv(layer,
                        weights[layer.name]['filter'],
                        weights[layer.name]['bias'] if layer.use_bias else None))
                elif isinstance(layer, vb_layers.BatchNormalization):
                    assign_ops.append(get_vb_assign_bn(layer,
                        weights[layer.name]['scale'],
                        weights[layer.name]['beta']))
            else:
                if isinstance(layer, vb_layers.Conv2D):
                    # Estimate shared_frac
                    dim1 = get_fan_in(concat_layer._inbound_tensors[0][0])
                    dim2 = get_fan_in(concat_layer._inbound_tensors[1][0])
                    shared_frac = get_fan_in(layer._inbound_tensors[0][0][0]) / \
                        get_fan_in(layer._inbound_tensors[0][0])

                    assign_ops.append(get_vb_assign_conv_concat(layer,
                        dim1, dim2, shared_frac, weights[layer.name]['filter']))
                    # Reset concat layer only after reached first conv layer
                    concat_layer = None
                elif isinstance(layer, vb_layers.BatchNormalization):
                    # Estimate shared_frac
                    dim1 = get_fan_in(concat_layer._inbound_tensors[0][0])
                    dim2 = get_fan_in(concat_layer._inbound_tensors[1][0])
                    shared_frac = get_fan_in(layer._inbound_tensors[0][0][0]) / \
                        get_fan_in(layer._inbound_tensors[0][0])

                    assign_ops.append(get_vb_assign_bn_concat(layer,
                        dim1, dim2, shared_frac,
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

def get_vb_assign_conv_concat(layer, dim1, dim2, shared_frac, filter_value):
    assign_ops = []

    if layer.shared_branch is None:
        for sub_layer in layer.branches:
            assign_ops.append(tf.assign(sub_layer.f, filter_value))

            if layer.use_bias:
                assign_ops.append(tf.assign(sub_layer.b, bias_value))
    else:
        part1 = filter_value[:, :, :dim1]
        part2 = filter_value[:, :, dim1:]

        # Shared to shared
        shared_to_shared = np.concatenate([
            part1[:, :, :int(shared_frac*dim1), :layer.shared_filters],
            part2[:, :, :int(shared_frac*dim2), :layer.shared_filters]
        ], axis=2)

        assign_ops.append(tf.assign(layer.shared_branch.f, shared_to_shared))

        # CrossWeights
        shared_to_unique = np.concatenate([
            part1[:, :, :int(shared_frac*dim1), layer.shared_filters:],
            part2[:, :, :int(shared_frac*dim2), layer.shared_filters:]
        ], axis=2)
        unique_to_shared = np.concatenate([
            part1[:, :, int(shared_frac*dim1):, :layer.shared_filters],
            part2[:, :, int(shared_frac*dim2):, :layer.shared_filters]
        ], axis=2)
        unique_to_unique = np.concatenate([
            part1[:, :, int(shared_frac*dim1):, layer.shared_filters:],
            part2[:, :, int(shared_frac*dim2):, layer.shared_filters:]
        ], axis=2)

        for sub_layer in layer.branches:
            assign_ops.append(tf.assign(sub_layer.shared_to_unique.f, shared_to_unique))
            assign_ops.append(tf.assign(sub_layer.unique_to_shared.f, unique_to_shared))
            assign_ops.append(tf.assign(sub_layer.unique_to_unique.f, unique_to_unique))

    return assign_ops

def get_vb_assign_bn_concat(layer, dim1, dim2, shared_frac, scale_value, beta_value):
    assign_ops = []

    if layer.shared_branch is None:
        for sub_layer in layer.branches:
            assign_ops.append(tf.assign(sub_layer.scale, scale_value))
            assign_ops.append(tf.assign(sub_layer.beta, beta_value))
    else:
        scale_part1 = scale_value[:dim1]
        scale_part2 = scale_value[dim1:]
        beta_part1 = beta_value[:dim1]
        beta_part2 = beta_value[dim1:]

        # Shared to shared
        assign_ops.append(tf.assign(layer.shared_branch.scale, np.concatenate([
            scale_part1[:int(shared_frac*dim1)],
            scale_part2[:int(shared_frac*dim2)]
        ])))
        assign_ops.append(tf.assign(layer.shared_branch.beta, np.concatenate([
            beta_part1[:int(shared_frac*dim1)],
            beta_part2[:int(shared_frac*dim2)]
        ])))

        scale_unique = np.concatenate([
            scale_part1[int(shared_frac*dim1):],
            scale_part2[int(shared_frac*dim2):]
        ])
        beta_unique = np.concatenate([
            beta_part1[int(shared_frac*dim1):],
            beta_part2[int(shared_frac*dim2):]
        ])

        for sub_layer in layer.branches:
            assign_ops.append(tf.assign(sub_layer.scale, scale_unique))
            assign_ops.append(tf.assign(sub_layer.beta, beta_unique))

    return assign_ops
