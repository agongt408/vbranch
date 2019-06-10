from .. import layers as L
from .. import vb_layers as VBL

from ..engine import Sequential, SequentialVB, Model, ModelVB

import tensorflow as tf

def default(input_tensor, num_classes, *layers_spec, name=None):
    # NOTE: input_tensor is a single Tensor object

    # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    # Wrap input tensor in Input layer in order to retrieve inbound name
    # when printing model summary
    ip = L.Input(input_tensor)
    x = ip

    for i, filters in enumerate(layers_spec):
        for l in range(2):
            x = L.Conv2D(filters, 3, 'conv2d_%d_%d' % (i+1, l+1))(x)
            x = L.BatchNormalization('bn_%d_%d' % (i + 1, l+1))(x)
            x = L.Activation('relu', 'relu_%d_%d' % (i+1, l+1))(x)

        if i < len(layers_spec) - 1:
            x = L.AveragePooling2D((2,2), 'avg_pool2d_'+str(i + 1))(x)
        else:
            x = L.GlobalAveragePooling2D('global_avg_pool2d')(x)

            # Embedding layers
            x = L.Dense(layers_spec[-1], 'fc1')(x)
            x = L.BatchNormalization('bn_fc1')(x)
            x = L.Activation('relu', 'relu_fc1')(x)
            x = L.Dense(num_classes, 'output')(x)

    model = Model(input_tensor, x, name=name)

    return model

def vbranch_default(inputs, final_spec, *layers_spec, branches=1, name=None):
    """
    Virtual branching version of CNN model for classification and
    one-shot learning. Created using Functional approach.

    Args:
        - inputs: single Tensor or list of Tensors
        - final_spec: tuple of (num_classes, shared_units)
        - layers_spec: tuple(s) of (filters_list, shared_filters)
        - branches: number of branches
    """

    # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    ip = VBL.Input(inputs, branches)
    x = ip

    for i, (filters_list, shared_filters) in enumerate(layers_spec):
        for l in range(2):
            x = VBL.Conv2D(filters_list,3,branches,
                'conv2d_%d_%d'%(i+1,l+1),shared_filters)(x)
            x = VBL.BatchNormalization(branches,'bn_%d_%d' % (i+1, l+1))(x)
            x = VBL.Activation('relu',branches,'relu_%d_%d' % (i+1, l+1))(x)

        if i < len(layers_spec) - 1:
            x = VBL.AveragePooling2D((2,2),branches,'avg_pool2d_'+str(i+1))(x)
        else:
            x = VBL.GlobalAveragePooling2D(branches,'global_avg_pool2d')(x)

            # Embedding layers
            x = VBL.Dense(layers_spec[-1][0],branches,'fc1',
                layers_spec[-1][1])(x)
            x = VBL.BatchNormalization(branches, 'bn_fc1')(x)
            x = VBL.Activation('relu', branches, 'relu_fc1')(x)

            # If using shared params
            if final_spec[1] > 0:
                x = VBL.Dense([final_spec[0]]*branches,branches,'fc2',
                                final_spec[1])(x)
                x = VBL.MergeSharedUnique(branches, 'output')(x)
            else:
                x = VBL.Dense([final_spec[0]]*branches,branches,'output',
                                    final_spec[1])(x)

    model = ModelVB(ip, x, name=name)

    return model

# def seq_version(input_tensor, num_classes, *layers_spec, name=None):
#     # NOTE: input_tensor is a single Tensor object
#
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#         model = Sequential(input_tensor, name=name)
#
#         for i, filters in enumerate(layers_spec):
#             for l in range(2):
#                 model.add(L.Conv2D(filters, 3, 'conv2d_%d_%d' % (i+1, l+1)))
#                 model.add(L.BatchNormalization('bn_%d_%d' % (i + 1, l+1)))
#                 model.add(L.Activation('relu', 'relu_%d_%d' % (i+1, l+1)))
#
#             if i < len(layers_spec) - 1:
#                 model.add(L.AveragePooling2D((2,2), 'avg_pool2d_'+str(i + 1)))
#             else:
#                 model.add(L.GlobalAveragePooling2D('global_avg_pool2d'))
#
#                 # Embedding layers
#                 model.add(L.Dense(layers_spec[-1], 'fc1'))
#                 model.add(L.BatchNormalization('bn_fc1'))
#                 model.add(L.Activation('relu', 'relu_fc1'))
#                 model.add(L.Dense(num_classes, 'output'))
#
#     return model
#
# def vbranch_seq(inputs, final_spec, *layers_spec, branches=1):
#     """
#     Virtual branching version of CNN model. Created using sequential approach.
#
#     Args:
#         - inputs: single Tensor or list of Tensors
#         - final_spec: tuple of (num_classes, shared_units)
#         - layers_spec: tuple(s) of (filters_list, shared_filters)
#         - branches: number of branches
#     """
#
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#         model = SequentialVB(inputs, name=name)
#
#         for i, (filters_list, shared_filters) in enumerate(layers_spec):
#             for l in range(2):
#                 model.add(VBL.Conv2D(filters_list,3,branches,
#                     'conv2d_%d_%d'%(i+1,l+1),shared_filters))
#                 model.add(VBL.BatchNormalization(branches,'bn_%d_%d'%(i+1,l+1)))
#                 model.add(VBL.Activation('relu',branches,'relu_%d_%d'%(i+1,l+1)))
#
#             if i < len(layers_spec) - 1:
#                 model.add(VBL.AveragePooling2D((2,2),branches,
#                     'avg_pool2d_'+str(i+1)))
#             else:
#                 model.add(VBL.GlobalAveragePooling2D(branches,
#                     'global_avg_pool2d'))
#
#                 # Embedding layers
#                 model.add(VBL.Dense(layers_spec[-1][0],branches,'fc1',
#                     layers_spec[-1][1]))
#                 model.add(VBL.BatchNormalization(branches, 'bn_fc1'))
#                 model.add(VBL.Activation('relu', branches, 'relu_fc1'))
#
#                 # If using shared params
#                 if final_spec[1] > 0:
#                     model.add(VBL.Dense([final_spec[0]]*branches,branches,'fc2',
#                                         final_spec[1]))
#                     model.add(VBL.MergeSharedUnique(branches, 'output'))
#                 else:
#                     model.add(VBL.Dense([final_spec[0]]*branches,branches,
#                                         'output', final_spec[1]))
#
#     return model
