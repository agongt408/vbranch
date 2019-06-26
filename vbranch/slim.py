from . import layers as L
from . import vb_layers as VBL
from .vb_layers.core import VBOutput
from .engine import Model, ModelVB

import tensorflow as tf

# Conventional layers and virtual branching layers are packaged into same function
# Automatically determine n_branches based on input
# If ambiguous, user can supply n_branches arg when declaring layer
# For vbranch version, if last layer (name='output'), then automatically
# merge shared and unique portions of output using MergeSharedUnique layer
# Dynamically determine layer name based on existing names in scope

# Core

def Input(x):
    if type(x) is list:
        return VBL.Input(x, len(x))
    return L.Input(x)

def Dense(x, units, use_bias=True, name=None, n_branches=None, shared=0,
        merge=False):
    if name is None:
        name = unused_scope('fc')

    if isinstance(x, VBOutput) or n_branches is not None:
        if type(shared) is int:
            shared_units = shared
        elif type(shared) is float:
            shared_units = int(shared * units)
        else:
            raise ValueError('shared must be int or float')

        if n_branches is None:
            n_branches = len(x)
        units_list = [units] * n_branches
        return VBL.Dense(units_list, n_branches, name, shared_units,
            merge=merge)(x)

    return L.Dense(units, name, use_bias=use_bias)(x)

def BatchNormalization(x, epsilon=1e-6, name=None, n_branches=None, merge=False):
    if name is None:
        name = unused_scope('bn')

    if isinstance(x, VBOutput) or n_branches is not None:
        if n_branches is None:
            n_branches = len(x)
        return VBL.BatchNormalization(n_branches, name, epsilon=epsilon,
            merge=merge)(x)

    return L.BatchNormalization(name, epsilon=epsilon)(x)

def Activation(x, activation, name=None, n_branches=None, merge=False):
    if name is None:
        name = unused_scope(activation.lower())

    if isinstance(x, VBOutput) or n_branches is not None:
        if n_branches is None:
            n_branches = len(x)
        return VBL.Activation(activation, n_branches, name, merge=merge)(x)

    return L.Activation(activation, name)(x)

def Conv2D(x, filters, kernel_size, strides=1, padding='valid',
        use_bias=True, name=None, n_branches=None, shared=0, merge=False):
    if name is None:
        name = unused_scope('conv2d')

    if isinstance(x, VBOutput) or n_branches is not None:
        if type(shared) is int:
            shared_filters = shared
        elif type(shared) is float:
            shared_filters = int(shared * filters)
        else:
            raise ValueError('shared must be int or float')

        if n_branches is None:
            n_branches = len(x)
        filters_list = [filters] * n_branches

        return VBL.Conv2D(filters_list, kernel_size, n_branches, name,
            shared_filters, strides, padding, merge=merge)(x)

    return L.Conv2D(filters, kernel_size, name, strides, padding, use_bias)(x)

# Pooling

def AveragePooling2D(x, pool_size, strides=None, padding='valid', name=None,
        n_branches=None, merge=False):
    if name is None:
        name = unused_scope('avg_pool2d')

    if isinstance(x, VBOutput) or n_branches is not None:
        if n_branches is None:
            n_branches = len(x)
        return VBL.AveragePooling2D(pool_size, n_branches, name, strides,
            padding, merge=merge)(x)

    return L.AveragePooling2D(pool_size, name, strides, padding)(x)

def GlobalAveragePooling2D(x, name=None, n_branches=None, merge=False):
    if name is None:
        name = unused_scope('global_avg_pool2d')

    if isinstance(x, VBOutput) or n_branches is not None:
        if n_branches is None:
            n_branches = len(x)
        return VBL.GlobalAveragePooling2D(n_branches, name, merge=merge)(x)

    return L.GlobalAveragePooling2D(name)(x)

def MaxPooling2D(x, pool_size, strides=None, padding='valid', name=None,
        n_branches=None, merge=False):
    if name is None:
        name = unused_scope('max_pool2d')

    if isinstance(x, VBOutput) or n_branches is not None:
        if n_branches is None:
            n_branches = len(x)
        return VBL.MaxPooling2D(pool_size, n_branches, name, strides,
            padding, merge=merge)(x)

    return L.MaxPooling2D(pool_size, name, strides, padding)(x)

# Merge

def Add(x, name=None, n_branches=None, merge=False):
    if name is None:
        name = unused_scope('add')

    if any([isinstance(x_, VBOutput) for x_ in x]) or n_branches is not None:
        if n_branches is None:
            assert all([len(x_) == len(x[0]) for x_ in x])
            n_branches = len(x[0])
        return VBL.Add(n_branches, name, merge=merge)(x)

    return L.Add(name)(x)

# Utils

def exist_scope(name):
    curr_scope = tf.get_variable_scope().name
    scope = curr_scope + '/' + name

    collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    try:
        tf.get_default_graph().get_operation_by_name(scope + '/output')
        exist_op = True
    except KeyError:
        exist_op = False

    if len(collection) > 0 or exist_op:
        return True

    return False

def unused_scope(init_name):
    """Finds first unused scope"""
    def iterate(name):
        if name.find('_') < 0:
            return name + '_1'

        reverse_name = name[::-1]
        counter = int(name[-reverse_name.find('_'):])
        return name[:-reverse_name.find('_')] + str(counter + 1)

    name = init_name
    while exist_scope(name):
        name = iterate(name)

    return name
