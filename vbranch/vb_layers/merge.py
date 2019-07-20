from .. import layers as L
from .core import Layer, VBOutput
from ..utils.layer import *

from tensorflow import Tensor

class Add(Layer):
    def __init__(self, n_branches, name, merge=False):
        super().__init__(name, n_branches, merge)

    @Layer.call
    def __call__(self, x):
        # x: list of VBOutput objects
        assert type(x) is list, 'x is not a list of VBOutput objects'
        return _recursive_smart_add(x)

class Concatenate(Layer):
    def __init__(self, n_branches, name, merge=False):
        super().__init__(name, n_branches, merge)

    @Layer.call
    def __call__(self, x):
        # x: list of VBOutput objects
        assert type(x) is list, 'x is not a list of VBOutput objects'
        return _recursive_smart_concat(x)

def _make_frame(x, frame):
    """
    Make a nested empty list with same shape as x
    """
    if (type(x) is not list) and (not isinstance(x, VBOutput)):
        return

    for x_ in x:
        frame.append([])
        _make_frame(x_, frame[-1])

def _recursive_smart_add(x_list):
    """
    smart_add with broadcast functionality
    """
    if all([isinstance(x, Tensor) or isinstance(x, EmptyOutput) for x in x_list]):
        return smart_add_n(x_list)

    # Assume all x in x_list have the same shape
    placeholder = []
    _make_frame(x_list[0], placeholder)

    for i in range(len(placeholder)):
        input_ = [x[i] for x in x_list]
        placeholder[i] = _recursive_smart_add(input_)

    return placeholder

def _recursive_smart_concat(x_list):
    """
    smart_add with broadcast functionality
    """
    if all([isinstance(x, Tensor) or isinstance(x, EmptyOutput) for x in x_list]):
        return smart_concat(x_list, axis=-1)

    # Assume all x in x_list have the same shape
    placeholder = []
    _make_frame(x_list[0], placeholder)

    for i in range(len(placeholder)):
        input_ = [x[i] for x in x_list]
        placeholder[i] = _recursive_smart_concat(input_)

    return placeholder
