import tensorflow as tf
import numpy as np

class Summary(object):
    """Helper class used to print model summaries"""

    def __init__(self, *labels):
        self.labels = labels
        self.rows = [] # Stores contents of each row
        self.show_line = [] # Stores whether to show line for each row

    def add(self, *items, show_line=True):
        assert len(self.labels) == len(items)
        self.rows.append(items)
        self.show_line.append(show_line)

    def show(self):
        # Include labels
        # Array of lists
        print_rows = [self.labels,] + self.rows
        print_show_line = [True,] + self.show_line

        widths = []
        for i in range(len(self.labels)):
            w = np.max([len(str(row[i])) for row in print_rows]) + 2
            widths.append(w)

        total_width = np.sum(widths)

        for r, row in enumerate(print_rows):
            str_f = ''
            for i in range(len(row)):
                str_f += ('{:<'+str(widths[i])+'}').format(str(row[i]))

            print(str_f)

            if print_show_line[r]:
                print('-' * total_width)

def get_shape(x):
    if isinstance(x, EmptyOutput):
        return []
    else:
        return x.get_shape().as_list()

def get_shape_as_str(tensor):
    shape = tensor.get_shape().as_list()
    return shape_to_str(shape)

def shape_to_str(shape):
    return str(shape).replace(' ', '')

def get_num_params(tensor):
    return np.prod(tensor.get_shape().as_list())

# From VB layers

def smart_add(x, y, name='add'):
    return smart_add_n([x, y], name=name)

def smart_add_n(x_list, name='add'):
    x_add = []
    for x in x_list:
        if not isinstance(x, EmptyOutput):
            x_add.append(x)

    if len(x_add) == 0:
        return EmptyOutput()
    # elif len(x_add) == 1:
    #     return x_add[0]

    return tf.add_n(x_add, name=name)

def smart_concat(xs, axis=-1, name='concat'):
    if isinstance(xs, tf.Tensor):
        return xs

    # Intelligently concat x and y to avoid error when concating EmptyOutput
    x_concat = []
    for x in xs:
        if not isinstance(x, EmptyOutput):
            x_concat.append(x)

    if len(x_concat) == 0:
        return EmptyOutput()
    # elif len(x_concat) == 1:
    #     return x_concat[0]

    return tf.concat(x_concat, axis=axis, name=name)

def eval_params(func):
    """
    Decorator to evaluate the parameters returned by get_weights method
    using a tf session. Initializes variables if needed."""

    def inner(layer, sess=None):
        variables = func(layer)

        if sess is None:
            weights = variables
        else:
            weights = sess.run(variables)
        return weights

    return inner

class EmptyOutput(object):
    pass

def check_2d_param(p):
    """Returns correctly formatted parameter for pool_size, kernel_size, strides"""
    if type(p) is int:
        p_list = [p, p]
    elif type(p) is tuple or type(p) is list:
        p_list = list(p)
    else:
        raise ValueError('parameter must be int or tuple')

    return p_list

def get_fan_in(inputs):
    if type(inputs) is list:
        fan_in_list = []
        for x in inputs:
            if not isinstance(x, EmptyOutput):
                fan_in_list.append(x.get_shape().as_list()[-1])
        fan_in = np.sum(fan_in_list)
    else:
        fan_in = inputs.get_shape().as_list()[-1]
    return fan_in
