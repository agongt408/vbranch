import tensorflow as tf
import numpy as np
import os

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

def get_shape_as_str(tensor):
    shape = tensor.get_shape().as_list()
    return shape_to_str(shape)

def shape_to_str(shape):
    return str(shape).replace(' ', '')

def get_num_params(tensor):
    return np.prod(tensor.get_shape().as_list())

# From VB layers

# def smart_add(x, y):
#     # Intelligently add x and y to avoid error when adding EmptyOutput
#     if isinstance(x, EmptyOutput) and isinstance(y, EmptyOutput):
#         return EmptyOutput()
#     elif isinstance(x, EmptyOutput):
#         return y
#     elif isinstance(y, EmptyOutput):
#         return x
#     else:
#         return x + y

def smart_add(x, y, name='add'):
    return smart_add_n([x, y], name=name)

def smart_add_n(x_list, name='add'):
    x_add = []
    for x in x_list:
        if not isinstance(x, EmptyOutput):
            x_add.append(x)

    if len(x_add) == 0:
        return EmptyOutput()

    return tf.add_n(x_add, name=name)

def smart_concat(xs, axis=-1, name='concat'):
    # Intelligently concat x and y to avoid error when concating EmptyOutput
    x_concat = []
    for x in xs:
        if not isinstance(x, EmptyOutput):
            x_concat.append(x)
    return tf.concat(x_concat, axis=axis, name=name)

def eval_params(func):
    """
    Decorator to evaluate the parameters returned by get_weights method
    using a tf session. Initializes variables if needed."""

    def inner(layer, eval_vars=False):
        variables = func(layer)

        if eval_vars:
            with tf.Session() as sess:
                try:
                    weights = sess.run(variables)
                except tf.errors.FailedPreconditionError:
                    sess.run(tf.global_variables_initializer())
                    weights = sess.run(variables)
        else:
            weights = variables

        return weights
    return inner

class EmptyOutput(object):
    pass

def TFSessionGrow():
    # https://www.tensorflow.org/guide/using_gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def restore_sess(sess, model_path):
    meta_path = os.path.join(model_path, 'ckpt.meta')
    ckpt = tf.train.get_checkpoint_state(model_path)

    imported_graph = tf.train.import_meta_graph(meta_path)
    imported_graph.restore(sess, ckpt.model_checkpoint_path)

# Model path helper functions

def _dir_path(dataset, arch, n_classes, samples_per_class):
    if dataset == 'toy':
        # Further organize results by number of classes and samples_per_class
        dirpath = os.path.join('{}-{}'.format(dataset, arch),'C%d'%n_classes,
            'SpC%d' % samples_per_class)
    else:
        dirpath = os.path.join('{}-{}'.format(dataset, arch))
    return dirpath

def get_model_path(dataset, arch, n_classes=None, samples_per_class=None, model_id=1):
    # Get path to save model
    dirpath = _dir_path(dataset, arch, n_classes, samples_per_class)
    model_path = os.path.join('models', dirpath, 'model_%d' % model_id)

    if not os.path.isdir(model_path):
        os.system('mkdir -p ' + model_path)

    return model_path

def _vb_dir_path(dataset,arch,n_branches,shared, n_classes=None,
        samples_per_class=None):

    if dataset == 'toy':
        # Further organize results by number of classes and samples_per_class
        dirpath = os.path.join('vb-{}-{}'.format(dataset, arch),
            'C%d'%n_classes, 'SpC%d' % samples_per_class, 'B%d'%n_branches,
            'S{:.2f}'.format(shared))
    else:
        dirpath = os.path.join('vb-{}-{}'.format(dataset, arch),
            'B%d'%n_branches, 'S{:.2f}'.format(shared))
    return dirpath

def get_vb_model_path(dataset, arch, n_branches, shared, n_classes=None,
        samples_per_class=None, model_id=1):
    # Get path to save model
    dirpath = _vb_dir_path(dataset, arch, n_branches, shared,
        n_classes, samples_per_class)
    model_path = os.path.join('models', dirpath, 'model_%d' % model_id)

    if not os.path.isdir(model_path):
        os.system('mkdir -p ' + model_path)

    return model_path
