import tensorflow as tf
import numpy as np
import os
import pandas as pd
import json

def TFSessionGrow():
    # https://www.tensorflow.org/guide/using_gpu
    config = tf.ConfigProto(inter_op_parallelism_threads=8)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def restore_sess(sess, model_path):
    meta_path = os.path.join(model_path, 'ckpt.meta')
    ckpt = tf.train.get_checkpoint_state(model_path)

    imported_graph = tf.train.import_meta_graph(meta_path)
    imported_graph.restore(sess, ckpt.model_checkpoint_path)

# Model path helper functions

def get_path(dataset, arch, *prefixes, vb=False, model_id=None, **kwargs):
    path = '{}-{}'.format(dataset, arch)
    if vb:
        path = 'vb-' + path

    for name, val in kwargs.items():
        if type(val) is int or type(val) is str:
            path = os.path.join(path, '{}{}'.format(name, val))
        elif type(val) is float:
            path = os.path.join(path, '{}{:.2f}'.format(name, val))
        else:
            raise ValueError('invalid value, {}, of type {}'.\
                format(val, type(val)))

    for prefix in prefixes[::-1]:
        path = os.path.join(prefix, path)

    if model_id is not None:
        path = os.path.join('models', path, 'model_{}'.format(model_id))

    return path

def get_dir_path(dataset, arch, n_classes=None, samples_per_class=None):
    if dataset == 'toy':
        dirpath = get_path(dataset, arch, C=n_classes, SpC=samples_per_class)
    else:
        dirpath = get_path(dataset, arch)
    return dirpath

def get_model_path(dataset, arch, n_classes=None, samples_per_class=None, model_id=1):
    # Get path to save model
    dirpath = get_dir_path(dataset, arch, n_classes, samples_per_class)
    model_path = os.path.join('models', dirpath, 'model_%d' % model_id)

    if not os.path.isdir(model_path):
        os.system('mkdir -p ' + model_path)

    return model_path

def get_vb_dir_path(dataset,arch,n_branches,shared, n_classes=None,
        samples_per_class=None):
    if dataset == 'toy':
        dirpath = get_path(dataset, arch, vb=True, C=n_classes,
            SpC=samples_per_class, B=n_branches, S=shared)
    else:
        dirpath = get_path(dataset, arch, vb=True, B=n_branches, S=shared)
    return dirpath

def get_vb_model_path(dataset, arch, n_branches, shared, n_classes=None,
        samples_per_class=None, model_id=1):
    # Get path to save model
    dirpath = get_vb_dir_path(dataset, arch, n_branches, shared,
        n_classes, samples_per_class)
    model_path = os.path.join('models', dirpath, 'model_%d' % model_id)

    if not os.path.isdir(model_path):
        os.system('mkdir -p ' + model_path)

    return model_path

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def save_results(data, dirname, filename, mode='w'):
    """Helper to save `data` dict to csv or json"""

    # Create folder to store csv
    dirpath = os.path.join('results', dirname)
    if not os.path.isdir(dirpath):
        os.system('mkdir -p ' + dirpath)

    filepath = os.path.join(dirpath, filename)

    if 'csv' in filename:
        if mode == 'w':
            results = pd.DataFrame(data=data)
        elif mode == 'a':
            results = pd.DataFrame(data=data, index=[0])
        else:
            raise ValueError('invalid file I/O mode ("w" or "a")')

        if os.path.isfile(filepath) and mode == 'a':
            results.to_csv(filepath, mode=mode, header=False)
        else:
            results.to_csv(filepath, mode=mode)
    elif 'json' in filename:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        raise ValueError('filename must be .csv or .json')

    return filepath

def p_console(*args):
    # Print to console
    print(bcolors.HEADER, *args, bcolors.ENDC)
