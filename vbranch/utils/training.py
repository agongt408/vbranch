from .. import datasets

import os
import pandas as pd
import numpy as np
import tensorflow as tf

def lr_exp_decay_scheduler(init_lr, t0, t1, decay, warm_up=0):
    """NOTE: `episode` starts from 1"""
    def func(episode):
        if episode <= warm_up:
            lr = (episode - 1) * (init_lr - init_lr*decay) / warm_up + init_lr*decay
        elif episode < t0:
            lr = init_lr
        else:
            lr = init_lr * np.power(decay, (episode - t0) / (t1 - t0))
        return lr
    return func

def lr_step_scheduler(*args):
    """
    args must be in format (t, lr), (t, lr), ...
    """
    def func(episode):
        i = 0
        while episode > args[i][0]:
            i += 1
        return args[i][1]
    return func

# # https://arxiv.org/pdf/1807.00537.pdf
# def lr_warm_up_scheduler():
#     """NOTE: `episode` starts from 1"""
#     def func(episode):
#         if episode <= 20:
#             lr = (episode - 1) * (1e-3 - 5e-5) / 20 + 5e-5
#         elif episode <= 80:
#             lr = 1e-3
#         elif episode <= 100:
#             lr = 1e-4
#         else:
#             lr = 1e-5
#         return lr
#     return func

def beta1_scheduler(t0, beta_init=0.9, beta_final=0.5):
    """NOTE: `episode` starts from 1"""
    def func(episode):
        if episode > t0:
            return beta_final
        return beta_init
    return func

def get_data(dataset, architecture, num_classes=10, num_features=784,
        samples_per_class=1000, one_hot=True, train_frac=1, seed=100,
        preprocess=False):
    """
    Load (or generate) data
    Args:
        - dataset: 'mnist' or 'toy'
        - architecture: 'fcn' or 'cnn' (must be 'fcn' if `dataset`='toy')
        - num_classes: number of classes to generate toy dataset (must be 10
        if `dataset`='mnist')
        - num_features: number of features
        - samples_per_class: number of samples per class
    """

    # Load (or generate) data
    if dataset == 'mnist':
        # Load data from MNIST
        if architecture.find('fcn') > -1:
            (X_train, y_train), (X_test, y_test) = \
                datasets.mnist.load_data(format='fcn', one_hot=one_hot,
                    preprocess=preprocess)
        elif architecture.find('cnn') > -1:
            (X_train, y_train), (X_test, y_test) = \
                datasets.mnist.load_data(format='cnn', one_hot=one_hot,
                    preprocess=preprocess)
        else:
            raise ValueError('invalid architecture:', architecture)
    elif dataset == 'toy':
        # Generate toy dataset
        # assert architecture in ['fcn', 'fcn2'], 'architecture must be fcn'
        assert not num_classes is None, 'num_classes cannot be None'

        num_samples = num_classes * samples_per_class

        print('Creating dataset (hypercube)...')
        (X_train, y_train), (X_test, y_test) = \
            datasets.toy.generate_from_hypercube(num_samples=num_samples,
                num_features=num_features, num_classes=num_classes)

        print('Training set:', X_train.shape, y_train.shape)
        print('Testing set:', X_test.shape, y_test.shape)
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data(one_hot,
            preprocess=preprocess)
    else:
        raise ValueError('invalid dataset: ' + dataset)

    if train_frac < 1:
        np.random.seed(seed)
        subsample = np.random.choice(len(X_train), int(train_frac*len(X_train)),
            replace=False)
        X_train = X_train[subsample]
        y_train = y_train[subsample]

    return (X_train, y_train), (X_test, y_test)

def bag_samples(X, Y, n, max_samples=1.0, bootstrap=False):
    """
    Perform bagging on numpy arrays X, Y
    Args:
        - n: number of samples
        - max_samples: float or int, size of each random sample
        - bootstrap: if true, choose with replacement
    """

    x_list = []
    y_list = []

    if type(max_samples) is float:
        samples = int(max_samples * len(X))
    elif type(max_samples) is int:
        samples = max_samples
    else:
        raise ValueError('max_samples must be float or int')

    for i in range(n):
        choice = np.random.choice(len(X), samples, replace=bootstrap)
        x_sample = X[choice]
        y_sample = Y[choice]

        if n > 1:
            x_list.append(x_sample)
            y_list.append(y_sample)
        else:
            x_list = x_sample
            y_list = y_sample

    return x_list, y_list

def get_data_iterator(x_shape, y_shape, batch_size, n=1, share_xy=True):
    batch_size_ = tf.placeholder('int64', name='batch_size')

    x_test = tf.placeholder('float32', x_shape, name='x')
    y_test = tf.placeholder('float32', y_shape, name='y')

    inputs = []
    labels_one_hot = []
    train_init_op = []
    test_init_op = []

    for i in range(n):
        if not share_xy and n > 1:
            x = tf.placeholder('float32', x_shape, name=f'vb{i+1}_x')
            y = tf.placeholder('float32', y_shape, name=f'vb{i+1}_y')
        else:
            x, y = x_test, y_test

        train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).\
            repeat().batch(batch_size_).shuffle(buffer_size=4*batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).\
            repeat().batch(batch_size_)

        iter_ = tf.data.Iterator.from_structure(('float32','float32'),
                                                (x_shape, y_shape))
        input_, label_one_hot_ = iter_.get_next('input')

        if n == 1:
            inputs = input_
            labels_one_hot = label_one_hot_
            train_init_op = iter_.make_initializer(train_dataset)
            test_init_op = iter_.make_initializer(test_dataset, name='test_init_op')
        else:
            inputs.append(input_)
            labels_one_hot.append(label_one_hot_)
            train_init_op.append(iter_.make_initializer(train_dataset))
            test_init_op.append(iter_.make_initializer(test_dataset,
                name=f'test_init_op_{i+1}'))

    return inputs, labels_one_hot, train_init_op, test_init_op

def get_data_iterator_from_generator(generators, input_dim, n=1, labels=False):
    """
    Create baseline/vbranch iterator from generator (train), and tensor slices
    (test). E.g., used for Omniglot dataset.
    Args:
        - train_gen: Python generator or instance of class that implements
        __next__ method
        - input_dim: dimension of expected input
        - n: number of branches
    """
    def wrap(generator):
        def func():
            while True:
                batch = next(generator)
                yield batch
        return func

    if labels:
        x = (tf.placeholder('float32', input_dim[0], name='x'),
            tf.placeholder('float32', input_dim[1], name='y'))
        output_types = ('float32', 'float32')
    else:
        x = tf.placeholder('float32', input_dim, name='x')
        output_types = 'float32'
    batch_size = tf.placeholder('int64', name='batch_size')

    inputs = []
    train_init_op = []
    test_init_op = []

    for i in range(n):
        train_gen = generators[i] if type(generators) is list else generators
        print(output_types, input_dim)
        train_dataset = tf.data.Dataset.from_generator(wrap(train_gen),
            output_types, output_shapes=input_dim)
        test_dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        iterator = tf.data.Iterator.from_structure(output_types, input_dim)

        if n == 1:
            inputs = iterator.get_next('input')
            train_init_op = iterator.make_initializer(train_dataset)
            test_init_op = iterator.make_initializer(test_dataset,
                name='test_init_op')
        else:
            inputs.append(iterator.get_next(name='input_'+str(i+1)))
            train_init_op.append(iterator.make_initializer(train_dataset))
            test_init_op.append(iterator.make_initializer(test_dataset,
                name=f'test_init_op_{i+1}'))

    inputs = [list(x) for x in zip(*inputs)] if labels and n>1 else inputs
    return inputs, train_init_op, test_init_op

# Enable non-replacement A sampling for multiple branches
class SyncGenerator(object):
    def __init__(self, generator, batch_size, n_branches):
        self.n_branches = n_branches
        self.gen = generator
        self.batch = None
        self.requests = 0

    def get(self, i):
        if self.batch is None:
            self.batch = next(self.gen)
            self.requests = self.n_branches

        start = i*self.batch_size
        end = (i+1)*self.batch_size
        branch_batch = self.batch[start: end]
        self.requests -= 1

        if self.requests == 0:
            self.batch = None

        return branch_batch

class Slicer(object):
    def __init__(self, parent, branch):
        self.parent = parent
        self.branch = branch

    def __next__(self):
        return self.parent.get(self.branch)
