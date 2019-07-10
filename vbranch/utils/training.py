from .. import datasets

import os
import pandas as pd
import numpy as np
import tensorflow as tf

def lr_exp_decay_scheduler(init_lr, t0, t1, decay):
    """NOTE: `episode` starts from 1"""
    def func(episode):
        if episode < t0:
            return init_lr
        lr = init_lr * np.power(decay, (episode - t0) / (t1 - t0))
        return lr
    return func

def beta1_scheduler(t0, beta_init=0.9, beta_final=0.5):
    def func(episode):
        if episode > t0:
            return beta_final
        return beta_init
    return func

def get_data(dataset, architecture, num_classes=10, num_features=784,
        samples_per_class=1000, one_hot=True, train_frac=1, seed=100):
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
                datasets.mnist.load_data(format='fcn', one_hot=one_hot)
        elif architecture.find('cnn') > -1:
            (X_train, y_train), (X_test, y_test) = \
                datasets.mnist.load_data(format='cnn', one_hot=one_hot)
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
    else:
        raise ValueError('invalid dataset: ' + dataset)

    if train_frac < 1:
        np.random.seed(seed)
        subsample = np.random.choice(len(X_train), int(train_frac*len(X_train)),
            replace=False)
        X_train = X_train[subsample]
        y_train = y_train[subsample]

    return (X_train, y_train), (X_test, y_test)

def bag_samples(X, Y, n, max_samples=1.0, bootstrap=True):
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

def wrap_iterator(generator, *args):
    def func():
        while True:
            batch = generator.next(*args).astype('float32')
            yield batch
    return func

def get_data_iterator(x_shape, y_shape, batch_size, n=1, share_xy=True):
    batch_size_ = tf.placeholder('int64', name='batch_size')

    if share_xy or n == 1:
        x = tf.placeholder('float32', x_shape, name='x')
        y = tf.placeholder('float32', y_shape, name='y')

    inputs = []
    labels_one_hot = []
    train_init_op = []
    test_init_op = []

    for i in range(n):
        if not share_xy and not n == 1:
            x = tf.placeholder('float32', x_shape, name='vb{:d}_x'.format(i+1))
            y = tf.placeholder('float32', y_shape, name='vb{:d}_y'.format(i+1))

        train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).\
            repeat().batch(batch_size_).shuffle(buffer_size=4*batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x,y)).\
            repeat().batch(batch_size_)

        iter_ = tf.data.Iterator.from_structure(('float32','float32'), (x_shape, y_shape))
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
                name='test_init_op_'+str(i+1)))

    return inputs, labels_one_hot, train_init_op, test_init_op

def get_data_iterator_from_generator(train_gen, input_dim, *args, n=1):
    """
    Create baseline/vbranch iterator from generator (train), and tensor slices
    (test). E.g., used for Omniglot dataset.
    Args:
        - train_gen: object with next() method (not an iterator or Python
        generator)
        - input_dim: dimension of expected input
        - n: number of branches
    """

    x = tf.placeholder('float32', input_dim, name='x')
    batch_size = tf.placeholder('int64', name='batch_size')

    inputs = []
    train_init_op = []
    test_init_op = []

    for i in range(n):
        train_dataset = tf.data.Dataset.\
            from_generator(wrap_iterator(train_gen, *args),
                'float32', output_shapes=input_dim)
        test_dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        iterator = tf.data.Iterator.from_structure('float32', input_dim)

        if n == 1:
            inputs = iterator.get_next('input')
            train_init_op = iterator.make_initializer(train_dataset)
            test_init_op = iterator.make_initializer(test_dataset,
                name='test_init_op')
        else:
            inputs.append(iterator.get_next(name='input_'+str(i+1)))
            train_init_op.append(iterator.make_initializer(train_dataset))
            test_init_op.append(iterator.make_initializer(test_dataset,
                name='test_init_op_'+str(i+1)))

    return inputs, train_init_op, test_init_op
