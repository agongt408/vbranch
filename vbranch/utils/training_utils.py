from .. import datasets

import os
import pandas as pd
import numpy as np

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
    """Helper to save `data` dict to csv"""

    # Create folder to store csv
    csv_dir = os.path.join('results', dirname)
    if not os.path.isdir(csv_dir):
        os.system('mkdir -p ' + csv_dir)

    csv_path = os.path.join(csv_dir, filename)

    if mode == 'w':
        results = pd.DataFrame(data=data)
    elif mode == 'a':
        results = pd.DataFrame(data=data, index=[0])
    else:
        raise ValueError('invalid file I/O mode ("w" or "a")')

    if os.path.isfile(csv_path) and mode == 'a':
        results.to_csv(csv_path, mode=mode, header=False)
    else:
        results.to_csv(csv_path, mode=mode)

    return csv_path

def lr_exp_decay_scheduler(init_lr, t0, t1, decay):
    """NOTE: `episode` starts from 1"""
    def func(episode):
        if episode < t0:
            return init_lr
        lr = init_lr * np.power(decay, (episode - t0) / (t1 - t0))
        return lr
    return func

def get_data(dataset, architecture, num_classes=10, num_features=784,
        samples_per_class=1000):
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
            (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
                datasets.mnist.load_data(format='fcn')
        elif architecture.find('cnn') > -1:
            (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
                datasets.mnist.load_data(format='cnn')
        else:
            raise ValueError('invalid architecture:', architecture)
    elif dataset == 'toy':
        # Generate toy dataset
        # assert architecture in ['fcn', 'fcn2'], 'architecture must be fcn'
        assert not num_classes is None, 'num_classes cannot be None'

        num_samples = num_classes * samples_per_class

        print('Creating dataset (hypercube)...')
        (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
            datasets.toy.generate_from_hypercube(num_samples=num_samples,
                num_features=num_features, num_classes=num_classes)

        print('Training set:', X_train.shape)
        print('Testing set:', X_test.shape)
    else:
        raise ValueError('invalid dataset: ' + dataset)

    return (X_train, y_train_one_hot), (X_test, y_test_one_hot)

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

        x_list.append(x_sample)
        y_list.append(y_sample)

    return x_list, y_list

def p_console(*args):
    # Print to console
    print(bcolors.HEADER, *args, bcolors.ENDC)
