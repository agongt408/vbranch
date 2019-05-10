import pandas as pd
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', default='mnist',
                    nargs='?', choices=['mnist', 'toy'], help='dataset')
parser.add_argument('--architecture', action='store', default='fcn',
                    nargs='?', choices=['fcn', 'cnn', 'fcn2'],
                    help='model architecture, i.e., fcn or cnn')
parser.add_argument('--num_classes', action='store', nargs='*', type=int,
                    help='number of classes in toy dataset (list)')
parser.add_argument('--samples_per_class', action='store', nargs='*', type=int,
                    help='samples per class (list)')
parser.add_argument('--shared_frac', action='store', nargs='*', type=float,
                    default=[0, 0.25, 0.5, 0.75, 1],
                    help='samples per class (list)')
parser.add_argument('--num_branches', action='store', nargs='*', type=int,
                    help='number of branches (list)')

def _get_results_path(dataset, arch, num_classes, samples_per_class, vb=False):
    if dataset == 'toy':
        # Further organize results by number of classes and samples_per_class
        dirpath = os.path.join('{}-{}'.format(dataset, arch),
            'C%d'%num_classes, 'SpC%d' % samples_per_class)
    else:
        dirpath = os.path.join('{}-{}'.format(dataset, arch))

    if vb:
        dirpath = 'vb-' + dirpath

    return os.path.join('results', dirpath)

def get_baseline_acc_from_file(dataset, arch, num_classes, samples_per_class):
    dirpath = _get_results_path(dataset, arch, num_classes, samples_per_class)

    acc_list = []
    for f in glob(dirpath + '/train_*'):
        csv = pd.read_csv(f)
        last_row = csv.iloc[-1]
        acc_list.append(last_row['val_acc'])

    return np.mean(acc_list), np.std(acc_list)

def get_ensemble_acc_from_file(dataset, arch, num_classes, samples_per_class,
        branches):

    dirpath = _get_results_path(dataset, arch, num_classes, samples_per_class)

    ensemble_results = {}
    for b in branches:
        f = os.path.join(dirpath, 'B%d-test.csv' % b)
        csv = pd.read_csv(f)
        acc_list = csv['before_mean_acc']
        ensemble_results[b] = [np.mean(acc_list), np.std(acc_list)]

    return ensemble_results

def get_vbranch_acc_from_file(dataset, arch, num_classes, samples_per_class,
        branches, shared_frac):

    dirpath = _get_results_path(dataset, arch, num_classes, samples_per_class,
        vb=True)

    vbranch_results = {}
    for b in branches:
        vbranch_results[b] = {}
        for s in shared_frac:
            acc_list = []
            path = os.path.join(dirpath,'B%d'%b,'S{:.2f}'.format(s))
            for f in glob(path + '/train_*'):
                csv = pd.read_csv(f)
                last_row = csv.iloc[-1]
                acc_list.append(last_row['val_acc_ensemble'])

            vbranch_results[b][s] = \
                [np.mean(acc_list), np.std(acc_list)]

    return vbranch_results

if __name__ == '__main__':
    args = parser.parse_args()

    baseline_results = {}
    for num_classes in args.num_classes:
        baseline_results[num_classes] = {}
        for samples_per_class in args.samples_per_class:
            baseline_results[num_classes][samples_per_class] = \
                get_baseline_acc_from_file(args.dataset, args.architecture,
                    num_classes, samples_per_class)

    ensemble_results = {}
    for num_classes in args.num_classes:
        ensemble_results[num_classes] = {}
        for samples_per_class in args.samples_per_class:
            ensemble_results[num_classes][samples_per_class] = \
                get_ensemble_acc_from_file(args.dataset, args.architecture,
                    num_classes, samples_per_class, args.num_branches)

    vbranch_results = {}
    for num_classes in args.num_classes:
        vbranch_results[num_classes] = {}
        for samples_per_class in args.samples_per_class:
            vbranch_results[num_classes][samples_per_class] = \
                get_vbranch_acc_from_file(args.dataset, args.architecture,
                    num_classes, samples_per_class, args.num_branches,
                    args.shared_frac)

    results = {
        'baseline':baseline_results,
        'ensemble':ensemble_results,
        'vbranch':vbranch_results
    }

    path = os.path.join('results','{}-{}.json'.format(args.dataset,
        args.architecture))

    with open(path, 'w') as fp:
        json.dump(results, fp)

    print('Finished!')
