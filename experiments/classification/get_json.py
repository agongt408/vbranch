import sys
sys.path.insert(0, '.')

from vbranch.utils.generic import get_path

import pandas as pd
import os
import numpy as np
from glob import glob
import json
import argparse

# # Parse command line arguments
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--dataset', action='store', default='mnist',
#                     nargs='?', choices=['mnist', 'toy'], help='dataset')
# parser.add_argument('--architecture', action='store', default='fcn',
#                     nargs='?', help='model architecture, i.e., fcn or cnn')
# parser.add_argument('--num_classes', action='store', nargs='*', type=int,
#                     help='number of classes in toy dataset (list)')
# parser.add_argument('--samples_per_class', action='store', nargs='*', type=int,
#                     help='samples per class (list)')
# parser.add_argument('--shared_frac', action='store', nargs='*', type=float,
#                     default=[0, 0.25, 0.5, 0.75, 1],
#                     help='shared frac list')
# parser.add_argument('--num_branches', action='store', nargs='*', type=int,
#                     help='number of branches (list)')

# Params
dataset = 'mnist'
architecture = 'fcn'
shared_frac = [0, 0.25, 0.5, 0.75, 1]
num_branches = range(2, 7)

param1 = [0.01, 0.05, 0.1, 0.2]
param2 = [8, 16, 32, 64, 128]

def _get_results_path(dataset, arch, p1, p2, vb=False):
    # Set kwargs manually before running
    dirpath = get_path(dataset, arch, 'results', 'data_exp-2', vb=vb, F=p1, Ba=p2)
    return dirpath

def get_baseline_acc_from_file(dataset, arch, p1, p2):
    dirpath = _get_results_path(dataset, arch, p1, p2)
    print(dirpath)

    acc_list = []
    for f in glob(dirpath + '/train_*'):
        csv = pd.read_csv(f)
        last_row = csv.iloc[-1]
        acc_list.append(last_row['val_acc'])

    assert len(acc_list) > 0, '{} {}'.format(p1, p2)
    return np.mean(acc_list), np.std(acc_list)

def get_ensemble_acc_from_file(dataset, arch, p1, p2, branches):
    dirpath = _get_results_path(dataset, arch, p1, p2)

    ensemble_results = {}
    for b in branches:
        f = os.path.join(dirpath, 'B%d-test.csv' % b)
        csv = pd.read_csv(f)
        acc_list = csv['before_mean_acc']
        assert len(acc_list) > 0, '{} {} {}'.format(p1, p2, b)
        ensemble_results[b] = [np.mean(acc_list), np.std(acc_list)]

    return ensemble_results

def get_vbranch_acc_from_file(dataset, arch, p1, p2, branches, shared_frac):
    dirpath = _get_results_path(dataset, arch, p1, p2, vb=True)

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

            assert len(acc_list) > 0, '{} {} {} {}'.format(p1, p2, b, s)
            vbranch_results[b][s] = \
                [np.mean(acc_list), np.std(acc_list)]

    return vbranch_results

if __name__ == '__main__':
    # args = parser.parse_args()

    baseline_results = {}
    for p1 in param1:
        baseline_results[p1] = {}
        for p2 in param2:
            baseline_results[p1][p2] = \
                get_baseline_acc_from_file(dataset, architecture, p1, p2)

    ensemble_results = {}
    for p1 in param1:
        ensemble_results[p1] = {}
        for p2 in param2:
            ensemble_results[p1][p2] = \
                get_ensemble_acc_from_file(dataset, architecture,
                    p1, p2, num_branches)

    vbranch_results = {}
    for p1 in param1:
        vbranch_results[p1] = {}
        for p2 in param2:
            vbranch_results[p1][p2] = \
                get_vbranch_acc_from_file(dataset, architecture,
                    p1, p2, num_branches, shared_frac)

    results = {
        'baseline':baseline_results,
        'ensemble':ensemble_results,
        'vbranch':vbranch_results
    }

    path = os.path.join('results','{}-{}.json'.format(dataset, architecture))

    with open(path, 'w') as fp:
        json.dump(results, fp)

    print('Finished!')
