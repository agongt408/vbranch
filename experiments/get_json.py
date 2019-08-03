import sys
sys.path.insert(0, '.')

import pandas as pd
import os
import numpy as np
from glob import glob
import json
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='mnist', nargs='?', help='dataset')
parser.add_argument('--architecture', default='fcn',
                    nargs='?', help='model architecture, i.e., fcn or cnn')
parser.add_argument('--dir', default='results', nargs='?',
                    help='path to directory contaning results/experiment')
parser.add_argument('--max_branches', default=4, nargs='?', type=int,
                    help='branches from 2 to max_branches (inclusive)')
parser.add_argument('--shared_frac', nargs='*', type=float,
                    default=[0, 0.25, 0.5, 0.75, 1], help='shared frac list')
parser.add_argument('--params', nargs='*', help='additional hyperparameters')
parser.add_argument('--metric', default='val_acc',
                    nargs='?', help='performance metric')
parser.add_argument('--epoch', default=-1, nargs='?', type=int,
                    help='epoch to take results')

def get_baseline_acc_from_file(params, dirpath, metric, epoch):
    print(dirpath)

    acc_list = []
    for f in glob(dirpath + '/train_*'):
        csv = pd.read_csv(f)
        acc_list.append(csv.iloc[epoch][metric])

    assert len(acc_list) > 0, params
    return np.mean(acc_list), np.std(acc_list)

def get_ensemble_acc_from_file(params, branches, dirpath):
    ensemble_results = {}
    for b in branches:
        f = os.path.join(dirpath, 'B%d-test.csv' % b)
        csv = pd.read_csv(f)
        acc_list = csv['before_mean_acc']
        assert len(acc_list) > 0, params
        ensemble_results[b] = [np.mean(acc_list), np.std(acc_list)]

    return ensemble_results

def get_vbranch_acc_from_file(params, branches, shared_frac, dirpath, metric, epoch):
    vbranch_results = {}
    for b in branches:
        vbranch_results[b] = {}
        for s in shared_frac:
            acc_list = []
            path = os.path.join(dirpath, f'B{b}', f'S{s:.2f}')
            print(path)
            for f in glob(path + '/train_*'):
                csv = pd.read_csv(f)
                acc_list.append(csv.iloc[epoch][f'{metric}_ensemble'])

            assert len(acc_list) > 0, params
            vbranch_results[b][s] = [np.mean(acc_list), np.std(acc_list)]

    return vbranch_results

def get_results(setup, *params, dirpath='results', p_count=0,
        branches=[], shared_frac=[], metric='val_acc', epoch=-1):

    if p_count == len(params):
        if setup == 'baseline':
            results = get_baseline_acc_from_file(params, dirpath, metric, epoch)
        # elif setup == 'ensemble':
        #     results = get_ensemble_acc_from_file(params, branches, dirpath)
        else:
            results = get_vbranch_acc_from_file(params, branches,
                shared_frac, dirpath, metric, epoch)
        return results

    results = {}
    config_list = glob(dirpath + f'/{params[p_count]}*')
    if len(config_list) == 0:
        raise ValueError(f'{dirpath} has no child directories')

    for config in config_list:
        p = config[len(dirpath)+len(f'/{params[p_count]}'):]
        results[p] = get_results(setup, *params, dirpath=config,
            p_count=p_count+1, branches=branches, shared_frac=shared_frac,
            epoch=epoch)

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    branches = range(2, args.max_branches+1)

    if args.params is None:
        params = []
    else:
        params = args.params

    baseline_results = get_results('baseline', *params,
        dirpath=os.path.join(args.dir, f'{args.dataset}-{args.architecture}'),
        metric=args.metric, epoch=args.epoch-1)

    # ensemble_results = get_results('ensemble', *params,
    #     dirpath=os.path.join(args.dir, f'{args.dataset}-{args.architecture}'),
    #     branches=branches)

    vbranch_results = get_results('vbranch', *params,
        dirpath=os.path.join(args.dir, f'vb-{args.dataset}-{args.architecture}'),
        branches=branches, shared_frac=args.shared_frac, metric=args.metric,
        epoch=args.epoch-1)

    results = {
        'baseline' : baseline_results,
        # 'ensemble' : ensemble_results,
        'vbranch' : vbranch_results
    }

    path = os.path.join(args.dir, f'{args.dataset}-{args.architecture}.json')

    with open(path, 'w') as fp:
        json.dump(results, fp, indent=4)

    print('Finished!')
