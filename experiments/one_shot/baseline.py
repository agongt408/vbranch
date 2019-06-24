import sys
sys.path.insert(0, '.')

import vbranch as vb
from vbranch.applications.cnn import *
from vbranch.applications.resnet import *

from vbranch.utils.generic_utils import restore_sess, _dir_path, get_model_path
from vbranch.utils.training_utils import p_console, save_results, get_data, get_data_iterator_from_generator
from vbranch.utils.test_utils import compute_one_shot_acc, baseline_one_shot
from vbranch.callbacks import one_shot_acc

import tensorflow as tf
import numpy as np
import os
from scipy.special import softmax
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
from glob import glob

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', default='omniglot',
                    nargs='?', choices=['omniglot'], help='dataset')
parser.add_argument('--architecture', action='store', default='cnn',
                    nargs='?', choices=['simple', 'res'],
                    help='model architecture, i.e., simple cnn or resnet')
parser.add_argument('--A',action='store',default=4,nargs='?',type=int,help='A')
parser.add_argument('--P',action='store',default=8,nargs='?',type=int,help='P')
parser.add_argument('--K',action='store',default=4,nargs='?',type=int,help='K')

parser.add_argument('--epochs', action='store', default=90, nargs='?',
                    type=int, help='number of epochs to train model')
parser.add_argument('--model_id',action='store',nargs='*',type=int,default=[1],
                    help='list of checkpoint model ids')
parser.add_argument('--steps_per_epoch', action='store', default=100, nargs='?',
                    type=int, help='number of training steps per epoch')
parser.add_argument('--test', action='store_true', help='testing mode')
parser.add_argument('--trials', action='store', default=1, nargs='?', type=int,
                    help='number of trials to perform, if 1, then model_id used')

def build_model(architecture, train_gen, input_dim, output_dim,
        lr_scheduler, **kwargs):

    inputs, train_init_op, test_init_op = get_data_iterator_from_generator(
        train_gen, input_dim, **kwargs)

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        if architecture == 'simple':
            model = SimpleCNNLarge(inputs, output_dim, name=name)
        elif architecture == 'res':
            model = ResNet18(inputs, output_dim, name=name)
        else:
            raise ValueError('Invalid architecture')

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # Compile model
        model.compile(optimizer, 'triplet_omniglot', train_init_op, test_init_op,
                      callbacks={'acc': one_shot_acc(n_branches=1)},
                      schedulers={'lr:0': lr_scheduler}, **kwargs)

    return model

def train(dataset, arch, model_id, epochs,steps_per_epoch, **kwargs):
    model_path = get_model_path(dataset, arch, model_id=model_id)
    p_console('Save model path: '+ model_path)

    tf.reset_default_graph()

    if dataset == 'omniglot':
        train_gen = vb.datasets.omniglot.load_generator(set='train')
        input_dim = [None, 105, 105, 1]
        output_dim = 128
        lr_scheduler = lr_exp_decay_scheduler(0.001, epochs//2, epochs, 0.001)

    model = build_model(architecture, train_gen, input_dim, output_dim,
        lr_scheduler, **kwargs)
    history = model.fit({}, epochs, steps_per_epoch, val_dict=None,
        log_path=model_path)
    save_results(history, dirpath, 'train_%d.csv' % model_id, mode='w')

def test(dataset, arch, model_id_list,train_dict={},test_dict={}, acc_dict={}):
    # Load data
    total_runs = 20
    run_data = [get_run(r+1) for r in range(total_runs)]

    baseline_acc_list = []
    model_train_runs = []
    model_test_runs = []
    model_accs = []

    for model_id in model_id_list:
        if model_id in train_dict.keys():
            train_runs = train_dict[model_id]
            test_runs = test_dict[model_id]
            acc = acc_dict[model_id]
        else:
            tf.reset_default_graph()
            model_path = get_model_path(dataset, arch, model_id=model_id)

            with TFSessionGrow() as sess:
                restore_sess(sess, model_path)
                acc, train_runs, test_runs = baseline_one_shot(sess,
                    return_outputs=True)

        model_train_runs.append(train_runs)
        model_test_runs.append(test_runs)
        model_accs.append(acc)

    # Average embedding
    mean_acc_runs = []
    test_embed = np.mean(model_test_runs, axis=0)
    train_embed = np.mean(model_train_runs, axis=0)

    for r in range(total_runs):
        train_files = run_data[r][0]
        answers_files = run_data[r][-1]
        acc = compute_one_shot_acc(test_embed[r], train_embed[r],train_files,
            answers_files)
        mean_acc_runs.append(acc)

    mean_acc = np.mean(mean_acc_runs)

    # Concatenate embedding
    concat_acc_runs = []
    test_embed = np.concatenate(model_test_runs, axis=-1)
    train_embed = np.concatenate(model_train_runs, axis=-1)

    for r in range(total_runs):
        train_files = run_data[r][0]
        answers_files = run_data[r][-1]
        acc = compute_one_shot_acc(test_embed[r], train_embed[r],train_files,
            answers_files)
        concat_acc_runs.append(acc)

    concat_acc = np.mean(concat_acc_runs)

    print(model_id_list)
    print('Individual accuracies:', model_accs)
    print('Average embedding acc:', mean_acc)
    print('Concatenate embedding acc:', concat_acc)

    results_dict = {'mean_acc' : mean_acc, 'concat_acc' : concat_acc}
    dirpath = _dir_path(dataset, arch)
    save_results(results_dict, dirpath, 'B%d-test.csv'%len(model_id_list), 'a')

    return train_dict, test_dict, acc_dict

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(bcolors.HEADER + 'MODE: TEST' + bcolors.ENDC)

        if args.trials == 1:
            # args.model_id is a list of model ids
            test(args.dataset, args.architecture, args.model_id)
        else:
            # Store outputs from runs
            train_dict = {}
            test_dict = {}

            avail_runs = glob('models/{}-{}_*'.format(args.dataset,
                args.architecture))
            avail_ids = [int(path[path.index('_')+1:]) for path in avail_runs]

            for i in range(args.trials):
                # Choose models for ensemble
                model_ids = np.random.choice(avail_ids, len(args.model_id),
                    replace=False)
                model_ids.sort()

                # Compute acc
                train_dict, test_dict = test(args.dataset, args.architecture,
                    model_ids, train_dict,test_dict)
    else:
        print(bcolors.HEADER + 'MODE: TRAIN' + bcolors.ENDC)

        if args.trials == 1:
            for id in args.model_id:
                # Run trial with specified model ids
                train(args.dataset, args.architecture,id,
                    args.epochs,args.steps_per_epoch,
                    A=args.A, P=args.P, K=args.K)
        else:
            # Run n trials with model id from 1 to args.trials
            for i in range(args.trials):
                train(args.dataset, args.architecture,i+1,
                    args.epochs,args.steps_per_epoch,
                    A=args.A, P=args.P, K=args.K)

    print('Finished!')
