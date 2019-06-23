import sys
sys.path.insert(0, '.')

import vbranch as vb
from vbranch.applications.cnn import *
from vbranch.applications.resnet import *

from vbranch.utils.generic_utils import TFSessionGrow, restore_sess, _vb_dir_path, get_vb_model_path
from vbranch.utils.training_utils import p_console, save_results, get_data, get_data_iterator_from_generator
from vbranch.utils.test_utils import compute_one_shot_acc, vbranch_one_shot
from vbranch.callbacks import one_shot_acc

import tensorflow as tf
import numpy as np
import os
import argparse
import time
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', default='omniglot',
                    nargs='?', choices=['omniglot'], help='dataset')
parser.add_argument('--architecture', action='store', default='simple',
                    nargs='?', choices=['simple', 'res'],
                    help='model architecture, i.e., simple cnn or resnet')
parser.add_argument('--A',action='store',default=4,nargs='?',type=int,help='A')
parser.add_argument('--P',action='store',default=8,nargs='?',type=int,help='P')
parser.add_argument('--K',action='store',default=4,nargs='?',type=int,help='K')

parser.add_argument('--num_branches', action='store', default=2, nargs='?',
                    type=int, help='number of virtual branches')
parser.add_argument('--shared_frac', action='store', default=0, nargs='?',
                    type=float, help='fraction of layer to share weights [0,1)')

parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--trials', action='store', default=1, nargs='?', type=int,
                    help='number of trials to perform, if 1, then model_id used')
parser.add_argument('--epochs', action='store', default=90, nargs='?',
                    type=int, help='number of epochs to train model')
parser.add_argument('--model_id',action='store',nargs='*',type=int,default=[1],
                    help='list of checkpoint model ids')
parser.add_argument('--steps_per_epoch', action='store', default=100, nargs='?',
                    type=int, help='number of training steps per epoch')
parser.add_argument('--m',action='store',nargs='?',help='msg in results file')

def build_model(architecture, train_gen, input_dim, output_dim,
        lr_scheduler, n_branches, shared, **kwargs):

    inputs, train_init_op, test_init_op = get_data_iterator_from_generator(
        train_gen, input_dim, **kwargs)

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        if architecture == 'simple':
            model = SimpleCNNLarge(inputs, output_dim, name=name, shared_frac=shared)
        elif architecture == 'res':
            model = ResNet18(inputs, output_dim, name=name, shared_frac=shared)
        else:
            raise ValueError('Invalid architecture')

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # Compile model
        model.compile(optimizer, 'triplet_omniglot', train_init_op, test_init_op,
                      callbacks={'acc': one_shot_acc(n_branches)},
                      schedulers={'lr:0': lr_scheduler}, **kwargs)

    return model

def train(dataset, architecture, n_branches, model_id, epochs,
        steps_per_epoch, shared_frac, **kwargs):
    model_path = get_model_path(dataset, arch, model_id=model_id)
    p_console('Save model path: '+ model_path)

    tf.reset_default_graph()

    if dataset == 'omniglot':
        train_gen = vb.datasets.omniglot.load_generator(set='train')
        input_dim = [None, 105, 105, 1]
        output_dim = 128
        lr_scheduler = lr_exp_decay_scheduler(0.001, epochs//2, epochs, 0.001)

    model = build_model(architecture, train_gen, input_dim, output_dim,
        lr_scheduler, n_branches, shared_frac, **kwargs)
    history = model.fit({}, epochs, steps_per_epoch, val_dict=None,
        log_path=model_path)
    save_results(history, dirpath, 'train_%d.csv' % model_id, mode='w')

def test(dataset, architecture, n_branches, model_id, shared_frac, message):
    model_path = get_vb_model_path(dataset, arch, model_id=model_id)
    tf.reset_default_graph()

    with TFSessionGrow() as sess:
        restore_sess(sess, model_path)
        # average_acc, average_baseline = vbranch_one_shot(sess, n_branches,
        #     mode='average')
        concat_acc, concat_baseline = vbranch_one_shot(sess, n_branches,
            mode='concat')

    print('Indiv accs:', concat_baseline)
    print('Ensemble acc:', acc_v)

    results_dict = {}
    for i in range(n_branches):
        results_dict['acc_'+str(i+1)] = concat_baseline[i]
    results_dict['acc_ensemble'] = concat_acc

    dirpath = _vb_dir_path(dataset, arch, n_branches, shared)
    save_results(results_dict, dirpath, 'test.csv', mode='a')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(bcolors.HEADER + 'MODE: TEST' + bcolors.ENDC)
        for id in args.model_id:
            test(args.dataset,args.architecture,args.num_branches,id,
                args.shared_frac,args.m)
    else:
        print(bcolors.HEADER + 'MODE: TRAIN' + bcolors.ENDC)

        if args.trials == 1:
            for id in args.model_id:
                train(args.dataset,args.architecture, args.num_branches, id,
                    args.A, args.P,args.K,args.epochs,args.steps_per_epoch,
                    args.shared_frac)
        else:
            for i in range(args.trials):
                train(args.dataset, args.architecture, args.num_branches, i+1,
                    args.A, args.P,args.K,args.epochs,args.steps_per_epoch,
                    args.shared_frac)

    print('Finished!')
