import sys
sys.path.insert(0, '.')

from vbranch.applications.cnn import *
from vbranch.applications.resnet import *
from vbranch.datasets import omniglot
from vbranch.callbacks import one_shot_acc
from vbranch.losses import triplet_omniglot
from vbranch.utils import *

import tensorflow as tf
import numpy as np
import os
import argparse

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
        lr_scheduler, n_branches, shared, A, P, K):

    inputs, train_init_op, test_init_op = get_data_iterator_from_generator(
        train_gen, input_dim, n=n_branches)
    lr = tf.placeholder('float32', name='lr')

    name = 'model'
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if architecture == 'simple':
            model = SimpleCNNLarge(inputs, output_dim, name=name, shared_frac=shared)
        elif architecture == 'res':
            model = ResNet18(inputs, output_dim, name=name, shared_frac=shared)
        else:
            raise ValueError('Invalid architecture')

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        model.compile(optimizer, triplet_omniglot(A, P, K),
                      train_init_op, test_init_op,
                      callbacks={'acc': one_shot_acc(n_branches)},
                      schedulers={'lr:0': lr_scheduler})
    return model

def train(dataset, arch, n_branches, model_id, epochs,
        steps_per_epoch, shared_frac, A, P, K):
    model_path = get_vb_model_path(dataset, arch, n_branches, shared_frac,
        model_id=model_id)
    p_console('Save model path: '+ model_path)

    tf.reset_default_graph()

    if dataset == 'omniglot':
        train_gen = omniglot.load_generator('train', A, P, K)
        input_dim = [None, 105, 105, 1]
        output_dim = 128
        lr_scheduler = lr_exp_decay_scheduler(0.001, 2*epochs//3, epochs, 0.001)

    model = build_model(arch, train_gen, input_dim, output_dim,
        lr_scheduler, n_branches, shared_frac, A, P, K)
    model.summary()

    history = model.fit(epochs, steps_per_epoch, log_path=model_path)
    dirpath = get_vb_dir_path(dataset, arch, n_branches, shared_frac)
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
        p_console('MODE: TEST')
        for id in args.model_id:
            test(args.dataset,args.architecture,args.num_branches,id,
                args.shared_frac,args.m)
    else:
        p_console('MODE: TRAIN')

        if args.trials == 1:
            for id in args.model_id:
                train(args.dataset,args.architecture, args.num_branches, id,
                    args.epochs,args.steps_per_epoch, args.shared_frac,
                    A=args.A, P=args.P, K=args.K)
        else:
            for i in range(args.trials):
                train(args.dataset, args.architecture, args.num_branches, i+1,
                    args.epochs,args.steps_per_epoch, args.shared_frac,
                    A=args.A, P=args.P, K=args.K)

    print('Finished!')
