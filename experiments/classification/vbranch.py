import sys
sys.path.insert(0, '.')

import vbranch as vb
from vbranch.utils import bcolors, save_results, get_data

import tensorflow as tf
import numpy as np
import os
from scipy.special import softmax
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', default='mnist',
                    nargs='?', choices=['mnist', 'toy'], help='dataset')
parser.add_argument('--num_classes', action='store', default=10, nargs='?',
                    type=int, help='number of classes in toy dataset')
parser.add_argument('--num_features', action='store', default=784, nargs='?',
                    type=int, help='number of features in toy dataset')
parser.add_argument('--samples_per_class',action='store',default=1000,nargs='?',
                    type=int, help='samples per class')

parser.add_argument('--architecture', action='store', default='fcn',
                    nargs='?', choices=['fcn', 'cnn'],
                    help='model architecture, i.e., fcn or cnn')
parser.add_argument('--batch_size', action='store', default=32, nargs='?',
                    type=int, help='batch size')
parser.add_argument('--epochs', action='store', default=10, nargs='?',
                    type=int, help='number of epochs to train model')
parser.add_argument('--model_id',action='store',nargs='*',type=int,default=[1],
                    help='list of checkpoint model ids')
parser.add_argument('--num_branches', action='store', default=2, nargs='?',
                    type=int, help='number of virtual branches')
parser.add_argument('--shared_frac', action='store', default=0, nargs='?',
                    type=float, help='fraction of layer to share weights [0,1)')
parser.add_argument('--steps_per_epoch', action='store', default=100, nargs='?',
                    type=int, help='number of training steps per epoch')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--trials', action='store', default=1, nargs='?', type=int,
                    help='number of trials to perform, if 1, then model_id used')

def get_data_as_tensor(x_shape, y_shape, num_branches, BATCH_SIZE):
    x = tf.placeholder('float32', x_shape, name='x')
    y = tf.placeholder('float32', y_shape, name='y')
    batch_size = tf.placeholder('int64', name='batch_size')

    iterators = [None] * num_branches
    inputs = [None] * num_branches
    labels_one_hot = [None] * num_branches

    for i in range(num_branches):
        dataset = tf.data.Dataset.from_tensor_slices((x,y)).\
            batch(batch_size).repeat().shuffle(buffer_size=4*BATCH_SIZE)

        iterators[i] = dataset.make_initializable_iterator()
        inputs[i], labels_one_hot[i] = iterators[i].get_next('input')

    return inputs, labels_one_hot, iterators

def build_model(architecture,inputs,labels, num_classes,num_branches,model_id,
        shared_frac, test=False):

    if architecture == 'fcn':
        model = vb.vbranch_simple_fcn(inputs,
            ([512]*num_branches, int(512*shared_frac)),
            ([num_classes]*num_branches, int(num_classes*shared_frac)),
            branches=num_branches, name='model_' + str(model_id))
    elif architecture == 'cnn':
        model = vb.vbranch_simple_cnn(inputs, (num_classes, 0),
            ([16]*num_branches, int(16*shared_frac)),
            ([32]*num_branches, int(32*shared_frac)),
            branches=num_branches, name='model_' + str(model_id))
    else:
        raise ValueError('invalid model')

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    model.compile(optimizer, 'softmax_cross_entropy_with_logits',
                    labels_one_hot=labels, test=test)
    if not test:
        model.summary()

    return model

def train(dataset, arch, num_branches, model_id, num_classes, num_features,
        samples_per_class, epochs, steps_per_epoch, batch_size, shared_frac):

    model_path = _get_model_path(dataset, arch, num_branches, shared_frac,
        num_classes, samples_per_class, model_id)

    print(bcolors.HEADER+'Save model path: '+ model_path+ bcolors.ENDC)

    (X_train, y_train), (X_test, y_test) = get_data(dataset,arch,num_classes,
                                            num_features, samples_per_class)

    tf.reset_default_graph()

    # Convert data to iterator using Dataset API
    x_shape = (None,) + X_train.shape[1:]
    y_shape = (None, num_classes)
    inputs, labels_one_hot, iterators = \
        get_data_as_tensor(x_shape, y_shape, num_branches, batch_size)

    # Build and compile model
    model = build_model(arch, inputs, labels_one_hot, num_classes,
        num_branches, model_id, shared_frac)

    # Build copy of model for testing
    x_place = tf.placeholder('float32', x_shape, name='x_test')
    y_place = tf.placeholder('float32', y_shape, name='y_test')
    test_model = build_model(arch, x_place, [y_place] * num_branches,
        num_classes, num_branches, model_id, shared_frac, test=True)

    history = model.fit(iterators, X_train, y_train, epochs, steps_per_epoch,
        batch_size, validation=(X_test, y_test), test_model=test_model,
        save_model_path=model_path)

    dirpath = _get_dir_path(dataset, arch, num_branches, shared_frac,
        num_classes, samples_per_class)
    save_results(history, dirpath, 'train_{}.csv'.format(model_id))

def test(dataset,architecture,num_branches,model_id,shared_frac,
        num_classes, num_features, samples_per_class):

    model_path = _get_model_path(dataset, architecture, num_branches,
        shared_frac, num_classes, samples_per_class, model_id)

    print(bcolors.HEADER + 'Load model path: ' + model_path + bcolors.ENDC)

    (X_train, y_train), (X_test, y_test) = get_data(dataset, architecture,
        num_classes, num_features, samples_per_class)

    losses = ['loss_{}_1:0'.format(i+1) for i in range(num_branches)]
    indiv_accs = ['acc_{}_1:0'.format(i+1) for i in range(num_branches)]

    tf.reset_default_graph()

    with tf.Session() as sess:
        meta_path = os.path.join(model_path, 'ckpt.meta')
        ckpt = tf.train.get_checkpoint_state(model_path)

        imported_graph = tf.train.import_meta_graph(meta_path)
        imported_graph.restore(sess, ckpt.model_checkpoint_path)

        losses_v,acc_v,indiv_accs_v = sess.run([losses,'acc_ensemble_1:0',
            indiv_accs], feed_dict={'x_test:0': X_test, 'y_test:0': y_test})

    print('Losses:', losses_v)
    print('Indiv accs:', indiv_accs_v)
    print('Ensemble acc:', acc_v)

    results_dict = {}
    for i in range(num_branches):
        results_dict['loss_'+str(i+1)] = losses_v[i]
        results_dict['acc_'+str(i+1)] = indiv_accs_v[i]
    results_dict['acc_ensemble'] = acc_v

    dirpath = _get_dir_path(dataset, architecture, num_branches, shared_frac,
        num_classes, samples_per_class)
    save_results(results_dict, dirpath, 'test.csv', mode='a')

def _get_dir_path(dataset, arch, num_branches, shared_frac,
        num_classes, samples_per_class):

    if dataset == 'toy':
        # Further organize results by number of classes and samples_per_class
        dirpath = os.path.join('vb-{}-{}'.format(dataset, arch),
            'C%d'%num_classes, 'SpC%d' % samples_per_class, 'B%d'%num_branches,
            'S{:.2f}'.format(shared_frac))
    else:
        dirpath = 'vb-{}-{}'.format(dataset, arch)
    return dirpath

def _get_model_path(dataset, arch, num_branches, shared_frac, num_classes,
        samples_per_class, model_id):

    # Get path to save model
    dirpath = _get_dir_path(dataset, arch, num_branches, shared_frac,
        num_classes, samples_per_class)
    model_path = os.path.join('models', dirpath, 'model_%d' % model_id)

    if not os.path.isdir(model_path):
        os.system('mkdir -p ' + model_path)

    return model_path

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(bcolors.HEADER + 'MODE: TEST' + bcolors.ENDC)

        for id in args.model_id:
            test(args.dataset, args.architecture,args.num_branches,id,
                args.shared_frac, args.num_classes, args.num_features,
                args.samples_per_class)
    else:
        print(bcolors.HEADER + 'MODE: TRAIN' + bcolors.ENDC)

        if args.trials == 1:
            for id in args.model_id:
                train(args.dataset, args.architecture, args.num_branches,id,
                    args.num_classes, args.num_features,args.samples_per_class,
                    args.epochs, args.steps_per_epoch, args.batch_size,
                    args.shared_frac)
        else:
            for i in range(args.trials):
                train(args.dataset, args.architecture, args.num_branches,i+1,
                    args.num_classes, args.num_features,args.samples_per_class,
                    args.epochs, args.steps_per_epoch,args.batch_size,
                    args.shared_frac)

    print('Finished!')
