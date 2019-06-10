import sys
sys.path.insert(0, '.')

import vbranch as vb
from vbranch.utils.training_utils import p_console, save_results, get_data

import tensorflow as tf
import numpy as np
import os
import argparse
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
                    nargs='?', help='model architecture, i.e., fcn or cnn')
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

def get_iter(x_shape, y_shape, num_branches, BATCH_SIZE):
    x = tf.placeholder('float32', x_shape, name='x')
    y = tf.placeholder('float32', y_shape, name='y')
    batch_size = tf.placeholder('int64', name='batch_size')

    iterators = [None] * num_branches
    inputs = [None] * num_branches
    labels_one_hot = [None] * num_branches

    for i in range(num_branches):
        iterators[i] = tf.data.Dataset.from_tensor_slices((x,y)).\
            repeat().batch(batch_size).shuffle(buffer_size=4*BATCH_SIZE).\
            make_initializable_iterator()
        inputs[i], labels_one_hot[i] = iterators[i].get_next('input')

    return inputs, labels_one_hot, iterators

def build_model(architecture,inputs,labels, n_classes, n_branches,name,shared):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if architecture == 'fcn':
            model = vb.vbranch_simple_fcn(inputs,
                ([512]*n_branches, int(512*shared)),
                ([n_classes]*n_branches, int(n_classes*shared)),
                branches=n_branches, name=name)
        elif architecture == 'fcn2':
            model = vb.vbranch_simple_fcn(inputs,
                ([512]*n_branches, int(512*shared)),
                ([256]*n_branches, int(256*shared)),
                ([n_classes]*n_branches, int(n_classes*shared)),
                branches=n_branches, name=name)
        elif architecture == 'fcn3':
            model = vb.vbranch_simple_fcn(inputs,
                ([512]*n_branches, int(512*shared)),
                ([512]*n_branches, int(512*shared)),
                ([n_classes]*n_branches, int(n_classes*shared)),
                branches=n_branches, name=name)
        elif architecture == 'fcn4':
            model = vb.vbranch_simple_fcn(inputs,
                ([512]*n_branches, int(512*shared)),
                ([512]*n_branches, int(512*shared)),
                ([512]*n_branches, int(512*shared)),
                ([n_classes]*n_branches, int(n_classes*shared)),
                branches=num_branches, name=name)
        elif architecture == 'cnn':
            model = vb.vbranch_simple_cnn(inputs, (n_classes, 0),
                ([16]*n_branches, int(16*shared)),
                ([32]*n_branches, int(32*shared)),
                branches=n_branches, name=name)
        else:
            raise ValueError('invalid model')

        if type(labels) is list:
            labels_list = labels
        else:
            labels_list = [labels] * n_branches

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        model.compile(optimizer, 'softmax_cross_entropy_with_logits',
                        labels_one_hot=labels_list)

    return model

def train(dataset, arch, n_branches, model_id, n_classes, n_features,
        samples_per_class, epochs, steps_per_epoch, batch_size, shared):

    name = 'model_' + str(model_id)
    model_path = _model_path(dataset, arch, n_branches, shared,
        n_classes, samples_per_class, model_id)
    p_console('Save model path: ' + model_path)

    (X_train, y_train), (X_test, y_test) = get_data(dataset,arch,n_classes,
                                            n_features, samples_per_class)
    tf.reset_default_graph()

    # Convert data to iterator using Dataset API
    x_shape = (None,) + X_train.shape[1:]
    y_shape = (None, n_classes)
    inputs, labels, iterators = get_iter(x_shape,y_shape,n_branches,batch_size)

    # Build and compile model
    model = build_model(arch, inputs, labels, n_classes, n_branches, name, shared)

    # Build copy of model for testing
    x_place = tf.placeholder('float32', x_shape, name='x_test')
    y_place = tf.placeholder('float32', y_shape, name='y_test')

    model_copy = build_model(arch,x_place,y_place,n_classes,n_branches,name,shared)

    history = model.fit(iterators, X_train, y_train, epochs, steps_per_epoch,
        batch_size, validation=(X_test, y_test), test_model=model_copy,
        save_model_path=model_path)

    dirpath = _dir_path(dataset, arch, n_branches, shared,
        n_classes, samples_per_class)

    save_results(history, dirpath, 'train_{}.csv'.format(model_id))

def test(dataset, arch, n_branches, model_id, shared, n_classes,
        n_features, samples_per_class):

    model_path = _model_path(dataset, arch, n_branches,
        shared, n_classes, samples_per_class, model_id)

    p_console('Load model path: ' + model_path)

    (X_train, y_train), (X_test, y_test) = get_data(dataset, arch,
        n_classes, n_features, samples_per_class)

    losses = []
    indiv_accs = []
    for i in range(n_branches):
        losses.append('model_{}_1/loss_{}:0'.format(model_id, i+1))
        indiv_accs.append('model_{}_1/acc_{}:0'.format(model_id, i+1))

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
    for i in range(n_branches):
        results_dict['loss_'+str(i+1)] = losses_v[i]
        results_dict['acc_'+str(i+1)] = indiv_accs_v[i]
    results_dict['acc_ensemble'] = acc_v

    dirpath = _dir_path(dataset, arch, n_branches, shared,
        n_classes, samples_per_class)

    save_results(results_dict, dirpath, 'test.csv', mode='a')

def _dir_path(dataset,arch,n_branches,shared,n_classes,samples_per_class):
    if dataset == 'toy':
        # Further organize results by number of classes and samples_per_class
        dirpath = os.path.join('vb-{}-{}'.format(dataset, arch),
            'C%d'%n_classes, 'SpC%d' % samples_per_class, 'B%d'%n_branches,
            'S{:.2f}'.format(shared))
    else:
        dirpath = os.path.join('vb-{}-{}'.format(dataset, arch),
            'B%d'%n_branches, 'S{:.2f}'.format(shared))
    return dirpath

def _model_path(dataset, arch, n_branches, shared, n_classes,
        samples_per_class, model_id):
    # Get path to save model
    dirpath = _dir_path(dataset, arch, n_branches, shared,
        n_classes, samples_per_class)
    model_path = os.path.join('models', dirpath, 'model_%d' % model_id)

    if not os.path.isdir(model_path):
        os.system('mkdir -p ' + model_path)

    return model_path

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        p_console('MODE: TEST')

        for id in args.model_id:
            test(args.dataset, args.architecture,args.num_branches,id,
                args.shared_frac, args.num_classes, args.num_features,
                args.samples_per_class)
    else:
        p_console('MODE: TRAIN')

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
