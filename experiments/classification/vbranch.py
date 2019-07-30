import sys
sys.path.insert(0, '.')

from vbranch.applications.fcn import *
from vbranch.applications.cnn import *
from vbranch.callbacks import classification_acc
from vbranch.losses import softmax_cross_entropy_with_logits
from vbranch.utils import *

import tensorflow as tf
import numpy as np
import os
import argparse

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
parser.add_argument('--train_frac', action='store', default=1., type=float,
                    help='fraction of original dataset to use for training')
parser.add_argument('--bagging', action='store', default=0, type=float,
                    help='fraction of dataset to sample for bagging')
parser.add_argument('--bootstrap', action='store_true',
                    help='if true, sample with replacement')

parser.add_argument('--path', action='store', nargs='?', default=None,
                    help='manually specify path to save model checkpoint and results')

def build_model(architecture, n_classes, x_shape, y_shape, batch_size,
        n_branches, shared, bagging):

    inputs, labels, train_init_ops, test_init_ops = get_data_iterator(x_shape,
        y_shape, batch_size=batch_size, n=n_branches, share_xy=(bagging==0))

    name = 'model'
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if architecture == 'fcn':
            model = SimpleFCNv1(inputs, n_classes, name=name, shared_frac=shared)
        elif architecture == 'fcn2':
            model = SimpleFCNv2(inputs, n_classes, name=name, shared_frac=shared)
        elif architecture == 'fcn3':
            model = SimpleFCNv3(inputs, n_classes, name=name, shared_frac=shared)
        elif architecture == 'fcn4':
            model = SimpleFCNv4(inputs, n_classes, name=name, shared_frac=shared)
        elif architecture == 'cnn':
            model = SimpleCNNSmall(inputs, n_classes, name=name, shared_frac=shared)
        else:
            raise ValueError('Invalid architecture')

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        model.compile(optimizer, softmax_cross_entropy_with_logits(),
                      train_init_ops, test_init_ops, labels=labels,
                      callbacks={'acc':classification_acc(n_branches, n_classes)})

    return model

def train(dataset, arch, n_branches, model_id, n_classes, n_features,
        samples_per_class, epochs, steps_per_epoch, batch_size, shared,
        path, train_frac, bagging, bootstrap):
    if path is None:
        model_path = get_vb_model_path(dataset, arch, n_branches, shared,
            n_classes, samples_per_class, model_id)
        dirpath = _vb_dir_path(dataset, arch, n_branches, shared,
            n_classes, samples_per_class)
    else:
        path = os.path.join(path, f'B{n_branches}', 'S{:.2f}'.format(shared))
        model_path = os.path.join('models', path, f'model_{model_id}')
        if not os.path.isdir(model_path):
            os.system('mkdir -p ' + model_path)
        dirpath = path

    p_console('Save model path: ' + model_path)

    (X_train, y_train), (X_test, y_test) = get_data(dataset,arch,n_classes,
                                            n_features, samples_per_class,
                                            train_frac=train_frac)
    x_shape = (None,) + X_train.shape[1:]
    y_shape = (None, n_classes)

    if bagging <= 1:
        x_train_list, y_train_list = bag_samples(X_train, y_train, n_branches,
                                             max_samples=float(bagging),
                                             bootstrap=bootstrap)
        for i, (x_, y_) in enumerate(zip(x_train_list, y_train_list)):
            print('Bag {}:'.format(i+1), x_.shape, y_.shape)

    tf.reset_default_graph()
    model = build_model(arch, n_classes, x_shape, y_shape, batch_size,
        n_branches, shared, bagging)
    model.summary()

    if n_branches == 1 or bagging == 0:
        train_dict = {'x:0': X_train, 'y:0': y_train, 'batch_size:0': batch_size}
    else:
        train_dict = {'x:0': X_train, 'y:0': y_train}
        for i in range(n_branches):
            train_dict['vb{}_x:0'.format(i+1)] = x_train_list[i]
            train_dict['vb{}_y:0'.format(i+1)] = y_train_list[i]
        train_dict['batch_size:0'] = batch_size

    val_dict = {'x:0': X_test, 'y:0': y_test, 'batch_size:0': len(X_test)}

    history = model.fit(epochs, steps_per_epoch, train_dict=train_dict,
        val_dict=val_dict, log_path=model_path)
    save_results(history, dirpath, 'train_{}.csv'.format(model_id))

def test(dataset, arch, n_branches, model_id, shared, n_classes,
        n_features, samples_per_class):

    model_path = get_vb_model_path(dataset, arch, n_branches,
        shared, n_classes, samples_per_class, model_id)

    p_console('Load model path: ' + model_path)

    _, (X_test, y_test) = get_data(dataset, arch, n_classes, n_features,
        samples_per_class)

    tf.reset_default_graph()

    with TFSessionGrow() as sess:
        restore_sess(sess, model_path)
        acc_v, indiv_accs_v = vbranch_classification(sess, X_test, y_test,
            num_classes=n_classes, mode='before', n_branches=n_branches)

    # print('Losses:', losses_v)
    print('Indiv accs:', indiv_accs_v)
    print('Ensemble acc:', acc_v)

    results_dict = {}
    for i in range(n_branches):
        # results_dict['loss_'+str(i+1)] = losses_v[i]
        results_dict['acc_'+str(i+1)] = indiv_accs_v[i]
    results_dict['acc_ensemble'] = acc_v

    dirpath = _vb_dir_path(dataset, arch, n_branches, shared,
        n_classes, samples_per_class)

    save_results(results_dict, dirpath, 'test.csv', mode='a')

if __name__ == '__main__':
    args = parser.parse_args()
    print('167> bootstrap', args.bootstrap)
    print('168> batch size', args.batch_size)

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
                    args.shared_frac, args.path, args.train_frac,
                    args.bagging, args.bootstrap)
        else:
            for i in range(args.trials):
                train(args.dataset, args.architecture, args.num_branches,i+1,
                    args.num_classes, args.num_features,args.samples_per_class,
                    args.epochs, args.steps_per_epoch,args.batch_size,
                    args.shared_frac, args.path, args.train_frac,
                    args.bagging, args.bootstrap)

    print('Finished!')
