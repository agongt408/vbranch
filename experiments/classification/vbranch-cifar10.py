import sys
sys.path.insert(0, '.')

from vbranch.applications.densenet import DenseNet
from vbranch.callbacks import classification_acc
from vbranch.losses import softmax_cross_entropy_with_logits
from vbranch.utils import *
from vbranch.datasets.cifar10 import load_data

import tensorflow as tf
import numpy as np
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--depth', action='store', default=100, nargs='?',
                    type=int, help='densenet architecture depth param')
parser.add_argument('--growth_rate', action='store', default=12, nargs='?',
                    type=int, help='densenet architecture depth param')

parser.add_argument('--batch_size', action='store', default=64, nargs='?',
                    type=int, help='batch size')
parser.add_argument('--epochs', action='store', default=200, nargs='?',
                    type=int, help='number of epochs to train model')
parser.add_argument('--model_id',action='store',nargs='*',type=int,default=[1],
                    help='list of checkpoint model ids')
parser.add_argument('--num_branches', action='store', default=2, nargs='?',
                    type=int, help='number of virtual branches')
parser.add_argument('--shared_frac', action='store', default=0, nargs='?',
                    type=float, help='fraction of layer to share weights [0,1)')
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

def build_model(depth, growth_rate, x_shape, y_shape, batch_size, n_branches, shared, bagging):
    inputs, labels, train_init_ops, test_init_ops = get_data_iterator(x_shape,
        y_shape, batch_size=batch_size, n=n_branches, share_xy=(bagging==0))
    lr = tf.placeholder('float32', name='lr')
    lr_scheduler = lr_step_scheduler((100, 0.001), (150, 0.0001), (200, 0.00001))
    acc_clbk = classification_acc(n_branches, 10, batch_size=250)

    name = 'model'
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        model = DenseNet(depth, growth_rate, inputs, 
            name=name, shared_frac=shared)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        model.compile(optimizer, softmax_cross_entropy_with_logits(),
                      train_init_ops, test_init_ops, labels=labels,
                      callbacks={'acc': acc_clbk},
                      schedulers={'lr:0': lr_scheduler})

    return model

def train(depth, growth_rate, n_branches, model_id, epochs, batch_size, shared,
        path, train_frac, bagging, bootstrap):
    if path is None:
        arch_name = f'densenet-{depth}-{growth_rate}'
        model_path = get_vb_model_path('cifar10', arch_name, n_branches, shared,
            model_id=model_id)
        dirpath = get_vb_dir_path('cifar10', arch_name, n_branches, shared)
    else:
        path = os.path.join(path, f'B{n_branches}', f'S{shared:.2f}')
        model_path = os.path.join('models', path, f'model_{model_id}')
        if not os.path.isdir(model_path):
            # os.system('mkdir -p ' + model_path)
            os.system('mkdir ' + model_path)
        dirpath = path

    p_console('Save model path: ' + model_path)

    (X_train, y_train), (X_test, y_test) = load_data(preprocess=True, one_hot=True)
    x_shape = (None,) + X_train.shape[1:]
    y_shape = (None, 10)

    if bagging > 0:
        x_train_list, y_train_list = bag_samples(X_train, y_train, n_branches,
                                             max_samples=float(bagging),
                                             bootstrap=bootstrap)
        for i, (x_, y_) in enumerate(zip(x_train_list, y_train_list)):
            print('Bag {}:'.format(i+1), x_.shape, y_.shape)

    tf.reset_default_graph()
    model = build_model(depth, growth_rate, x_shape, y_shape, batch_size,
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

    val_dict = {'x:0': X_test, 'y:0': y_test, 'batch_size:0': 250}

    history = model.fit(epochs, len(X_train) // batch_size, 
        train_dict=train_dict, val_dict=val_dict, 
        log_path=model_path, verbose=1)

    save_results(history, dirpath, f'train_{model_id}.csv')

def test(depth, growth_rate, n_branches, model_id, shared):
    model_path = get_vb_model_path(dataset, arch, n_branches,
        shared, n_classes=10, model_id=model_id)

    p_console('Load model path: ' + model_path)

    _, (X_test, y_test) = load_data(preprocess=True, one_hot=True)

    tf.reset_default_graph()
    with TFSessionGrow() as sess:
        restore_sess(sess, model_path)
        acc_v, indiv_accs_v = vbranch_classification(sess, X_test, y_test,
            num_classes=10, mode='before', n_branches=n_branches)

    # print('Losses:', losses_v)
    print('Indiv accs:', indiv_accs_v)
    print('Ensemble acc:', acc_v)

    results_dict = {}
    for i in range(n_branches):
        # results_dict['loss_'+str(i+1)] = losses_v[i]
        results_dict['acc_'+str(i+1)] = indiv_accs_v[i]
    results_dict['acc_ensemble'] = acc_v

    arch_name = f'densenet-{depth}-{growth_rate}'
    dirpath = get_vb_dir_path('cifar10', arch_name, n_branches, shared)

    save_results(results_dict, dirpath, 'test.csv', mode='a')

if __name__ == '__main__':
    args = parser.parse_args()
    print('167> bootstrap', args.bootstrap)
    print('168> batch size', args.batch_size)

    if args.test:
        p_console('MODE: TEST')

        for id in args.model_id:
            test(args.depth, args.growth_rate, args.num_branches,id,
                args.shared_frac)
    else:
        p_console('MODE: TRAIN')

        if args.trials == 1:
            for id in args.model_id:
                train(args.depth, args.growth_rate, args.num_branches,id,
                    args.epochs, args.batch_size, args.shared_frac, 
                    args.path, args.train_frac, args.bagging, args.bootstrap)
        else:
            for i in range(args.trials):
                train(args.depth, args.growth_rate, args.num_branches,i+1,
                    args.epochs, args.batch_size, args.shared_frac, 
                    args.path, args.train_frac, args.bagging, args.bootstrap)

    print('Finished!')
