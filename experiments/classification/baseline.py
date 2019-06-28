import sys
sys.path.insert(0, '.')

from vbranch.applications.fcn import *
from vbranch.applications.cnn import *

from vbranch.utils.generic import TFSessionGrow, restore_sess, _dir_path, get_model_path, p_console, save_results
from vbranch.utils.training import get_data, get_data_iterator
from vbranch.utils.test.classification import compute_acc_from_logits, baseline_classification
from vbranch.callbacks import classification_acc
from vbranch.losses import softmax_cross_entropy_with_logits

import tensorflow as tf
import numpy as np
import os
import argparse
from glob import glob

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
parser.add_argument('--steps_per_epoch', action='store', default=100, nargs='?',
                    type=int, help='number of training steps per epoch')
parser.add_argument('--test', action='store_true', help='testing mode')
parser.add_argument('--trials', action='store', default=1, nargs='?', type=int,
                    help='number of trials to perform, if 1 then model_id used')
parser.add_argument('--train_frac', action='store', default=1., type=float,
                    help='fraction of original dataset to use for training')

parser.add_argument('--path', action='store', nargs='?', default=None,
                    help='manually specify path to save model checkpoint and results')

def build_model(architecture, n_classes, x_shape, y_shape, batch_size):
    inputs, labels, train_init_op, test_init_op = get_data_iterator(x_shape,
        y_shape, batch_size, n=1, share_xy=True)

    name = 'model'
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if architecture == 'fcn':
            model = SimpleFCNv1(inputs, n_classes, name=name)
        elif architecture == 'fcn2':
            model = SimpleFCNv2(inputs, n_classes, name=name)
        elif architecture == 'fcn3':
            model = SimpleFCNv3(inputs, n_classes, name=name)
        elif architecture == 'fcn4':
            model = SimpleFCNv4(inputs, n_classes, name=name)
        elif architecture == 'cnn':
            model = SimpleCNNSmall(inputs, n_classes, name=name)
        else:
            raise ValueError('Invalid architecture')

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        model.compile(optimizer, softmax_cross_entropy_with_logits(),
            train_init_op, test_init_op, labels=labels,
            callbacks={'acc':classification_acc(n_classes=n_classes)})

    return model

def train(dataset,arch,model_id,n_classes,n_features,samples_per_class,
        epochs, steps_per_epoch,batch_size, path):
    if path is None:
        model_path = get_model_path(dataset, arch, n_classes,
            samples_per_class,model_id)
        dirpath = _dir_path(dataset, arch, n_classes, samples_per_class)
    else:
        model_path = os.path.join('models', path, 'model_{}'.format(model_id))
        if not os.path.isdir(model_path):
            os.system('mkdir -p ' + model_path)
        dirpath = path

    p_console('Save model path: ' + model_path)

    # Load data
    (X_train, y_train), (X_test, y_test) = get_data(dataset, arch,
        n_classes, n_features, samples_per_class)
    x_shape = (None,) + X_train.shape[1:]
    y_shape = (None, n_classes)

    tf.reset_default_graph()
    model = build_model(arch, n_classes, x_shape, y_shape, batch_size)
    model.summary()

    train_dict = {'x:0': X_train, 'y:0': y_train, 'batch_size:0': batch_size}
    val_dict = {'x:0': X_test, 'y:0': y_test, 'batch_size:0': len(X_test)}
    history = model.fit(epochs, steps_per_epoch, train_dict=train_dict,
        val_dict=val_dict, log_path=model_path)

    save_results(history, dirpath, 'train_%d.csv' % model_id, mode='w')

def test(dataset,arch,model_id_list,n_classes,n_features,samples_per_class,
        output_dict={}, acc_dict={}):
    print(model_id_list)

    _, (X_test, y_test) = get_data(dataset, arch, n_classes,
        n_features, samples_per_class)

    test_outputs = []
    test_accs = []

    for model_id in model_id_list:
        if model_id in output_dict.keys():
            output = output_dict[model_id]
            acc = acc_dict[model_id]
        else:
            tf.reset_default_graph()
            with TFSessionGrow() as sess:
                model_path = get_model_path(dataset, arch, n_classes,
                    samples_per_class, model_id)
                restore_sess(sess, model_path)

                # Compute accuracy
                acc_dict[model_id], output_dict[model_id] = \
                    baseline_classification(sess, X_test, y_test,
                    num_classes=n_classes, return_logits=True)

        test_outputs.append(output_dict[model_id])
        test_accs.append(acc_dict[model_id])

    before_mean_acc = compute_acc_from_logits(test_outputs, y_test,
        num_classes=n_classes, mode='before')
    after_mean_acc = compute_acc_from_logits(test_outputs, y_test,
        num_classes=n_classes, mode='after')

    print('Individual accs:', test_accs)
    print('Before mean acc:', before_mean_acc)
    print('After mean acc:', after_mean_acc)

    results_dict = {}
    for i, model_id in enumerate(model_id_list):
        results_dict['model_id_'+str(i+1)] = model_id
        results_dict['acc_'+str(i+1)] = test_accs[i]
    results_dict['before_mean_acc'] = before_mean_acc
    results_dict['after_mean_acc'] = after_mean_acc

    dirpath = _dir_path(dataset, arch, n_classes, samples_per_class)
    save_results(results_dict, dirpath, 'B%d-test.csv'%len(model_id_list), 'a')

    return output_dict, acc_dict

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        p_console('MODE: TEST')

        if args.trials == 1:
            # args.model_id is a list of model ids
            test(args.dataset,args.architecture,args.model_id,args.num_classes,
                args.num_features, args.samples_per_class)
        else:
            # Store output, acc, and dict in case need to be reused
            output_dict = {}
            acc_dict = {}

            dirpath = os.path.join('models', _dir_path(args.dataset,
                args.architecture, args.num_classes, args.samples_per_class))
            avail_runs = glob(dirpath + '/model_*')
            avail_ids = [int(path[path.index('_')+1:]) for path in avail_runs]

            for i in range(args.trials):
                model_ids = np.random.choice(avail_ids, len(args.model_id),
                    replace=False)
                output_dict,acc_dict = test(args.dataset, args.architecture,
                    model_ids, args.num_classes, args.num_features,
                    args.samples_per_class, output_dict, acc_dict)
    else:
        p_console('MODE: TRAIN')

        if args.trials == 1:
            for model_id in args.model_id:
                # Run trial with specified model id
                train(args.dataset, args.architecture,model_id,args.num_classes,
                    args.num_features, args.samples_per_class, args.epochs,
                    args.steps_per_epoch, args.batch_size, args.path)
        else:
            # Run n trials with model id from 1 to args.trials
            for i in range(args.trials):
                train(args.dataset, args.architecture,i+1,args.num_classes,
                    args.num_features, args.samples_per_class, args.epochs,
                    args.steps_per_epoch, args.batch_size, args.path)

    print('Finished!')
