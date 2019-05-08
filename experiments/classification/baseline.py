import sys
sys.path.insert(0, '.')

import vbranch as vb
from vbranch import utils

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
parser.add_argument('--steps_per_epoch', action='store', default=100, nargs='?',
                    type=int, help='number of training steps per epoch')
parser.add_argument('--test', action='store_true', help='testing mode')
parser.add_argument('--trials', action='store', default=1, nargs='?', type=int,
                    help='number of trials to perform, if 1 then model_id used')

def get_data_as_tensor(x_shape, y_shape, BATCH_SIZE):
    x = tf.placeholder('float32', x_shape, name='x')
    y = tf.placeholder('float32', y_shape, name='y')
    batch_size = tf.placeholder('int64', name='batch_size')

    dataset = tf.data.Dataset.from_tensor_slices((x,y)).\
        batch(batch_size).repeat().shuffle(buffer_size=4*BATCH_SIZE)

    iter_ = dataset.make_initializable_iterator()
    inputs, labels_one_hot = iter_.get_next('input')

    return inputs, labels_one_hot, iter_

def build_model(architecture, inputs, labels, num_classes,model_id,test=False):
    if architecture == 'fcn':
        model = vb.simple_fcn(inputs, 512, num_classes,
            name='model_'+str(model_id))
    elif architecture == 'cnn':
        model = vb.simple_cnn(inputs, num_classes, 16, 32,
            name='model_'+str(model_id))
    else:
        raise ValueError('Invalid architecture')

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    model.compile(optimizer, 'softmax_cross_entropy_with_logits',
                    labels_one_hot=labels, test=test)
    if not test:
        model.summary()

    return model

def train(dataset,arch,model_id,num_classes,num_features,samples_per_class,
        epochs, steps_per_epoch,batch_size):

    # Get path to save model
    model_path = _get_model_path(dataset, arch, num_classes, samples_per_class,
        model_id)
    print(utils.bcolors.HEADER+'Save model path: '+model_path+utils.bcolors.ENDC)

    # Load data
    (X_train,y_train),(X_test,y_test) = utils.get_data(dataset, arch,
        num_classes, num_features, samples_per_class)

    tf.reset_default_graph()

    # Convert data to iterator using Dataset API
    x_shape = (None,) + X_train.shape[1:]
    y_shape = (None, num_classes)
    inputs, labels, iterator = get_data_as_tensor(x_shape,y_shape,batch_size)

    model = build_model(arch, inputs, labels, num_classes, model_id)

    # Build copy of model for testing
    x_place = tf.placeholder('float32', x_shape, name='x_test')
    y_place = tf.placeholder('float32', y_shape, name='y_test')
    test_model = build_model(arch,x_place,y_place,num_classes,model_id,test=True)

    history = model.fit(iterator, X_train, y_train, epochs, steps_per_epoch,
        batch_size, validation=(X_test, y_test), test_model=test_model,
        save_model_path=model_path)

    dirpath = _get_dir_path(dataset, arch, num_classes, samples_per_class)
    utils.save_results(history, dirpath, 'train_%d.csv' % model_id, mode='w')

def test(dataset,arch,model_id_list,num_classes,num_features,samples_per_class,
        output_dict={}, acc_dict={}):

    print(model_id_list)

    _, (X_test, y_test) = utils.get_data(dataset, arch, num_classes,
        num_features, samples_per_class)

    test_outputs = []
    test_accs = []

    for model_id in model_id_list:
        if model_id in output_dict.keys():
            output = output_dict[model_id]
            acc = acc_dict[model_id]
        else:
            tf.reset_default_graph()

            with tf.Session() as sess:
                model_path = _get_model_path(dataset, arch, num_classes,
                    samples_per_class, model_id)
                meta_path = os.path.join(model_path, 'ckpt.meta')
                ckpt = tf.train.get_checkpoint_state(model_path)

                imported_graph = tf.train.import_meta_graph(meta_path)
                imported_graph.restore(sess, ckpt.model_checkpoint_path)

                output = sess.run('model_{}_1/output:0'.format(model_id),
                    feed_dict={'x_test:0':X_test})

            # Compute accuracy outside of the graph
            acc = utils.compute_acc(output, y_test, num_classes)

            output_dict[model_id] = output
            acc_dict[model_id] = acc

        test_outputs.append(output)
        test_accs.append(acc)

    before_mean_acc = utils.compute_before_mean_acc(test_outputs,y_test,num_classes)
    after_mean_acc = utils.compute_after_mean_acc(test_outputs,y_test,num_classes)

    print('Individual accs:', test_accs)
    print('Before mean acc:', before_mean_acc)
    print('After mean acc:', after_mean_acc)

    results_dict = {}
    for i, model_id in enumerate(model_id_list):
        results_dict['acc_'+str(model_id)] = test_accs[i]
    results_dict['before_mean_acc'] = before_mean_acc
    results_dict['after_mean_acc'] = after_mean_acc

    dirpath = _get_dir_path(dataset, arch, num_classes, samples_per_class)
    utils.save_results(results_dict, dirpath, 'B%d-test.csv' % \
        len(model_id_list), mode='a')

    return output_dict, acc_dict

def _get_dir_path(dataset, arch, num_classes, samples_per_class):
    if dataset == 'toy':
        # Further organize results by number of classes and samples_per_class
        dirpath = os.path.join('{}-{}'.format(dataset, arch),'C%d'%num_classes,
            'SpC%d' % samples_per_class)
    else:
        dirpath = '{}-{}'.format(dataset, arch)
    return dirpath

def _get_model_path(dataset, arch, num_classes, samples_per_class, model_id):
    # Get path to save model
    dirpath = _get_dir_path(dataset, arch, num_classes, samples_per_class)
    model_path = os.path.join('models', dirpath, 'model_%d' % model_id)

    if not os.path.isdir(model_path):
        os.system('mkdir -p ' + model_path)

    return model_path

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(utils.bcolors.HEADER + 'MODE: TEST' + utils.bcolors.ENDC)

        if args.trials == 1:
            # args.model_id is a list of model ids
            test(args.dataset,args.architecture,args.model_id,args.num_classes,
                args.num_features, args.samples_per_class)
        else:
            # Store output, acc, and dict in case need to be reused
            output_dict = {}
            acc_dict = {}

            dirpath = _get_dir_path(args.dataset, args.architecture,
                args.num_classes, args.samples_per_class)
            avail_runs = glob(dirpath + '/model_*')
            avail_ids = [int(path[path.index('_')+1:]) for path in avail_runs]

            for i in range(args.trials):
                model_ids = np.random.choice(avail_ids, len(args.model_id),
                    replace=False)
                output_dict,acc_dict = test(args.dataset, args.architecture,
                    model_ids, args.num_classes, args.num_features,
                    args.samples_per_class, output_dict, acc_dict)
    else:
        print(utils.bcolors.HEADER + 'MODE: TRAIN' + utils.bcolors.ENDC)

        if args.trials == 1:
            for model_id in args.model_id:
                # Run trial with specified model id
                train(args.dataset, args.architecture,model_id,args.num_classes,
                    args.num_features, args.samples_per_class, args.epochs,
                    args.steps_per_epoch, args.batch_size)
        else:
            # Run n trials with model id from 1 to args.trials
            for i in range(args.trials):
                train(args.dataset, args.architecture,i+1,args.num_classes,
                    args.num_features, args.samples_per_class, args.epochs,
                    args.steps_per_epoch, args.batch_size)

    print('Finished!')
