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
# Number of classes used only when generating toy dataset
parser.add_argument('--num_classes', action='store', default=10, nargs='?',
                    help='number of classes in toy dataset')

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
        model = vb.simple_fcn(inputs,128,num_classes,
            name='model_'+str(model_id))
    elif architecture == 'cnn':
        model = vb.simple_cnn(inputs,num_classes,16,32,
            name='model_'+str(model_id))
    else:
        raise ValueError('Invalid architecture')

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    model.compile(optimizer, 'softmax_cross_entropy_with_logits',
                    labels_one_hot=labels, test=test)
    if not test:
        model.summary()

    return model

def train(dataset,arch,model_id,num_classes,epochs,steps_per_epoch,batch_size):
    # Ensure folder exists to store saved model checkpoints
    if not os.path.isdir('models'):
        os.system('mkdir models')

    # Save model path
    model_path = os.path.join('models','{}-{}_{:d}'.format(dataset,arch,model_id))
    print(utils.bcolors.HEADER+'Saved path: '+model_path+utils.bcolors.ENDC)

    # Load data
    (X_train,y_train),(X_test,y_test) = utils.get_data(dataset,arch,num_classes)

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

    utils.save_results(history, '{}-{}'.format(dataset, arch),
        'train_{}.csv'.format(model_id), mode='w')

def test(dataset,arch,model_id_list,num_classes,output_dict={},acc_dict={}):
    print(model_id_list)

    _, (X_test, y_test) = get_data(dataset, arch, num_classes)

    test_outputs = []
    test_accs = []

    for id in model_id_list:
        if id in output_dict.keys():
            output = output_dict[id]
            acc = acc_dict[id]
        else:
            with tf.Session() as sess:
                model_path = 'models/{}-{}_{}'.format(dataset, arch, id)
                meta_path = os.path.join(model_path, 'ckpt.meta')
                ckpt = tf.train.get_checkpoint_state(model_path)

                imported_graph = tf.train.import_meta_graph(meta_path)
                imported_graph.restore(sess, ckpt.model_checkpoint_path)

                output = sess.run('model_{}_1/output:0'.format(id),
                    feed_dict={'x_test:0':X_test})

            # Compute accuracy outside of the Graph
            acc = compute_acc(output, y_test, num_classes)

            output_dict[id] = output
            acc_dict[id] = acc

        test_outputs.append(output)
        test_accs.append(acc)

    before_mean_acc = compute_before_mean_acc(test_outputs,y_test, num_classes)
    after_mean_acc = compute_after_mean_acc(test_outputs, y_test, num_classes)

    print('Individual accs:', test_accs)
    print('Before mean acc:', before_mean_acc)
    print('After mean acc:', after_mean_acc)

    results_dict = {}
    for i, id in enumerate(model_id_list):
        results_dict['acc_'+str(id)] = test_accs[i]
    results_dict['before_mean_acc'] = before_mean_acc
    results_dict['after_mean_acc'] = after_mean_acc

    utils.save_results(results_dict, '{}-{}'.format(dataset, arch),
        'B{}-test.csv'.format(len(model_id_list)), mode='a')

    return output_dict, acc_dict, loss_dict

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(utils.bcolors.HEADER + 'MODE: TEST' + utils.bcolors.ENDC)

        if args.trials == 1:
            # args.model_id is a list of model ids
            test(args.dataset,args.architecture,args.model_id,args.num_classes)
        else:
            # Store output, acc, and dict in case need to be reused
            output_dict = {}
            acc_dict = {}

            avail_runs = glob('models/{}-{}_*'.format(args.dataset,
                args.architecture))
            avail_ids = [int(path[path.index('_')+1:]) for path in avail_runs]

            for i in range(args.trials):
                model_ids = np.random.choice(avail_ids, len(args.model_id),
                    replace=False)
                output_dict,acc_dict = test(args.dataset, args.architecture,
                    model_ids,args.num_classes,output_dict, acc_dict)
    else:
        print(utils.bcolors.HEADER + 'MODE: TRAIN' + utils.bcolors.ENDC)

        if args.trials == 1:
            for id in args.model_id:
                # Run trial with specified model id
                train(args.dataset, args.architecture,id,args.num_classes,
                    args.epochs,args.steps_per_epoch, args.batch_size)
        else:
            # Run n trials with model id from 1 to args.trials
            for i in range(args.trials):
                train(args.dataset, args.architecture,i+1,args.num_classes,
                    args.epochs,args.steps_per_epoch, args.batch_size)

    print('Finished!')
