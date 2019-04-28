import sys
sys.path.insert(0, '.')

import vbranch as vb
from vbranch.utils import training_utils
from vbranch.utils.test_utils import restore_sess,get_run,compute_one_shot_acc

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

def get_data_as_tensor(train_generator, input_dim, A, P, K):
    def batch_gen(A, P, K):
        def func():
            while True:
                batch = train_generator.next(A, P, K)
                batch = batch.astype('float32')
                yield batch
        return func

    train_dataset = tf.data.Dataset.from_generator(batch_gen(A, P, K),'float32',
                                                 output_shapes=input_dim)

    # Dataset for feeding non-triplet batched images from memory
    x = tf.placeholder('float32', input_dim, name='x')
    batch_size = tf.placeholder('int64', name='batch_size')
    test_dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)

    iter_ = tf.data.Iterator.from_structure('float32', input_dim)
    train_init_op = iter_.make_initializer(train_dataset)
    test_init_op = iter_.make_initializer(test_dataset, name='test_init_op')

    inputs = iter_.get_next()

    return inputs, train_init_op, test_init_op

def build_model(architecture, inputs, output_dim, model_id):
    name = 'model_' + str(model_id)

    if architecture == 'simple':
        model = vb.simple_cnn(inputs, output_dim, 32, 64, 128, 256, name=name)
    elif architecture == 'res':
        model = vb.resnet(inputs, output_dim, 32, 64, 128, 256, name=name)
    else:
        raise ValueError('Invalid architecture')

    return model

def train(dataset, architecture, model_id, A, P, K, epochs,steps_per_epoch):
    if not os.path.isdir('models'):
        os.system('mkdir models')

    model_name = '{}-{}_{:d}'.format(dataset, architecture, model_id)
    model_path = os.path.join('models', model_name)

    print(training_utils.bcolors.HEADER+'Save model path: '+\
        model_path+training_utils.bcolors.ENDC)

    # Load data
    if dataset == 'omniglot':
        train_gen = vb.datasets.omniglot.load_generator(set='train')
        input_dim = [None, 105, 105, 1]
        output_dim = 128

    tf.reset_default_graph()

    inputs, train_init_op, test_init_op = get_data_as_tensor(train_gen,
        input_dim, A,P,K)

    # Build and compile model
    model = build_model(architecture, inputs, output_dim, model_id)

    lr = tf.placeholder('float32', name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    model.compile(optimizer, 'triplet_'+dataset, A=A, P=P, K=K)
    model.summary()

    # Train
    train_loss_hist = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init_op)

        lr_sched = training_utils.lr_exp_decay_scheduler(0.001,epochs//3,
            epochs,0.001)

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            progbar = tf.keras.utils.Progbar(steps_per_epoch, verbose=2)
            # start = time.time()

            # Training
            learning_rate = lr_sched(e + 1)
            for i in range(steps_per_epoch):
                _, loss_value = sess.run([model.train_op, model.loss],
                                         feed_dict={lr:learning_rate})
                progbar.update(i + 1, values=[('loss', loss_value),
                    ('lr', learning_rate)])

            # epoch_time = time.time() - start
            # print(('Time={:.0f}, Loss={:.4f}'.format(epoch_time,loss_value)))
            train_loss_hist.append(loss_value)

        saver = tf.train.Saver()
        path = os.path.join(model_path, 'ckpt')
        saver.save(sess, path)

    # Store loss/acc values as csv
    training_utils.save_results({'train_loss':train_loss_hist}, '{}-{}'.\
        format(dataset, architecture),'train_{}.csv'.format(model_id),mode='w')

def test(dataset, architecture, model_id_list,train_dict={},test_dict={}):

    print(model_id_list)

    # Load data
    total_runs = 20
    run_data = [get_run(r+1) for r in range(total_runs)]

    model_train_runs = []
    model_test_runs = []

    for id in model_id_list:
        if id in train_dict.keys() and id in test_dict.keys():
            train_runs = train_dict[id]
            test_runs = test_dict[id]
        else:
            train_runs = []
            test_runs = []

            graph = tf.Graph()
            sess = tf.Session(graph=graph)

            with sess.as_default(), graph.as_default():
                restore_sess(sess, './models/{}-{}_{}'.format(dataset,
                    architecture, id))

                for r in range(total_runs):
                    train_files,test_files,train_ims,test_ims,answers_files = \
                        run_data[r]

                    feed_dict = {'x:0':train_ims,'batch_size:0':len(train_ims)}
                    sess.run('test_init_op', feed_dict=feed_dict)
                    train_runs.append(sess.run('model_%d'%id+'/'+'output:0'))

                    feed_dict = {'x:0':test_ims, 'batch_size:0':len(test_ims)}
                    sess.run('test_init_op', feed_dict=feed_dict)
                    test_runs.append(sess.run('model_%d'%id+'/'+'output:0'))

            train_dict[id] = train_runs
            test_dict[id] = test_runs

        model_train_runs.append(train_runs)
        model_test_runs.append(test_runs)

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

    print('Average embedding acc:', mean_acc)
    print('Concatenate embedding acc:', concat_acc)

    results_dict = {}
    results_dict['mean_acc'] = mean_acc
    results_dict['concat_acc'] = concat_acc

    training_utils.save_results(results_dict, '{}-{}'.format(dataset,architecture),
        'B{}-test.csv'.format(len(model_id_list)), mode='a')

    return train_dict, test_dict

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(training_utils.bcolors.HEADER + 'MODE: TEST' + training_utils.bcolors.ENDC)

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
        print(training_utils.bcolors.HEADER + 'MODE: TRAIN' + training_utils.bcolors.ENDC)

        if args.trials == 1:
            for id in args.model_id:
                # Run trial with specified model ids
                train(args.dataset, args.architecture,id,args.A, args.P, args.K,
                    args.epochs,args.steps_per_epoch)
        else:
            # Run n trials with model id from 1 to args.trials
            for i in range(args.trials):
                train(args.dataset, args.architecture,i+1,args.A, args.P, args.K,
                    args.epochs,args.steps_per_epoch)

    print('Finished!')
