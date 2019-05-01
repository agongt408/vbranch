import sys
sys.path.insert(0, '.')

import vbranch as vb
from vbranch.utils import training_utils
from vbranch.utils.test_utils import restore_sess,get_run,compute_one_shot_acc

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

def get_data_as_tensor(train_generator, num_branches, input_dim, A, P, K):
    def batch_gen(A, P, K):
        def func():
            while True:
                batch = train_generator.next(A, P, K)
                batch = batch.astype('float32')
                yield batch
        return func

    # Placeholder for feeding test images
    x = tf.placeholder('float32', input_dim, name='x')
    batch_size = tf.placeholder('int64', name='batch_size')

    train_datasets = []
    test_datasets = []
    inputs = [None] * num_branches
    train_init_ops = []
    test_init_ops = []

    for i in range(num_branches):
        train_datasets.append(tf.data.Dataset.from_generator(batch_gen(A, P, K),
                                'float32', output_shapes=input_dim))

        test_datasets.append(tf.data.Dataset.from_tensor_slices(x).\
            batch(batch_size))

        iterator = tf.data.Iterator.from_structure('float32', input_dim)
        inputs[i] = iterator.get_next(name='input_'+str(i+1))

        train_init_ops.append(iterator.make_initializer(train_datasets[i]))
        test_init_ops.append(iterator.make_initializer(test_datasets[i],
                            name='test_init_op_'+str(i+1)))

    return inputs, train_init_ops, test_init_ops

def build_model(architecture,inputs,output_dim,num_branches,model_id,shared_frac):
    if architecture == 'simple':
        filters = [32, 64, 128, 256]
        layers_spec = [([f]*num_branches, int(f*shared_frac)) for f in filters]

        model = vb.vbranch_simple_cnn(inputs, (output_dim, 0), *layers_spec,
            branches=num_branches, name='model_' + str(model_id))
    else:
        raise ValueError('invalid model')

    return model

def train(dataset, architecture, num_branches, model_id, A, P, K, epochs,
        steps_per_epoch, shared_frac):

    if not os.path.isdir('models'):
        os.system('mkdir models')

    model_name = 'vb-{}-{}-B{:d}-S{:.2f}_{:d}'.format(dataset,architecture,
        num_branches, shared_frac, model_id)
    model_path = os.path.join('models', model_name)

    print(training_utils.bcolors.HEADER + 'Save model path: ' + \
        model_path + training_utils.bcolors.ENDC)

    # Load data from MNIST
    if dataset == 'omniglot':
        train_gen = vb.datasets.omniglot.load_generator(set='train')
        input_dim = [None, 105, 105, 1]
        output_dim = 128

    tf.reset_default_graph()

    inputs, train_init_ops, test_init_ops = \
        get_data_as_tensor(train_gen, num_branches, input_dim, A, P, K)

    # Build and compile model
    model = build_model(architecture,inputs,output_dim,num_branches,
        model_id, shared_frac)

    lr = tf.placeholder('float32', name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    model.compile(optimizer, 'triplet_'+dataset, A=A, P=P, K=K)
    model.summary()

    # Run training ops
    train_loss_hist = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init_ops)

        lr_sched = training_utils.lr_exp_decay_scheduler(0.001,epochs//3,
            epochs,0.001)

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            progbar = tf.keras.utils.Progbar(steps_per_epoch, verbose=2)

            learning_rate = lr_sched(e + 1)
            for i in range(steps_per_epoch):
                _, loss_values = sess.run([model.train_ops, model.losses],
                                            feed_dict={lr:learning_rate})
                train_loss_hist.append(loss_values)

                # Update progress bar
                values = []
                for b in range(len(loss_values)):
                    values.append(('loss_'+str(b+1), loss_values[b]))
                values += [('lr', learning_rate),]
                progbar.update(i + 1, values=values)

        saver = tf.train.Saver()
        path = os.path.join(model_path, 'ckpt')
        saver.save(sess, path)

    train_loss_hist = np.array(train_loss_hist)

    # Store loss/acc values as csv
    results_dict = {}
    for i in range(num_branches):
        results_dict['train_loss_'+str(i+1)] = train_loss_hist[:, i]

    dirname = os.path.join('vb-{}-{}'.format(dataset, architecture),
        'B'+str(num_branches), 'S{:.2f}'.format(shared_frac))
    training_utils.save_results(results_dict, dirname,
        'train_{}.csv'.format(model_id))

def test(dataset, architecture, num_branches, model_id, shared_frac, message):
    model_path = './models/vb-{}-{}-B{:d}-S{:.2f}_{:d}'.\
        format(dataset, architecture, num_branches, shared_frac, model_id)

    print(training_utils.bcolors.HEADER + 'Load model path: ' + \
        model_path + training_utils.bcolors.ENDC)

    test_init_ops = ['test_init_op_'+str(i+1) for i in range(num_branches)]
    model_outputs = ['model_1/output_vb{}:0'.format(i+1) \
        for i in range(num_branches)]

    total_runs = 20
    train_pred_runs = []
    test_pred_runs = []

    run_data = [get_run(r+1) for r in range(total_runs)]

    with tf.Session() as sess:
        restore_sess(sess, model_path)

        for r in range(total_runs):
            train_files,test_files,train_imgs,test_imgs,answers_files = \
                run_data[r]

            feed_dict = {'x:0':train_imgs, 'batch_size:0':len(train_imgs)}
            sess.run(test_init_ops, feed_dict=feed_dict)
            train_pred_runs.append(sess.run(model_outputs))

            feed_dict = {'x:0':test_imgs, 'batch_size:0':len(test_imgs)}
            sess.run(test_init_ops, feed_dict=feed_dict)
            test_pred_runs.append(sess.run(model_outputs))

    mean_acc_runs = []
    for r in range(total_runs):
        test_embed = np.mean(test_pred_runs[r], axis=0)
        train_embed = np.mean(train_pred_runs[r], axis=0)
        train_files = run_data[r][0]
        answers_files = run_data[r][-1]

        acc = compute_one_shot_acc(test_embed, train_embed,train_files,
            answers_files)
        mean_acc_runs.append(acc)
    mean_acc = np.mean(mean_acc_runs)

    concat_acc_runs = []
    for r in range(total_runs):
        test_embed = np.concatenate(test_pred_runs[r], axis=-1)
        train_embed = np.concatenate(train_pred_runs[r], axis=-1)
        train_files = run_data[r][0]
        answers_files = run_data[r][-1]

        acc = compute_one_shot_acc(test_embed, train_embed,train_files,
            answers_files)
        concat_acc_runs.append(acc)
    concat_acc = np.mean(concat_acc_runs)

    print('Average embedding acc:', mean_acc)
    print('Concatenate embedding acc:', concat_acc)

    results_dict = {'mean_acc' : mean_acc, 'concat_acc' : concat_acc}

    dirname = os.path.join('vb-{}-{}'.format(dataset, architecture),
        'B'+str(num_branches), 'S{:.2f}'.format(shared_frac))
    training_utils.save_results(results_dict, dirname, 'test.csv', mode='a')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(training_utils.bcolors.HEADER + 'MODE: TEST' + \
            training_utils.bcolors.ENDC)

        for id in args.model_id:
            test(args.dataset,args.architecture,args.num_branches,id,
                args.shared_frac,args.m)
    else:
        print(training_utils.bcolors.HEADER + 'MODE: TRAIN' + \
            training_utils.bcolors.ENDC)

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
