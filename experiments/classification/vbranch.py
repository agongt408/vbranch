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
parser.add_argument('--num_branches', action='store', default=2, nargs='?',
                    type=int, help='number of virtual branches')
parser.add_argument('--shared_frac', action='store', default=0, nargs='?',
                    type=float, help='fraction of layer to share weights [0,1)')
parser.add_argument('--steps_per_epoch', action='store', default=100, nargs='?',
                    type=int, help='number of training steps per epoch')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--trials', action='store', default=1, nargs='?', type=int,
                    help='number of trials to perform, if 1, then model_id used')

def get_data_as_tensor(train_data, test_data, num_branches, BATCH_SIZE):
    batch_size = tf.placeholder('int64', name='batch_size')

    train_datasets = []
    test_datasets = []
    inputs = [None] * args.num_branches
    labels_one_hot = [None] * args.num_branches
    train_init_ops = []
    test_init_ops = []

    for i in range(num_branches):
        train_datasets.append(tf.data.Dataset.from_tensor_slices(train_data).\
            batch(batch_size).repeat().shuffle(buffer_size=4*BATCH_SIZE))

        test_datasets.append(tf.data.Dataset.from_tensor_slices(test_data).\
            batch(batch_size))

        iterator = tf.data.Iterator.from_structure(train_datasets[i].output_types,
                                               train_datasets[i].output_shapes)
        inputs[i], labels_one_hot[i] = iterator.get_next()

        train_init_ops.append(iterator.make_initializer(train_datasets[i]))
        test_init_ops.append(iterator.make_initializer(test_datasets[i],
                                                    name='test_init_op_'+str(i+1)))

    return inputs, labels_one_hot, train_init_ops, test_init_ops, batch_size

def build_model(architecture,inputs,num_classes,num_branches,model_id,shared_frac):
    if architecture == 'fcn':
        model = vb.vbranch_simple_fcn(inputs,
            ([128]*num_branches, int(128*shared_frac)),
            ([10]*num_branches, int(10*shared_frac)),
            branches=num_branches, name='model_' + str(model_id))
    elif architecture == 'cnn':
        model = vb.vbranch_simple_cnn(inputs, (num_classes, 0),
            ([16]*num_branches, int(16*shared_frac)),
            ([32]*num_branches, int(32*shared_frac)),
            branches=num_branches, name='model_' + str(model_id))
    else:
        raise ValueError('invalid model')

    return model

def train(dataset, architecture, num_branches, model_id, num_classes, epochs,
        steps_per_epoch, BATCH_SIZE, shared_frac):

    if not os.path.isdir('models'):
        os.system('mkdir models')

    model_name = 'vb-{}-{}-B{:d}-S{:.2f}_{:d}'.format(dataset, architecture,
        num_branches, shared_frac, model_id)
    model_path = os.path.join('models', model_name)

    print(bcolors.HEADER+'Save model path: '+ model_path+ bcolors.ENDC)

    (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
        get_data(dataset, architecture, num_classes)

    tf.reset_default_graph()

    train_data = (X_train.astype('float32'), y_train_one_hot)
    test_data = (X_test.astype('float32'), y_test_one_hot)

    inputs, labels_one_hot, train_init_ops, test_init_ops, batch_size = \
        get_data_as_tensor(train_data, test_data, num_branches, BATCH_SIZE)

    # Build and compile model
    model = build_model(architecture, inputs, num_classes, num_branches,
        model_id, shared_frac)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    model.compile(optimizer, 'softmax_cross_entropy_with_logits',
        labels_one_hot=labels_one_hot)
    model.summary()

    # Run training ops
    train_loss_hist = [[] for i in range(num_branches)]
    train_acc_hist = [[] for i in range(num_branches)]
    indiv_accs_hist = [[] for i in range(num_branches)]
    val_loss_hist = [[] for i in range(num_branches)]
    val_acc_hist = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            start = time.time()

            # Training
            sess.run(train_init_ops, feed_dict={batch_size: BATCH_SIZE})
            for i in range(steps_per_epoch):
                _, train_losses, train_accs = sess.run([model.train_ops,
                    model.losses, model.train_accs])

            # Validation
            sess.run(test_init_ops,feed_dict={batch_size:len(X_test)})
            val_losses, val_acc, indiv_accs = \
                sess.run([model.losses,model.test_acc,model.train_accs])

            for b in range(num_branches):
                train_loss_hist[b].append(train_losses[b])
                train_acc_hist[b].append(train_accs[b])
                indiv_accs_hist[b].append(indiv_accs[b])
                val_loss_hist[b].append(val_losses[b])

            val_loss = np.mean(val_losses)
            val_acc_hist.append(val_acc)

            str_log = 'Time={:.0f}, '.format(time.time() - start)
            for b in range(num_branches):
                str_log += 'Loss {}={:.4f}, Acc {}={:.4f}, '.\
                    format(b+1,train_loss_hist[b][-1],b+1,train_acc_hist[b][-1])
                str_log += 'Val Loss {}={:.4f}, Val Acc {}={:.4f}, '.\
                    format(b+1, val_losses[b], b+1, indiv_accs[b])
            str_log += 'Val Loss={:.4f}, Val Acc={:.4f}'.format(val_loss,val_acc)

            print(str_log)

        saver = tf.train.Saver()
        path = os.path.join(model_path, 'ckpt')
        saver.save(sess, path)

    # Store loss/acc values as csv
    results_dict = {}
    for i in range(num_branches):
        results_dict['train_loss_'+str(i+1)] = train_loss_hist[i]
        results_dict['train_acc_'+str(i+1)] = train_acc_hist[i]
        results_dict['val_loss_'+str(i+1)] = val_loss_hist[i]
        results_dict['val_acc_'+str(i+1)] = indiv_accs_hist[i]
    results_dict['val_acc'] = val_acc_hist

    dirname = os.path.join('vb-{}-{}'.format(dataset, architecture),
        'B'+str(num_branches), 'S{:.2f}'.format(shared_frac))
    save_results(results_dict, dirname, 'train_{}.csv'.format(model_id))

def test(dataset,architecture,num_branches,model_id,shared_frac,num_classes):
    
    model_path = './models/vb-{}-{}-B{:d}-S{:.2f}_{:d}'.\
        format(dataset, architecture, num_branches, shared_frac, model_id)

    print(bcolors.HEADER + 'Load model path: ' + model_path + bcolors.ENDC)

    (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
        get_data(dataset, architecture, num_classes)

    test_init_ops = ['test_init_op_'+str(i+1) for i in range(num_branches)]
    losses = ['loss_'+str(i+1)+':0' for i in range(num_branches)]
    train_acc_ops = ['train_acc_'+str(i+1)+':0' for i in range(num_branches)]

    with tf.Session() as sess:
        meta_path = os.path.join(model_path, 'ckpt.meta')
        ckpt = tf.train.get_checkpoint_state(model_path)

        imported_graph = tf.train.import_meta_graph(meta_path)
        imported_graph.restore(sess, ckpt.model_checkpoint_path)

        sess.run(test_init_ops, feed_dict={'batch_size:0': len(X_test)})
        val_losses,val_acc,indiv_accs = sess.run([losses,'test_acc:0',
                                                  train_acc_ops])

    val_loss = np.mean(val_losses)
    print('Loss:', val_loss)
    print('Acc:', val_acc)
    print('Indiv accs:', indiv_accs)

    results_dict = {}
    for i in range(num_branches):
        results_dict['acc_'+str(i+1)] = indiv_accs[i]
    results_dict['ensemble_acc'] = val_acc

    dirname = os.path.join('vb-{}-{}'.format(dataset, architecture),
        'B'+str(num_branches), 'S{:.2f}'.format(shared_frac))
    save_results(results_dict, dirname, 'test.csv', mode='a')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(bcolors.HEADER + 'MODE: TEST' + bcolors.ENDC)

        for id in args.model_id:
            test(args.architecture,args.num_branches,id,args.shared_frac,
                args.num_classes)
    else:
        print(bcolors.HEADER + 'MODE: TRAIN' + bcolors.ENDC)

        if args.trials == 1:
            for id in args.model_id:
                train(args.dataset, args.architecture, args.num_branches, id,
                    args.num_classes, args.epochs, args.steps_per_epoch,
                    args.batch_size, args.shared_frac)
        else:
            for i in range(args.trials):
                train(args.dataset, args.architecture, args.num_branches, i+1,
                    args.num_classes, args.epochs, args.steps_per_epoch,
                    args.batch_size, args.shared_frac)

    print('Finished!')
