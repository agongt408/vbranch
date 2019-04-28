# MNIST FCN with Virtual Branching

import sys
sys.path.insert(0, '.')

import vbranch as vb

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
                    help='number of trials to perform, if 1, then model_id used')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_data_as_tensor(train_data, test_data, BATCH_SIZE):
    batch_size = tf.placeholder('int64', name='batch_size')

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).\
        batch(batch_size).repeat().\
        shuffle(buffer_size=4*BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_data).\
        batch(batch_size).repeat()

    iter_ = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
    inputs, labels_one_hot = iter_.get_next()

    train_init_op = iter_.make_initializer(train_dataset)
    test_init_op = iter_.make_initializer(test_dataset, name='test_init_op')

    return inputs, labels_one_hot, train_init_op, test_init_op, batch_size

def build_model(architecture, inputs, num_classes, model_id):
    name = 'model_' + str(model_id)

    if architecture == 'fcn':
        model = vb.simple_fcn(inputs, 128, num_classes, name=name)
    elif architecture == 'cnn':
        model = vb.simple_cnn(inputs, num_classes, 16, 32, name=name)
    else:
        raise ValueError('Invalid architecture')

    return model

def train(architecture,model_id,num_classes,epochs,steps_per_epoch,BATCH_SIZE):
    if not os.path.isdir('models'):
        os.system('mkdir models')

    model_name = 'mnist-{}_{:d}'.format(architecture, model_id)
    model_path = os.path.join('models', model_name)

    print(bcolors.HEADER + 'Save model path: ' + model_path + bcolors.ENDC)

    # Load data from MNIST
    (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
        vb.datasets.mnist.load_data(format=architecture)

    tf.reset_default_graph()

    train_data = (X_train.astype('float32'), y_train_one_hot)
    test_data = (X_test.astype('float32'), y_test_one_hot)

    inputs, labels_one_hot, train_init_op, test_init_op, batch_size = \
        get_data_as_tensor(train_data, test_data, BATCH_SIZE)

    # Build and compile model
    model = build_model(architecture, inputs, num_classes, model_id)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    model.compile(optimizer, 'softmax_cross_entropy_with_logits',
        labels_one_hot=labels_one_hot)
    model.summary()

    # Train
    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            start = time.time()

            sess.run(train_init_op, feed_dict={batch_size: BATCH_SIZE})

            for i in range(steps_per_epoch):
                _, train_loss, train_acc = sess.run([model.train_op,
                    model.loss, model.acc])

                if i == steps_per_epoch - 1:
                    sess.run(test_init_op, feed_dict={batch_size:len(X_test)})

                    val_loss, val_acc = sess.run([model.loss, model.acc])

                    train_loss_hist.append(train_loss)
                    train_acc_hist.append(train_acc)
                    val_loss_hist.append(val_loss)
                    val_acc_hist.append(val_acc)

            epoch_time = time.time() - start

            print(('Time={:.0f}, Loss={:.4f}, Acc={:.4f}, ' + \
                'Val Loss={:.4f}, Val Acc={:.4f}').format(epoch_time,
                train_loss_hist[-1], train_acc_hist[-1], val_loss, val_acc))

        saver = tf.train.Saver()
        path = os.path.join(model_path, 'ckpt')
        saver.save(sess, path)

    # Store loss/acc values as csv
    _save_results({'train_loss':train_loss_hist,'train_acc':train_acc_hist,
        'val_loss':val_loss_hist, 'val_acc':val_acc_hist}, architecture,
        'train_{}.csv'.format(model_id), mode='w')

def test(architecture, model_id_list, num_classes,
        output_dict={}, acc_dict={}, loss_dict={}):

    # Compute accurary given class predictions and labels
    def compute_acc(pred, labels_one_hot, num_classes):
        pred_max = tf.keras.utils.to_categorical(np.argmax(pred, axis=-1),
            num_classes)
        return np.mean(np.sum(labels_one_hot*pred_max, axis=1))

    # Average predictions before softmax
    def compute_before_mean_acc(outputs, y_test_one_hot, num_classes):
        mean_output = softmax(np.array(outputs).mean(axis=0), axis=-1)
        return compute_acc(mean_output, y_test_one_hot, num_classes)

    # Average predictions after softmax
    def compute_after_mean_acc(outputs, y_test_one_hot, num_classes):
        mean_output = softmax(np.array(test_outputs), axis=-1).mean(axis=0)
        return compute_acc(mean_output, y_test_one_hot, num_classes)

    print(model_id_list)

    # Load data from MNIST
    (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
        vb.datasets.mnist.load_data(format=architecture)

    test_outputs = []
    test_accs = []
    test_losses = []

    for id in model_id_list:
        if id in output_dict.keys():
            output = output_dict[id]
            acc = acc_dict[id]
            loss = loss_dict[id]
        else:
            graph = tf.Graph()
            sess = tf.Session(graph=graph)

            with sess.as_default(), graph.as_default():
                model_path = 'models/mnist-{}_{}'.format(architecture, id)
                meta_path = os.path.join(model_path, 'ckpt.meta')
                ckpt = tf.train.get_checkpoint_state(model_path)

                imported_graph = tf.train.import_meta_graph(meta_path)
                imported_graph.restore(sess, ckpt.model_checkpoint_path)

                sess.run('test_init_op',feed_dict={'batch_size:0':len(X_test)})
                output,acc,loss = sess.run(['model_%s'%id+'/'+'output:0',
                                            'acc:0', 'loss:0'])

            output_dict[id] = output
            acc_dict[id] = acc
            loss_dict[id] = loss

        test_outputs.append(output)
        test_accs.append(acc)
        test_losses.append(loss)

    before_mean_acc = compute_before_mean_acc(test_outputs,y_test_one_hot,
        num_classes)
    after_mean_acc = compute_after_mean_acc(test_outputs, y_test_one_hot,
        num_classes)

    print('Individual accs:', test_accs)
    print('Before mean acc:', before_mean_acc)
    print('After mean acc:', after_mean_acc)

    results_dict = {}
    for i, id in enumerate(model_id_list):
        results_dict['acc_'+str(id)] = test_accs[i]
        results_dict['loss_'+str(id)] = test_losses[i]
    results_dict['before_mean_acc'] = before_mean_acc
    results_dict['after_mean_acc'] = after_mean_acc

    _save_results(results_dict, architecture, 'B{}-test.csv'.\
        format(len(model_id_list)), mode='a')

    return output_dict, acc_dict, loss_dict

def _save_results(data, architecture, filename, mode='w'):
    """Helper to save `data` dict to csv"""

    # Create folder to store csv
    csv_dir = os.path.join('results', 'mnist-{}'.format(architecture))
    if not os.path.isdir(csv_dir):
        os.system('mkdir -p ' + csv_dir)

    if mode == 'w':
        results = pd.DataFrame(data=data)
    elif mode == 'a':
        results = pd.DataFrame(data=data, index=[0])
    else:
        raise ValueError('invalid file I/O mode ("w" or "a")')

    csv_path = os.path.join(csv_dir, filename)
    results.to_csv(csv_path, mode=mode)

    return csv_path

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(bcolors.HEADER + 'MODE: TEST' + bcolors.ENDC)

        if args.trials == 1:
            # args.model_id is a list of model ids
            test(args.architecture, args.model_id, 10)
        else:
            # Store output, acc, and dict in case need to be reused
            output_dict = {}
            acc_dict = {}
            loss_dict = {}

            avail_runs = glob('models/mnist-{}_*'.format(args.architecture))
            avail_ids = [int(path[path.index('_')+1:]) for path in avail_runs]

            for i in range(args.trials):
                model_ids = np.random.choice(avail_ids, len(args.model_id),
                    replace=False)
                model_ids.sort()
                output_dict,acc_dict,loss_dict = test(args.architecture,
                            model_ids, 10, output_dict, acc_dict, loss_dict)
    else:
        print(bcolors.HEADER + 'MODE: TRAIN' + bcolors.ENDC)

        if args.trials == 1:
            for id in args.model_id:
                # Run trial with specified model id
                train(args.architecture,id,10,args.epochs,args.steps_per_epoch,
                    args.batch_size)
        else:
            # Run n trials with model id from 1 to args.trials
            for i in range(args.trials):
                train(args.architecture,i+1,10,args.epochs,args.steps_per_epoch,
                    args.batch_size)

    print('Finished!')
