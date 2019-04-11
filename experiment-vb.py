# MNIST FCN with Virtual Branching

import vbranch as vb

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
parser.add_argument('--m',action='store',nargs='?',help='msg in results file')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Load and preprocess data
def load_data(architecture, num_classes):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    if architecture == 'fcn':
        X_train = X_train.reshape([-1, 784])
        X_test = X_test.reshape([-1, 784])
    elif architecture == 'cnn':
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
    else:
        raise ValueError('Invalid architecture')

    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train_one_hot), (X_test, y_test_one_hot)

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
    with tf.variable_scope('model_' + str(model_id), reuse=tf.AUTO_REUSE):
        if architecture == 'fcn':
            model = vb.models.vbranch_fcn(inputs,
                ([128]*num_branches, int(128*shared_frac)),
                ([10]*num_branches, int(10*shared_frac)),
                branches=num_branches)
        elif architecture == 'cnn':
            model = vb.models.vbranch_cnn(inputs, num_classes,
                ([16]*num_branches, int(16*shared_frac)),
                ([32]*num_branches, int(32*shared_frac)),
                branches=num_branches)
        else:
            raise ValueError('Invalid architecture')
    return model

def get_shared_unshared_vars(num_branches):
    """
    Get shared variables (in order to later average gradients)
    and unshared variables (unique to each branch)"""

    shared_vars = []
    unshared_vars = [[] for i in range(num_branches)]

    for var in tf.global_variables():
        if 'shared_to_shared' in var.name:
            shared_vars.append(var)
        else:
            for i in range(num_branches):
                if 'vb'+str(i+1) in var.name:
                    unshared_vars[i].append(var)

    return shared_vars, unshared_vars

def get_train_ops(labels,logits,num_branches,optimizer,shared_vars,unshared_vars):
    losses = []
    # Store gradients from shared variables over each branch
    shared_grads = []
    unshared_train_ops = []

    for i in range(num_branches):
        loss = vb.losses.softmax_cross_entropy_with_logits(labels=labels[i],
            logits=logits[i], name='loss_'+str(i+1))
        losses.append(loss)

        # Compute gradients of shared vars for each branch (but don't apply)
        if len(shared_vars) > 0:
            shared_grads.append(optimizer.compute_gradients(loss,var_list=shared_vars))

        # Apply gradients for unshared vars for each branch
        if len(unshared_vars[i]) > 0:
            unshared_train_ops.append(optimizer.minimize(loss,var_list=unshared_vars[i]))

    # Take average of the gradients over each branch
    mean_shared_grads = []

    for v, var in enumerate(shared_vars):
        grad = tf.reduce_mean([shared_grads[i][v][0] for i in range(num_branches)],[0])
        mean_shared_grads.append((grad, var))

    if len(shared_vars) > 0:
        shared_train_op = optimizer.apply_gradients(mean_shared_grads)
    else:
        shared_train_op = []

    train_ops = [unshared_train_ops, shared_train_op]
    return train_ops, losses

def get_acc_ops(labels, logits, num_branches, num_classes):
    # Train accuracies
    train_acc_ops = []
    for i in range(num_branches):
        pred_max = tf.one_hot(tf.argmax(tf.nn.softmax(logits[i]), axis=-1),
                              num_classes)
        train_acc_ops.append(tf.reduce_mean(tf.reduce_sum(
            labels[i]*pred_max, [1]), name='train_acc_'+str(i+1)))

    # Test accuracy
    pred = tf.nn.softmax(tf.reduce_mean(logits, [0]))
    pred_max = tf.one_hot(tf.argmax(pred, axis=-1), num_classes)
    test_acc_op = tf.reduce_mean(tf.reduce_sum(labels[0]*pred_max, [1]),
        name='test_acc')

    return train_acc_ops, test_acc_op

def run_train_ops(epochs, steps_per_epoch, batch_size, num_branches,
        train_init_ops, test_init_ops, train_ops, losses, train_acc_ops,
        test_acc_op, model_path, BATCH_SIZE, TEST_BATCH_SIZE):

    train_loss_hist = [[] for i in range(num_branches)]
    train_acc_hist = [[] for i in range(num_branches)]
    val_loss_hist = []
    val_acc_hist = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            start = time.time()

            sess.run(train_init_ops, feed_dict={batch_size: BATCH_SIZE})

            for i in range(steps_per_epoch):
                _, train_losses, train_accs = sess.run([train_ops, losses,
                    train_acc_ops])

                if i == steps_per_epoch - 1:
                    sess.run(test_init_ops,feed_dict={batch_size:TEST_BATCH_SIZE})
                    val_losses, val_acc, indiv_accs = \
                        sess.run([losses, test_acc_op, train_acc_ops])

                    for b in range(num_branches):
                        train_loss_hist[b].append(train_losses[b])
                        train_acc_hist[b].append(train_accs[b])

                    val_loss = np.mean(val_losses)
                    val_loss_hist.append(val_loss)
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

    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist

def train(architecture, num_branches, model_id, num_classes, epochs,
        steps_per_epoch, BATCH_SIZE, shared_frac):

    model_name = 'vb-mnist-{}-B{:d}-S{:.2f}_{:d}'.format(architecture,
        num_branches, shared_frac, model_id)
    model_path = os.path.join('models', model_name)

    print(bcolors.HEADER + 'Save model path: ' + model_path + bcolors.ENDC)

    # Load data from MNIST
    (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
        load_data(architecture, num_classes)

    tf.reset_default_graph()

    train_data = (X_train.astype('float32'), y_train_one_hot)
    test_data = (X_test.astype('float32'), y_test_one_hot)

    inputs, labels_one_hot, train_init_ops, test_init_ops, batch_size = \
        get_data_as_tensor(train_data, test_data, num_branches, BATCH_SIZE)

    model = build_model(architecture, inputs, num_classes, num_branches,
        model_id, shared_frac)
    model.summary()

    # Declare training ops
    shared_vars, unshared_vars = get_shared_unshared_vars(num_branches)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_ops, losses = get_train_ops(labels_one_hot, model.output,
        num_branches, optimizer, shared_vars, unshared_vars)
    train_acc_ops, test_acc_op = get_acc_ops(labels_one_hot, model.output,
        num_branches, num_classes)

    # Run training ops
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        run_train_ops(epochs, steps_per_epoch, batch_size, num_branches,
            train_init_ops, test_init_ops, train_ops, losses, train_acc_ops,
            test_acc_op, model_path, BATCH_SIZE, len(X_test))

    # Store loss/acc values as csv
    results_dict = {}
    for i in range(num_branches):
        results_dict['train_loss_'+str(i+1)] = train_loss_hist[i]
        results_dict['train_acc_'+str(i+1)] = train_acc_hist[i]
    results_dict['val_loss'] = val_loss_hist
    results_dict['val_acc'] = val_acc_hist

    csv_path = pd.DataFrame(data=results_dict).to_csv(os.path.join('results',
        model_name+'-train.csv'))

def test(architecture, num_branches, model_id, shared_frac, message):
    model_path = './models/vb-mnist-{}-B{:d}-S{:.2f}_{:d}'.\
        format(architecture, num_branches, shared_frac, model_id)

    print(bcolors.HEADER + 'Load model path: ' + model_path + bcolors.ENDC)

    # Load data from MNIST
    (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
        load_data(architecture, 10)

    test_init_ops = ['test_init_op_'+str(i+1) for i in range(num_branches)]
    losses = ['loss_'+str(i+1)+':0' for i in range(num_branches)]
    train_acc_ops = ['train_acc_'+str(i+1)+':0' for i in range(num_branches)]

    with tf.Session() as sess:
        meta_path = os.path.join(model_path, 'ckpt.meta')
        ckpt = tf.train.get_checkpoint_state(model_path)

        imported_graph = tf.train.import_meta_graph(meta_path)
        imported_graph.restore(sess, ckpt.model_checkpoint_path)

        sess.run(test_init_ops, feed_dict={'batch_size:0': len(X_test)})
        val_losses,val_acc,indiv_accs = sess.run([losses,'test_acc:0',train_acc_ops])

    val_loss = np.mean(val_losses)
    print('Loss:', val_loss)
    print('Acc:', val_acc)
    print('Indiv accs:', indiv_accs)

    results_dict = {}
    for i in range(num_branches):
        results_dict['acc_'+str(i+1)] = indiv_accs[i]
    results_dict['ensemble_acc'] = val_acc
    results_dict['message'] = message

    csv_path = pd.DataFrame(data=results_dict, index=[0]).to_csv(os.path.join(
        'results', 'vb-mnist-{}-test.csv'.format(architecture)), mode='a')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(bcolors.HEADER + 'MODE: TEST' + bcolors.ENDC)

        for id in args.model_id:
            test(args.architecture,args.num_branches,id,args.shared_frac,args.m)
    else:
        print(bcolors.HEADER + 'MODE: TRAIN' + bcolors.ENDC)

        if args.trials == 1:
            for id in args.model_id:
                train(args.architecture, args.num_branches, id, 10,args.epochs,
                    args.steps_per_epoch,args.batch_size,args.shared_frac)
        else:
            for i in range(args.trials):
                train(args.architecture, args.num_branches, i+1, 10,args.epochs,
                    args.steps_per_epoch,args.batch_size,args.shared_frac)

    print('Finished!')
