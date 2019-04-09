# MNIST FCN with Virtual Branching

import vbranch as vb

import tensorflow as tf
import numpy as np
import os
from scipy.special import softmax
import matplotlib.pyplot as plt
import argparse
import time

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--architecture', action='store', default='fcn',
                    nargs='?', choices=['fcn', 'cnn'],
                    help='model architecture, i.e., fcn or cnn')
parser.add_argument('--batch_size', action='store', default=32, nargs='?',
                    type=int, help='batch size')
parser.add_argument('--epochs', action='store', default=10, nargs='?',
                    type=int, help='number of epochs to train model')
parser.add_argument('--model_id', action='store', default=1, nargs='?',
                    type=int, help='model id of checkpoint')
parser.add_argument('--steps_per_epoch', action='store', default=100, nargs='?',
                    type=int, help='number of training steps per epoch')
parser.add_argument('--test', action='store', nargs='*',
                    help='model ids of stored models to test in ensemble')

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
    with tf.variable_scope('model_' + str(model_id), reuse=tf.AUTO_REUSE):
        if architecture == 'fcn':
            model = vb.models.simple_fcn(inputs, 128, num_classes)
        elif architecture == 'cnn':
            model = vb.models.simple_cnn(inputs, num_classes, 16, 32)
        else:
            raise ValueError('Invalid architecture')
    return model

def get_train_ops(labels, logits, optimizer):
    loss = vb.losses.softmax_cross_entropy_with_logits(labels=labels,
        logits=logits, name='loss')
    train_op = optimizer.minimize(loss)
    return train_op, loss

def get_acc_ops(labels, logits, num_classes):
    pred = tf.nn.softmax(logits, name='pred')
    pred_max = tf.one_hot(tf.argmax(pred, axis=-1), num_classes)
    acc = tf.reduce_mean(tf.reduce_sum(labels*pred_max, [1]), name='acc')
    return acc

def run_train_ops(epochs, steps_per_epoch, batch_size,
        train_init_op, test_init_op, train_op, loss, acc, model_path,
        BATCH_SIZE, TEST_BATCH_SIZE):

    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            # progbar = tf.keras.utils.Progbar(steps_per_epoch)
            start = time.time()

            sess.run(train_init_op, feed_dict={batch_size: BATCH_SIZE})

            for i in range(steps_per_epoch):
                _, train_loss, train_acc = sess.run([train_op, loss, acc])

                # prog_vals = [("loss", train_loss), ("acc", train_acc)]

                train_loss_hist.append(train_loss)
                train_acc_hist.append(train_acc)

                if i == steps_per_epoch - 1:
                    sess.run(test_init_op, feed_dict={batch_size: TEST_BATCH_SIZE})
                    val_loss, val_acc = sess.run([loss, acc])

                    # prog_vals += [("val_loss", val_loss), ("val_acc", val_acc)]

                    val_loss_hist.append(val_loss)
                    val_acc_hist.append(val_acc)

                # progbar.update(i+1, values=prog_vals)

            epoch_time = time.time() - start
            mean_train_loss = np.mean(train_loss_hist[-5:])
            mean_train_acc = np.mean(train_acc_hist[-5:])

            print('Time={:.0f}, Loss={:.4f}, Acc={:.4f}, Val Loss={:.4f}, Val Acc={:.4f}'.\
                format(epoch_time, mean_train_loss, mean_train_acc, val_loss, val_acc))

        saver = tf.train.Saver()
        path = os.path.join(model_path, 'ckpt')
        saver.save(sess, path)

    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist

def train(architecture,model_id,num_classes,epochs,steps_per_epoch,BATCH_SIZE):
    model_path = './models/mnist-{}_{:d}'.format(architecture, model_id)

    print(bcolors.HEADER + 'Save model path: ' + model_path + bcolors.ENDC)

    # Load data from MNIST
    (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
        load_data(architecture, num_classes)

    tf.reset_default_graph()

    train_data = (X_train.astype('float32'), y_train_one_hot)
    test_data = (X_test.astype('float32'), y_test_one_hot)

    inputs, labels_one_hot, train_init_op, test_init_op, batch_size = \
        get_data_as_tensor(train_data, test_data, BATCH_SIZE)

    model = build_model(architecture, inputs, num_classes, model_id)
    model.summary()

    # Declare training ops
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op, loss = get_train_ops(labels_one_hot, model.output, optimizer)
    acc = get_acc_ops(labels_one_hot, model.output, num_classes)

    # Run training ops
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
        run_train_ops(epochs, steps_per_epoch, batch_size,
            train_init_op, test_init_op, train_op, loss, acc, model_path,
            BATCH_SIZE, len(X_test))

    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist

def compute_acc(pred, labels_one_hot, num_classes):
    pred_max = tf.keras.utils.to_categorical(np.argmax(pred, axis=-1), num_classes)
    return np.mean(np.sum(labels_one_hot*pred_max, axis=1))

def test(architecture, model_id_list, num_classes):
    # Load data from MNIST
    (X_train, y_train_one_hot), (X_test, y_test_one_hot) = \
        load_data(architecture, num_classes)

    test_outputs = []
    test_losses = []
    test_accs = []

    for id in model_id_list:
        graph = tf.Graph()
        sess = tf.Session(graph=graph)

        with sess.as_default(), graph.as_default():
            model_path = './models/mnist-{}_{}'.format(architecture, id)
            meta_path = os.path.join(model_path, 'ckpt.meta')
            ckpt = tf.train.get_checkpoint_state(model_path)

            imported_graph = tf.train.import_meta_graph(meta_path)
            imported_graph.restore(sess, ckpt.model_checkpoint_path)

            sess.run('test_init_op', feed_dict={'batch_size:0': len(X_test)})

            output, loss, acc = sess.run(['model_%s'%id+'/'+'output:0',
                'loss:0', 'acc:0'])

            test_outputs.append(output)
            test_losses.append(loss)
            test_accs.append(acc)

    # Average predictions before softmax
    before_mean_output = softmax(np.array(test_outputs).mean(axis=0), axis=-1)
    before_mean_acc = compute_acc(before_mean_output,y_test_one_hot,num_classes)

    # Average predictions after softmax
    after_mean_output = softmax(np.array(test_outputs), axis=-1).mean(axis=0)
    after_mean_acc = compute_acc(after_mean_output,y_test_one_hot,num_classes)

    print('Before mean acc:', before_mean_acc)
    print('After mean acc:', after_mean_acc)

    return before_mean_acc, after_mean_acc

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        print(bcolors.HEADER + 'MODE: TEST' + bcolors.ENDC)
        test(args.architecture, args.test, 10)
    else:
        print(bcolors.HEADER + 'MODE: TRAIN' + bcolors.ENDC)
        train(args.architecture, args.model_id, 10, args.epochs,
            args.steps_per_epoch, args.batch_size)

    print('Finished!')
