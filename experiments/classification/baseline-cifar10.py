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
from glob import glob

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
# parser.add_argument('--steps_per_epoch', action='store', default=100, nargs='?',
#                     type=int, help='number of training steps per epoch')
parser.add_argument('--test', action='store_true', help='testing mode')
parser.add_argument('--trials', action='store', default=1, nargs='?', type=int,
                    help='number of trials to perform, if 1 then model_id used')
# parser.add_argument('--train_frac', action='store', default=1., type=float,
#                     help='fraction of original dataset to use for training')

parser.add_argument('--path', action='store', nargs='?', default=None,
                    help='manually specify path to save model checkpoint and results')

def build_model(depth, growth_rate, x_shape, y_shape, batch_size):
    inputs, labels, train_init_op, test_init_op = get_data_iterator(x_shape,
        y_shape, batch_size, n=1, share_xy=True)
    lr = tf.placeholder('float32', name='lr')
    lr_scheduler = lr_step_scheduler((100, 0.001), (150, 0.0001), (200, 0.00001))

    name = 'model'
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        model = DenseNet(depth, growth_rate, inputs, name=name)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        model.compile(optimizer, softmax_cross_entropy_with_logits(),
            train_init_op, test_init_op, labels=labels,
            callbacks={'acc':classification_acc(n_classes=10, batch_size=250)},
            schedulers={'lr:0': lr_scheduler})

    return model

def train(depth, growth_rate, model_id, epochs, batch_size, path):
    if path is None:
        arch_name = f'densenet-{depth}-{growth_rate}'
        model_path = get_model_path('cifar10', arch_name, model_id=model_id)
        dirpath = get_dir_path('cifar10', arch_name)
    else:
        model_path = os.path.join('models', path, 'model_{}'.format(model_id))
        if not os.path.isdir(model_path):
            os.system('mkdir -p ' + model_path)
        dirpath = path

    p_console('Save model path: ' + model_path)

    # Load data
    (X_train, y_train), (X_test, y_test) = load_data(preprocess=True, one_hot=True)

    print('93>', X_train.shape, y_train.shape)
    print('94>', X_test.shape, y_test.shape)

    x_shape = (None,) + X_train.shape[1:]
    y_shape = (None, 10)

    tf.reset_default_graph()
    model = build_model(depth, growth_rate, x_shape, y_shape, batch_size)
    model.summary()

    train_dict = {'x:0': X_train, 'y:0': y_train, 'batch_size:0': batch_size}
    val_dict = {'x:0': X_test, 'y:0': y_test, 'batch_size:0': 250}
    history = model.fit(epochs, len(X_train) // batch_size, 
        train_dict=train_dict, val_dict=val_dict, log_path=model_path, verbose=1)

    save_results(history, dirpath, f'train_{model_id}.csv', mode='w')

def test(model_id_list, X_test, y_test, path, output_dict={}, acc_dict={}):
    print(model_id_list)
    test_outputs = []
    test_accs = []

    for model_id in model_id_list:
        if model_id in output_dict.keys():
            output = output_dict[model_id]
            acc = acc_dict[model_id]
        else:
            tf.reset_default_graph()
            with TFSessionGrow() as sess:
                model_path = os.path.join('models',path,f'model_{model_id}')
                restore_sess(sess, model_path)

                # Compute accuracy
                acc, output = baseline_classification(sess, X_test, y_test,
                    num_classes=n_classes, return_logits=True)

                output_dict[model_id] = output
                acc_dict[model_id] = acc

        test_outputs.append(output)
        test_accs.append(acc)

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

    print('158>', path)
    save_results(results_dict, path, f'B{len(model_id_list)}-test.csv', 'a')

    return output_dict, acc_dict

if __name__ == '__main__':
    args = parser.parse_args()
    print('157> batch size', args.batch_size)

    if args.test:
        p_console('MODE: TEST')

        _, (X_test, y_test) = load_data(preprocess=True, one_hot=True)

        if args.trials == 1:
            # args.model_id is a list of model ids
            test(args.depth, args.growth_rate, args.model_id, X_test, y_test)
        else:
            # Store output, acc, and dict in case need to be reused
            output_dict = {}
            acc_dict = {}

            if args.path is None:
                arch_name = f'densenet-{args.depth}-{args.growth_rate}'
                path = get_dir_path('cifar10', arch_name)
            else:
                path = args.path

            avail_runs = glob(os.path.join('models', path, 'model_*'))
            avail_ids = [int(p[-p[::-1].index('_'):]) for p in avail_runs]

            for i in range(args.trials):
                model_ids = np.random.choice(avail_ids, len(args.model_id),
                    replace=False)
                test(model_ids, X_test, y_test, path, output_dict, acc_dict)
    else:
        p_console('MODE: TRAIN')

        if args.trials == 1:
            for model_id in args.model_id:
                # Run trial with specified model id
                train(args.depth, args.growth_rate, model_id, 
                    args.epochs, args.batch_size, args.path)
        else:
            # Run n trials with model id from 1 to args.trials
            for i in range(args.trials):
                train(args.depth, args.growth_rate,
                    i+1, args.epochs, args.batch_size, args.path)

    print('Finished!')
