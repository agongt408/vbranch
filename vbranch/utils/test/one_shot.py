from .. import restore_sess

import tensorflow as tf
import cv2
import numpy as np
from scipy.spatial import distance
import os
import copy

# One-shot

def get_run(n_run):
    all_runs = 'omniglot/python/one-shot-classification/all_runs'

    if not os.path.isdir(all_runs):
        with zipfile.ZipFile(all_runs + '.zip','r') as zip_ref:
            zip_ref.extractall(all_runs)

    run_path = os.path.join(all_runs,'run%02d'%n_run,'class_labels.txt')
    with open(run_path) as f:
        content = f.read().splitlines()

    pairs = [line.split() for line in content]
    test_files  = [pair[0] for pair in pairs]
    train_files = [pair[1] for pair in pairs]

    answers_files = copy.copy(train_files)
    test_files.sort()
    train_files.sort()

    def f_load(f):
        path = os.path.join(all_runs, f)
        return cv2.imread(path)[..., 0]

    train_imgs = np.stack([f_load(f) for f in train_files]).\
                        astype('float32')[..., np.newaxis]
    test_imgs  = np.stack([f_load(f) for f in test_files]).\
                        astype('float32')[..., np.newaxis]

    return train_files, test_files, train_imgs, test_imgs, answers_files

def compute_one_shot_acc(test_pred, train_pred, train_files, answers_files):
    n_test = len(test_pred)
    n_train = len(train_pred)

    distM = np.zeros((n_test, n_train))
    for i in range(n_test):
        for c in range(n_train):
            distM[i,c] = distance.euclidean(test_pred[i],train_pred[c])

    YHAT = np.argmin(distM, axis=1)

    # compute the error rate
    correct = 0.0
    for i in range(n_test):
        if train_files[YHAT[i]] == answers_files[i]:
            correct += 1.0

    return correct / n_test

def baseline_one_shot(sess=None, total_runs=20, model_name='model',
        train_runs=None, test_runs=None, return_outputs=False):

    def get_feed_dict(X):
        feed_dict = {'x:0': X, 'batch_size:0': len(X)}
        return feed_dict

    run_data = [get_run(r+1) for r in range(total_runs)]
    run_acc_list = []
    train_outputs = []
    test_outputs = []

    for r in range(total_runs):
        train_files,test_files,train_ims,test_ims,answers_files = run_data[r]

        if sess is not None:
            output = os.path.join(model_name, 'output/output:0')
            sess.run('test_init_op', feed_dict=get_feed_dict(train_ims))
            train_run = sess.run(output)
            sess.run('test_init_op', feed_dict=get_feed_dict(test_ims))
            test_run = sess.run(output)
        else:
            train_run = train_runs[r]
            test_run = test_runs[r]

        train_outputs.append(train_run)
        test_outputs.append(test_run)
        run_acc_list.append(compute_one_shot_acc(test_run, train_run,
            train_files,answers_files))

    if return_outputs:
        return np.mean(run_acc_list), train_outputs, test_outputs

    return {'acc' : np.mean(run_acc_list)}

def vbranch_one_shot(sess, n_branches, total_runs=20, mode='concat',
        model_name='model', baseline=True):

    assert mode in ['average', 'concat']

    def get_feed_dict(X):
        feed_dict = {'x:0': X, 'batch_size:0': len(X)}
        return feed_dict

    run_data = [get_run(r+1) for r in range(total_runs)]
    run_acc_list = []
    # Store outputs for baseline acc computation
    train_run_list = []
    test_run_list = []

    # Get graph tensors
    test_init_op = ['test_init_op_'+str(i+1) for i in range(n_branches)]
    outputs = []
    for i in range(n_branches):
        name = os.path.join(model_name, 'output/vb{}/output:0'.format(i+1))
        outputs.append(name)

    for r in range(total_runs):
        train_files,test_files,train_ims,test_ims,answers_files = run_data[r]

        sess.run(test_init_op, feed_dict=get_feed_dict(train_ims))
        train_run = sess.run(outputs)
        sess.run(test_init_op, feed_dict=get_feed_dict(test_ims))
        test_run = sess.run(outputs)

        if mode == 'average':
            test_embed = np.mean(test_run, axis=0)
            train_embed = np.mean(train_run, axis=0)
        else:
            test_embed = np.concatenate(test_run, axis=-1)
            train_embed = np.concatenate(train_run, axis=-1)

            run_acc_list.append(compute_one_shot_acc(test_embed, train_embed,
                train_files, answers_files))

        if baseline:
            train_run_list.append(train_run)
            test_run_list.append(test_run)

    # Baseline
    baseline_acc_list = []
    train_run_list = np.stack(train_run_list)
    test_run_list = np.stack(test_run_list)

    for i in range(n_branches):
        acc = baseline_one_shot(total_runs=total_runs,
            train_runs=train_run_list[:, i], test_runs=test_run_list[:, i])
        baseline_acc_list.append(acc)

    return np.mean(run_acc_list), baseline_acc_list
