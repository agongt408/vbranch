from . import restore_sess

import tensorflow as tf
import cv2
import numpy as np
from scipy.spatial import distance
import os
import copy
from scipy.special import softmax

# Classification

# Compute accurary given class predictions and labels
def compute_acc_from_prob(pred, labels_one_hot, num_classes=None):
    pred_max = tf.keras.utils.to_categorical(np.argmax(pred, axis=-1),
        num_classes=num_classes)
    return np.mean(np.sum(labels_one_hot*pred_max, axis=1))

# Average predictions before softmax
def compute_acc_from_logits(logits, y_test_one_hot, num_classes=None, mode=None):
    assert mode in [None, 'before', 'after']

    if mode == 'before':
        # Tend to have better performance than mode='after'
        pred = softmax(np.array(logits).mean(axis=0), axis=-1)
    elif mode == 'after':
        pred = softmax(np.array(logits), axis=-1).mean(axis=0)
    else:
        # Baseline accuracy of single model
        pred = softmax(logits, axis=-1)

    return compute_acc_from_prob(pred, y_test_one_hot, num_classes)

def baseline_classification(sess, X_test, y_test, model_name='',
        num_classes=None, return_logits=False):
    # Convert to one-hot if needed
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)
    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test)

    logits = sess.run(os.path.join(model_name, 'output/output:0'),
        feed_dict={'x_test:0':X_test})

    if return_logits:
        return compute_acc_from_logits(logits, y_test, num_classes), logits

    return compute_acc_from_logits(logits, y_test, num_classes)

def vbranch_classification(sess, X_test, y_test, model_name='',
        num_classes=None, mode='before', n_branches=1, return_logits=False):
    # Convert to one-hot if needed
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)
    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test)

    outputs = []
    for i in range(n_branches):
        name = os.path.join(model_name, 'output/vb{}/output:0'.format(i+1))
        outputs.append(name)

    logits_list = sess.run(outputs, feed_dict={'x_test:0':X_test})
    vbranch_acc = compute_acc_from_logits(logits_list, y_test,
        num_classes, mode=mode)

    baseline_acc_list = []
    for logits in logits_list:
        acc = compute_acc_from_logits(logits, y_test, num_classes)
        baseline_acc_list.append(acc)

    return vbranch_acc, baseline_acc_list

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

def baseline_one_shot(sess=None, total_runs=20, model_name='', train_runs=None,
        test_runs=None):
    def get_feed_dict(X):
        feed_dict = {'x:0': X, 'batch_size:0': len(X)}
        return feed_dict

    run_data = [get_run(r+1) for r in range(total_runs)]
    run_acc_list = []

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

        run_acc_list.append(compute_one_shot_acc(test_run, train_run,
            train_files,answers_files))

    return np.mean(run_acc_list)

def vbranch_one_shot(sess, total_runs=20, mode='concat', model_name='model_1',
        n_branches=1, baseline=True):

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

# Correlation and Strength (from Random Forest paper)

def j_hat(preds_per_model, labels, n_classes):
    """
    preds_per_model: [samples, n_models]
    Y: 1-D array
    n_classes: scalar
    """

    hits_per_class = []
    total = len(preds_per_model)

    for i in range(n_classes):
        hits = np.sum(preds_per_model == i, axis=1)
        hits_per_class.append(hits)

    hits_per_class = np.array(hits_per_class).transpose(1,0)
    # return hits_per_class

    hits_per_class[range(labels.shape[0]), labels] = -1

    j_hat_list = np.argmax(hits_per_class, axis=1)
    return j_hat_list

def rmg(preds, labels, j):
    """Raw margin function"""

    result = (preds == labels).astype('int8') - (preds == j).astype('int8')
    return result

def margin(preds_per_model, labels, j):
    hit = (preds_per_model == labels).astype('int8')
    miss = (preds_per_model == j).astype('int8')
    result_per_model = hit - miss
    return np.mean(result_per_model, axis=1)

def compute_correlation_strength(preds, labels, n_classes, n_models):
    j_list = j_hat(preds, labels, n_classes)

    rmg_list = []
    for i in range(n_models):
        rmg_list.append(rmg(preds[:, i], labels, j_list))

    std_list = [np.std(x) for x in rmg_list]

    num_list = []
    dem_list = []

    for i in range(n_models):
        for j in range(i+1, n_models):
            rho = np.corrcoef(rmg_list[0], rmg_list[1])[0,1]
            num_list.append(rho * std_list[i] * std_list[j])
            dem_list.append(std_list[i] * std_list[j])

    mean_correlation = np.mean(num_list) / np.mean(dem_list)

    strength_list = margin(preds, np.tile(labels[:, np.newaxis], [1,n_models]),
                           np.tile(j_list[:, np.newaxis], [1,n_models]))
    strength = np.mean(strength_list)

    return mean_correlation, strength
