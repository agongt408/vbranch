import tensorflow as tf
import cv2
import numpy as np
from scipy.spatial import distance
import os
import copy
from scipy.special import softmax

# Classification metrics

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
    mean_output = softmax(np.array(outputs), axis=-1).mean(axis=0)
    return compute_acc(mean_output, y_test_one_hot, num_classes)

# One-shot metrics

def restore_sess(sess, model_path):
    meta_path = os.path.join(model_path, 'ckpt.meta')
    ckpt = tf.train.get_checkpoint_state(model_path)

    imported_graph = tf.train.import_meta_graph(meta_path)
    imported_graph.restore(sess, ckpt.model_checkpoint_path)

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
