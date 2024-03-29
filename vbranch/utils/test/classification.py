from .. import restore_sess

import tensorflow as tf
import numpy as np
import os
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

def baseline_classification(sess, x, y, model_name='model', num_classes=None,
        return_logits=False, batch_size=None):
    # Convert labels to one-hot if needed
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if len(y.shape) == 1:
        y = tf.keras.utils.to_categorical(y, num_classes)

    batch_size = len(x) if batch_size is None else batch_size
    feed_dict = {'x:0': x, 'y:0':y, 'batch_size:0': batch_size}
    sess.run('test_init_op', feed_dict=feed_dict)
    logits_list = []
    for _ in range(len(x) // batch_size):
        logits_list.append(sess.run(f'{model_name}/top/output/output:0'))
    logits = np.concatenate(logits_list)

    if return_logits:
        return compute_acc_from_logits(logits, y, num_classes), logits

    results = {'acc' : compute_acc_from_logits(logits, y, num_classes)}
    return results

def vbranch_classification(sess, x, y, n_branches, model_name='model',
        num_classes=None, mode='before', return_logits=False, batch_size=None):
    results = {}

    # Convert to one-hot if needed
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if len(y.shape) == 1:
        y = tf.keras.utils.to_categorical(y, num_classes)

    outputs = []
    test_init_ops = []
    for i in range(n_branches):
        outputs.append(f'{model_name}/top/output/vb{i+1}/output:0')
        test_init_ops.append(f'test_init_op_{i+1}')

    batch_size = len(x) if batch_size is None else batch_size
    feed_dict = {'x:0': x, 'y:0':y, 'batch_size:0': batch_size}
    sess.run(test_init_ops, feed_dict=feed_dict)
    logits_list = []
    for _ in range(len(x) // batch_size):
        logits_list.append(sess.run(outputs))
    logits_list = [np.concatenate(x) for x in zip(*logits_list)]
    # print(len(logits_list), logits_list[0].shape)

    # feed_dict = {'x:0': x, 'y:0':y, 'batch_size:0': len(x)}
    # sess.run(test_init_ops, feed_dict=feed_dict)
    # logits_list = sess.run(outputs)

    vbranch_acc = compute_acc_from_logits(logits_list, y, num_classes,mode=mode)

    baseline_acc_list = []
    for logits in logits_list:
        acc = compute_acc_from_logits(logits, y, num_classes)
        baseline_acc_list.append(acc)

    results['acc_ensemble'] = vbranch_acc
    for i, acc in enumerate(baseline_acc_list):
        results[f'acc_{i+1}'] = acc

    return results

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
    print(np.mean(result_per_model, axis=0))
    return np.mean(result_per_model, axis=1)

def compute_correlation_strength(outputs, labels_one_hot, n_classes, n_models):
    """
    Args:
        - outputs: list of outputs with shape [n_models, n_samples, n_classes]
        - labels: one hot labels
    """
    preds = np.array([np.argmax(x, axis=1) for x in outputs]).transpose(1,0)
    labels = np.argmax(labels_one_hot, axis=-1)
    j_list = j_hat(preds, labels, n_classes)

    rmg_list = []
    std_list = []
    for i in range(n_models):
        x = rmg(preds[:, i], labels, j_list)
        rmg_list.append(x)
        std_list.append(np.std(x))

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
    # print(strength_list)
    strength = np.mean(strength_list)

    return mean_correlation, strength
