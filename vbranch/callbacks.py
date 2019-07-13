from .utils.test.classification import baseline_classification, vbranch_classification
from .utils.test.one_shot import baseline_one_shot, vbranch_one_shot
from .utils.test.reid import get_score

def classification_acc(n_branches=1, n_classes=None):
    """Compute classification accuracy"""
    def func(sess, feed_dict):
        return baseline_classification(sess, feed_dict['x:0'], feed_dict['y:0'],
            num_classes=n_classes)

    def vb_func(sess, feed_dict):
        return vbranch_classification(sess, feed_dict['x:0'], feed_dict['y:0'],
            n_branches, num_classes=n_classes)

    if n_branches == 1:
        return func

    return vb_func

def one_shot_acc(n_branches=1):
    """Compute one shot accuracy"""
    def func(sess, feed_dict=None):
        return baseline_one_shot(sess)

    def vb_func(sess, feed_dict=None):
        return vbranch_one_shot(sess, n_branches)

    if n_branches == 1:
        return func

    return vb_func

def reid_acc(dataset, n_branches=1, **kwargs):
    """Compute re-ID rank and mAP scores"""
    def func(sess, feed_dict=None):
        return get_score(sess, dataset, n_branches=n_branches, **kwargs)
    return func
