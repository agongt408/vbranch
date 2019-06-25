from .utils.test.classification import baseline_classification, vbranch_classification
from .utils.test.one_shot import baseline_one_shot, vbranch_one_shot
from .utils.test.reid import get_score

def classification_acc(n_branches):
    def func(sess, feed_dict, n_branches):
        return baseline_classification(sess, feed_dict['x:0'], feed_dict['y:0'])

    def vb_func(sess, feed_dict, n_branches):
        return vbranch_classification(sess, feed_dict['x:0'], feed_dict['y:0'],
            n_branches)

    if n_branches == 1:
        return func

    return vb_func

def one_shot_acc(n_branches):
    def func(sess, feed_dict, n_branches):
        return baseline_one_shot(sess)

    def vb_func(sess, feed_dict, n_branches):
        return vbranch_one_shot(sess, n_branches)

    if n_branches == 1:
        return func

    return vb_func

def reid_acc(dataset, n_branches, img_dim=(256, 128, 3)):
    def func(sess, feed_dict, n_branches):
        return get_score(sess, dataset, img_dim=img_dim)

    return func
