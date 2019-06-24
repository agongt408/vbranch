from .utils.test_utils import baseline_classification, vbranch_classification
from .utils.test_utils import baseline_one_shot, vbranch_one_shot

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
