import tensorflow as tf

def softmax_cross_entropy_with_logits():
    """Cross entropy loss"""
    return _softmax_cross_entropy_with_logits

def triplet(P, K, margin=0.2):
    """
    Triplet loss (batch-hard variant)
    (https://arxiv.org/pdf/1703.07737.pdf)
    Args:
        - P: number of identities
        - K: number of samples per identity
        - margin: margin for hinge loss
    Returns:
        - callable function"""

    def func(y, pred, name='loss'):
        return _triplet(pred, P, K, margin, name)
    return func

def triplet_omniglot(A, P, K, margin=0.2):
    """
    Returns triplet loss for omniglot dataset
    Args:
        - A: number of alphabets
        - P: number of characters per alphabet
        - K: number of samples per character
        - margin: margin for hinge loss
    Return:
        - callable function"""

    def func(y, pred, name):
        return _triplet_omniglot(pred, A, P, K, margin, name)
    return func

def _softmax_cross_entropy_with_logits(labels, logits, name=None):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                logits=logits)
    loss = tf.reduce_mean(cross_entropy, name=name)
    return loss

def _triplet(pred, P, K, margin=0.2, name=None):
    assert margin == 'soft' or margin >= 0

    batch_losses = []
    for i in range(P):
        for a in range(K):
            pred_anchor = pred[i * K + a]
            pos = norm(pred_anchor, pred[i*K:(i + 1)*K])
            hard_pos = tf.reduce_max(pos)
            neg_samples = tf.concat([pred[0:i*K], pred[(i + 1)*K:]], 0)
            neg = norm(pred_anchor, neg_samples)
            hard_neg = tf.reduce_min(neg)

            if margin == 'soft':
                # Apply piecewise in order to avoid NaN warnings
                # Loss value error approx. 1e-9 using threshold value of 20
                spread = hard_pos - hard_neg
                condition = tf.less(spread, 20)
                loss = tf.cond(condition, lambda: log1p(spread), lambda: spread)
                # print(loss)
            else:
                loss = tf.maximum(margin + hard_pos - hard_neg, 0.0)
            batch_losses.append(loss)

    return tf.reduce_sum(batch_losses, name=name)

def _triplet_omniglot(pred, A, P, K, margin=0.2, name=None):
    alpha_losses = []
    for i in range(A):
        alpha_pred = pred[P*K*i : P*K*(i+1)]
        alpha_losses.append(_triplet(alpha_pred, P, K, margin))

    return tf.reduce_sum(alpha_losses, name=name)

def log1p(x):
    """Soft margin function"""
    return tf.log(1 + tf.exp(x))

def norm(x1, x2, axis=1, norm=1):
    return tf.pow(tf.reduce_sum(tf.pow(tf.abs(x1 - x2), norm),
        axis=axis), 1.0 / norm)
