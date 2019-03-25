import tensorflow as tf

def triplet(pred, P, K, margin=0.2, name=None):
    """
    Define triplet loss (batch-hard variant)
    (https://arxiv.org/pdf/1703.07737.pdf)
    Args:
        P: number of identities
        K: number of samples per identity
        margin: margin for hinge loss
    Returns:
        loss"""

    assert margin == 'soft' or margin >= 0, 'invalid margin={}'.format(margin)

    loss = tf.Variable(0, dtype='float32', name=name)

    for i in range(P):
        for a in range(K):
            pred_anchor = pred[i * K + a]

            pos = norm(pred_anchor, pred[i*K:(i + 1)*K])
            # print(pos.get_shape().as_list())
            hard_pos = tf.reduce_max(pos)

            neg = norm(pred_anchor, tf.concat([pred[0:i*K], pred[(i + 1)*K:]], 0))
            # print(neg.get_shape().as_list())
            hard_neg = tf.reduce_min(neg)

            if margin == 'soft':
                loss = loss + log1p(hard_pos - hard_neg)
            else:
                loss = loss + tf.maximum(margin + hard_pos - hard_neg, 0.0)

    return loss

def triplet_omniglot(pred, A, P, K, name, margin=0.2):
    """
    Args:
        A: number of alphabets
        P: number of characters per alphabet
        K: number of samples per character
        margin: margin for hinge loss"""

    alpha_losses = []
    for i in range(A):
        alpha_pred = pred[P*K*i : P*K*(i+1)]
        alpha_losses.append(triplet(alpha_pred, P, K, margin))

    loss = tf.reduce_mean(alpha_losses, name=name)
    return loss

def log1p(x):
    """For soft margin triplet loss"""
    return tf.log(1 + tf.exp(x))

def norm(x1, x2, axis=1, norm=1):
    return tf.pow(tf.reduce_sum(tf.pow(tf.abs(x1 - x2), norm), axis=axis), 1.0 / norm)
