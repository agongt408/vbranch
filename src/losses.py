import keras.backend as K
import tensorflow as tf
import numpy as np

from keras.losses import categorical_crossentropy

P_param = 4
K_param = 4
input_output_dim = 128
input_margin = 0.5

def log1p(x):
    '''For soft margin triplet loss'''
    return K.log(1 + K.exp(x))

def norm(x1, x2, axis=1, norm=1):
    '''Normalized distance measure'''
    return K.pow(K.sum(K.pow(K.abs(x1 - x2), norm), axis=axis), 1.0 / norm)


def triplet(P_param=4, K_param=4, output_dim=128, margin=0.2):
    '''
    Define triplet loss (batch-hard variant)
    (https://arxiv.org/pdf/1703.07737.pdf)

    Args:
        P_param: number of identities
        K_param: number of samples per identity
        output_dim: number of units in final embedding layer
        margin: margin for hinge loss

    Returns:
        loss
    '''

    def triplet_loss(y_true, y_pred):
        loss = K.variable(0, dtype='float32')

        print 'margin' , margin

        n_slice = y_pred.get_shape()[-1].value / output_dim

        for i in range(n_slice):
            print 'output dim' , output_dim
            embeddings = y_pred[:, i*output_dim:(i+1)*output_dim]

            for i in range(P_param):
                for a in range(K_param):
                    pred_anchor = embeddings[i*K_param + a]
                    hard_pos = K.max(norm(pred_anchor, embeddings[i*K_param:(i + 1)*K_param]))
                    hard_neg = K.min(norm(pred_anchor, K.concatenate([embeddings[0:i*K_param],
                                                                    embeddings[(i + 1)*K_param:]], 0)))
                    if margin == 'soft':
                        loss += log1p(hard_pos - hard_neg)
                    else:
                        loss += K.maximum(margin + hard_pos - hard_neg, 0.0)
        return loss

    return triplet_loss

def triplet_drop(P_param=4, K_param=4, masks=[], margin=0.2):
    '''
    Define triplet loss (batch-hard variant)
    (https://arxiv.org/pdf/1703.07737.pdf)

    Args:
        P_param: number of identities
        K_param: number of samples per identity
        output_dim: number of units in final embedding layer
        margin: margin for hinge loss

    Returns:
        loss
    '''

    def triplet_loss(y_true, y_pred):
        loss = K.variable(0, dtype='float32')

        print margin

        size = [np.where(m > 0)[0].shape[0] for m in masks]
        position = [0]
        for s in range(len(size)):
            position.append(position[s] + size[s])

        print position

        for b in range(len(masks)):
            embeddings = tf.slice(y_pred, [0,position[b]], [P_param*K_param, position[b+1] - position[b]])
            print embeddings.shape

            for i in range(P_param):
                for a in range(K_param):
                    pred_anchor = embeddings[i*K_param + a]
                    hard_pos = K.max(norm(pred_anchor, embeddings[i*K_param:(i + 1)*K_param]))
                    hard_neg = K.min(norm(pred_anchor, K.concatenate([embeddings[0:i*K_param],
                                                                    embeddings[(i + 1)*K_param:]], 0)))
                    if margin == 'soft':
                        loss += log1p(hard_pos - hard_neg)
                    else:
                        loss += K.maximum(margin + hard_pos - hard_neg, 0.0)
        return loss

    return triplet_loss


def cam_loss(y_true, y_pred):
    # return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # return K.sum(norm(y_true, y_pred, axis=(1,2), norm=1))
    # return tf.losses.hinge_loss(labels=y_true, logits=y_pred)
    return K.sum(K.flatten(tf.multiply(y_true, y_pred)))
    # return y_pred
