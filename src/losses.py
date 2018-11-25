import keras.backend as K

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

        print('margin' , margin)

        for i in range(P_param):
            for a in range(K_param):
                pred_anchor = y_pred[i * K_param + a]
                hard_pos = K.max(norm(pred_anchor, y_pred[i*K_param:(i + 1)*K_param]))
                hard_neg = K.min(norm(pred_anchor, K.concatenate([y_pred[0:i*K_param],
                    y_pred[(i + 1)*K_param:]], 0)))
                if margin == 'soft':
                    loss = loss + log1p(hard_pos - hard_neg)
                else:
                    loss = loss + K.maximum(margin + hard_pos - hard_neg, 0.0)
        return loss

    return triplet_loss
