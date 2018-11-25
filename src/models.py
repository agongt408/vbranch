from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
# from keras.regularizers import l2
import keras.backend as K

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, '../DenseNet/')
import densenet

# sys.path.insert(0, '..')
# import losses

from src import triplet

def TriNet(P_param=1, K_param=1, weights='imagenet', img_dim=(128,64,3)):
    '''
    Instantiates TriNet model (https://arxiv.org/pdf/1703.07737.pdf)
    # Arguments:
        P_param: P (see paper)
        K_param: K (see paper)
        weights: None, 'imagenet'
        shape: input shape
        output_dim: number of units in final embedding (FC) layer
    # Returns:
        TriNet model compiled with triplet loss function
    # Raises:
        ValueError if base is not supported
    '''

    ip = Input(shape=img_dim)
    base = densenet.DenseNetImageNet121(input_tensor=ip, weights=weights)

    x = base.layers[-2].output
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128, name='embedding')(x)

    model = Model(ip, x)
    model.compile(loss=triplet(P_param, K_param), optimizer=Adam(lr=0.0003))

    return model
