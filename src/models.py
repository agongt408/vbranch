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

def TriNet(P_param=1, K_param=1, weights='imagenet', img_dim=(128,64,3), margin=0.2):
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
    model.compile(loss=triplet(P_param, K_param, margin=margin),
                  optimizer=Adam(lr=0.0003))

    return model

def MetricEnsemble(model, weights_path_arr, img_dim=None):
    if img_dim is None:
        img_dim = model.input_shape[1:]

    ip_list = []
    concat_list = []
    config = model.get_config()
    for i, path in enumerate(weights_path_arr):
        branch = Model.from_config(config)
        branch.name = "model_{}".format(i+1)
        branch.set_weights(np.load(path))
        print("Model {} successfully loaded from {}".format(i+1, path))

        ip_list.append(Input(shape=img_dim))
        concat_list.append(branch(ip_list[i]))

    ensemble = Model(ip_list, concatenate(concat_list))
    return ensemble

def MetricEnsembleFlipConcat(model, weights_path_arr, img_dim=None):
    duplicate_weights_path_arr = []
    for p in weights_path_arr:
        duplicate_weights_path_arr.append(p)
        duplicate_weights_path_arr.append(p)

    ensemble = MetricEnsemble(model, duplicate_weights_path_arr, img_dim)
    return ensemble
