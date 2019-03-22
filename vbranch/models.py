# Declare models using tensorflow

from .layers import Dense, BatchNormalization, Activation

import tensorflow as tf

def simple_fcnet(input_tensor, *layers_spec):
    x = input_tensor
    for i in range(len(layers_spec)):
        x = Dense(layers_spec[i], 'fc'+str(i + 1))(x)
        x = BatchNormalization('bn'+str(i + 1))(x)
        x = Activation('relu', 'relu'+str(i + 1))(x)
        # print(x)
    return x
