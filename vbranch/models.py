# Declare models using tensorflow

from . import layers

import tensorflow as tf

def simple_fcnet(input_tensor, *units):
    x = input_tensor
    for i in range(len(units) - 1):
        x = layers.fc_layer(x, units[i], units[i + 1], 'fc'+str(i+1), True)
    return x
