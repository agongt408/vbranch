import tensorflow as tf
import numpy as np

def glorot_uniform(shape, fan_in, fan_out):
    """Return initialized tensor
    Args:
        - shape: tuple or list, desired shape of tensor
        - fan_in: number of input units
        - fan_out: number of output units
    """
    limit = np.sqrt(6 / (fan_in + fan_out))
    return tf.random.uniform(shape, minval=-limit, maxval=limit)
