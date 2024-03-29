import tensorflow as tf
import numpy as np

def glorot_uniform(shape, fan_in, fan_out):
    """Return initialized tensor
    Args:
        - shape: tuple or list, desired shape of tensor
        - fan_in: number of input units
        - fan_out: number of output units
    https://github.com/tensorflow/tensorflow/blob/e19c354920c3b246dda6598229210a582caaa1a9/tensorflow/python/ops/init_ops.py#L1423
    https://github.com/tensorflow/tensorflow/blob/e19c354920c3b246dda6598229210a582caaa1a9/tensorflow/python/ops/init_ops.py#L451
    """
    limit = np.sqrt(6 / (fan_in + fan_out))
    return tf.random.uniform(shape, minval=-limit, maxval=limit)

def rectifier_init(shape, fan_in):
    """
    Weight initialization used in DenseNet paper:
    https://arxiv.org/pdf/1502.01852.pdf
    """
    std = np.sqrt(2 / fan_in)
    # print('rectifier_init', std)
    return tf.random.normal(shape, mean=0.0, stddev=std)
