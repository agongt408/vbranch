import tensorflow as tf
import numpy as np

def load_data(format, one_hot=True, preprocess=False):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    if format == 'fcn':
        X_train = X_train.reshape([-1, 784])
        X_test = X_test.reshape([-1, 784])
    elif format == 'cnn':
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
    else:
        raise ValueError('invalid format')

    if preprocess:
        X_train = X_train / 127.5 - 1
        X_test = X_test / 127.5 - 1

    if one_hot:
        num_classes = 10
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (X_test, y_test)
