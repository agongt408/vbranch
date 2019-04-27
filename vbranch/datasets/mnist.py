import tensorflow as tf
import numpy as np

def load_data(format):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    if format == 'fcn':
        X_train = X_train.reshape([-1, 784])
        X_test = X_test.reshape([-1, 784])
    elif format == 'cnn':
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
    else:
        raise ValueError('invalid format')

    num_classes = 10
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train_one_hot), (X_test, y_test_one_hot)
