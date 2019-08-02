import tensorflow as tf
import numpy as np

def load_data(one_hot=True, preprocess=False):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    if preprocess:
        X_train = X_train / 127.5 - 1
        X_test = X_test / 127.5 - 1
        
    if one_hot:
        num_classes = 10
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (X_test, y_test)
