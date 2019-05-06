from sklearn.datasets import make_classification, make_blobs
import numpy as np
import tensorflow as tf

def generate_from_hypercube(num_samples, num_features, num_classes, seed=100):
    """
    Generate clusters on vertices of hypercube
    Args:
        - num_classes: number of classes
        - std: standard deviation along each dimension
        - seed: np random seed for reproducibility
    """

    X, y = make_classification(n_samples=2*num_samples, n_features=num_features,
        n_informative=num_classes, n_redundant=0, n_classes=num_classes,
        n_clusters_per_class=1, class_sep=4, random_state=seed, shuffle=True)

    X_train = X[:num_samples]
    y_train = tf.keras.utils.to_categorical(y[:num_samples], num_classes)
    X_test = X[num_samples:]
    y_test = tf.keras.utils.to_categorical(y[num_samples:], num_classes)

    return (X_train, y_train), (X_test, y_test)

def generate_from_blobs(num_samples, num_features, num_classes, seed=100):
    """
    Generate blobs around clusters randomly sampled from a box (with added
    random noise)"""

    X, y = make_blobs(n_samples=2*num_samples, n_features=2,
        clusters=num_classes, random_state=seed, shuffle=True)

    X_train = X[:num_samples]
    y_train = tf.keras.utils.to_categorical(y[:num_samples], num_classes)
    X_test = X[num_samples:]
    y_test = tf.keras.utils.to_categorical(y[num_samples:], num_classes)

    return (X_train, y_train), (X_test, y_test)
