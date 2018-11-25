import numpy as np
import os

from keras.models import model_from_json

# from config import *

MODELS_ROOT = '../../models'

def save_weights(model, root, it):
    if os.path.exists(MODELS_ROOT) == False:
        os.system('mkdir ' + MODELS_ROOT)

    if os.path.exists(os.path.join(MODELS_ROOT, '%s' % root)) == False:
        os.system('mkdir ' + os.path.join(MODELS_ROOT, '%s' % root))

    file_name = root + '_' + str(it) + '.npy'
    np.save(os.path.join(MODELS_ROOT, root, file_name), model.get_weights())

    return (os.path.join(MODELS_ROOT, root, file_name))

def load_model(json_file, weights_file=None):
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    if weights_file is not None:
        model.set_weights(np.load(weights_file))
    return model

def set_gpu_memory_fraction(f):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = f
    sess = tf.Session(config=config)
    set_session(sess)
    return sess

def step_decay_cont(epochs, era, init_lr=0.0003, drop=0.5,
                    epochs_drop=10.0, t1=50.0):
    def func(epoch, init_epoch=epochs*(era - 1), init_lr=init_lr):
        if epoch + init_epoch < t1:
            lrate = init_lr
        else:
            lrate = init_lr * np.power(
                drop, np.floor(
                (epoch + init_epoch - t1 + epochs_drop)/(epochs_drop)))

        return lrate
    return func
