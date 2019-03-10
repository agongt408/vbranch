import numpy as np
import os

import tensorflow as tf
from keras.models import model_from_json

# from config import *

MODELS_ROOT = '../../models'

def save_weights(model, out_dir, it):
    if os.path.exists(MODELS_ROOT) == False:
        os.system('mkdir ' + MODELS_ROOT)

    if not os.path.exists(out_dir):
        os.system('mkdir ' + out_dir)

    file_name = str(it) + '.npy'
    path = os.path.join(out_dir, file_name)
    np.save(path, model.get_weights())
    return path

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

# def step_decay_cont(epochs, era, init_lr=0.0003, drop=0.5,
#                     epochs_drop=10.0, t1=50.0):
#     def func(epoch, init_epoch=epochs*(era - 1), init_lr=init_lr):
#         if epoch + init_epoch < t1:
#             lrate = init_lr
#         else:
#             lrate = init_lr * np.power(
#                 drop, np.floor(
#                 (epoch + init_epoch - t1 + epochs_drop)/(epochs_drop)))
#
#         return lrate
#     return func

def step_decay(epoch, init_lr=0.0003, drop=0.5, epochs_drop=10.0, t1=50.0):
    def func(e):
        if epoch < t1:
            lrate = init_lr
        else:
            lrate = init_lr * np.power(drop, np.floor((epoch - t1)/epochs_drop + 1))
        return lrate
    return func

def step_decay2(epoch, init_lr=0.0003, t0=100.0, t1=150.0):
    def func(e):
        if epoch < t0:
            lrate = init_lr
        else:
            lrate = init_lr * np.power(0.001, (epoch - t0) / (t1 - t0))
            # print((epoch - t0) / (t1 - t0))
        return lrate
    return func

class TFSummary():
    def __init__(self, out_dir, val_dict=None):
        self.sw = tf.summary.FileWriter(out_dir)
        self.val_dict = val_dict

    # write tensorboard summaries
    def update(self, step, val_name=None):
        s = tf.Summary()

        if val_name is None:
            val_name = self.val_dict.keys()
        else:
            val_name = _to_list(val_name)

        for name in val_name:
            # vals = self.val_dict[name]
            v = s.value.add()
            v.simple_value = self.val_dict[name][-1]
            v.tag = name

        self.sw.add_summary(s, step)
        self.sw.flush()

    def update_term(self, step, val, name):
        s = tf.Summary()

        v = s.value.add()
        v.simple_value = val
        v.tag = name

        self.sw.add_summary(s, step)
        self.sw.flush()

def _to_list(x):
    if type(x) == list:
        return x
    return [x]
