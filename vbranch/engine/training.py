from .functional import Network, NetworkVB
from .. import losses as LL
from ..utils import TFSessionGrow

import tensorflow as tf
import os
from copy import copy
import numpy as np

class Model(Network):
    """The `Model` class adds training routine to `Network`"""

    def compile(self, optimizer, loss, train_init_op, test_init_op,
            callbacks={}, schedulers={}, assign_ops=[], labels=None):
        """
        Compiles model
        Args:
            - optimizer: optimizer object
            - train_init_op: training initializer
            - test_init_op: testing initializer
            - loss: str, function (see `losses.py`)
            - callbacks: dict of callable objects
            - schedulers: dict of callable objects
            - assign_ops: tf ops to initialize weights with pretrained values
        """

        self.optimizer = optimizer
        self.loss = {'loss' : loss(labels, self.output, name='loss') }
        self.train_op = optimizer.minimize(self.loss['loss'])

        self.train_init_op = train_init_op
        self.test_init_op = test_init_op
        self.callbacks = callbacks
        self.schedulers = schedulers
        self.assign_ops = assign_ops

    def fit(self, epochs, steps_per_epoch, train_dict={}, val_dict=None,
            log_path=None, call_step=1):
        """
        Train model given training and validation data
        Args:
            - epochs: number of epochs to train model, loss logged at end of
            each epoch to history
            - steps_per_epoch: number of steps per epoch
            - train_dict: dict of np arrays to initialize training iterators
            (optional, can be empty dict if using generators)
            - val_dict: optional, dict of np arrays to initialize tesing
            iterators
            - log_path: path to save model checkpoints
            - call_step: run callbacks after every `call_step` epochs
        Returns:
            - dict of losses and callbacks evaluated on training (and
            validation data if supplied)"""

        history = _fit(self.train_init_op, self.test_init_op, train_dict,
            epochs, steps_per_epoch, self.loss, self.train_op, val_dict,
            log_path, self.callbacks, self.schedulers,
            assign_ops=self.assign_ops, call_step=call_step)
        return history

class ModelVB(NetworkVB):
    def compile(self, optimizer, loss, train_init_ops, test_init_ops,
            callbacks={}, schedulers={}, assign_ops=[], labels=None):
        """
        Compiles virtual branching model
        Args:
            - optimizer: optimizer object
            - train_init_op: list of training initializers
            - test_init_op: list of testing initializers
            - loss: str, function (see `losses.py`)
            - callbacks: dict of callable objects
            - schedulers: dict of callable objects
            - assign_ops: tf ops to initialize weights with pretrained values
        """

        self.optimizer = optimizer

        if type(labels) is list:
            labels_list = labels
        else:
            labels_list = [labels] * self.n_branches

        self.losses = self._get_losses(loss, labels_list)
        self.train_ops = self._get_train_ops(optimizer)

        self.train_init_ops = train_init_ops
        self.test_init_ops = test_init_ops
        self.callbacks = callbacks
        self.schedulers = schedulers
        self.assign_ops = assign_ops

    def fit(self, epochs, steps_per_epoch, train_dict={}, val_dict=None,
            log_path=None, call_step=1):
        """
        Train virtual branching model given training and validation data
        Args:
            - epochs: number of epochs to train model, loss logged at end of
            each epoch to history
            - steps_per_epoch: number of steps per epoch
            - train_dict: dict of np arrays to initialize training iterators
            (optional, can be empty dict if using generators); note that
            different (x, y) data be need to be supplied for each branch if
            using bagging
            - val_dict: optional, dict of np arrays to initialize tesing
            iterators
            - log_path: path to save model checkpoints
            - call_step: run callbacks after every `call_step` epochs
        Returns:
            - dict of losses and callbacks evaluated on training (and
            validation data if supplied); both ensemble and per-branch metrics
            will be calculated"""

        history = _fit(self.train_init_ops, self.test_init_ops, train_dict,
            epochs, steps_per_epoch, self.losses, self.train_ops, val_dict,
            log_path, self.callbacks, self.schedulers, self.n_branches,
            self.assign_ops, call_step=call_step)
        return history

    def _get_shared_unshared_vars(self):
        """
        Returns shared variables (in order to later average gradients)
        and unshared variables (unique to each branch)"""

        shared_vars = []
        unshared_vars = [[] for i in range(self.n_branches)]

        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.name)

        for var in all_vars:
            if 'shared_to_shared' in var.name:
                shared_vars.append(var)
            else:
                for i in range(self.n_branches):
                    if 'vb'+str(i+1) in var.name:
                        unshared_vars[i].append(var)

        return shared_vars, unshared_vars

    def _get_losses(self, loss, labels):
        # Apply losses to each branch individually
        losses = {}
        for i in range(self.n_branches):
            name = 'loss_'+str(i+1)
            losses[name] = loss(labels[i], self.output[i], name=name)
        return losses

    def _get_train_ops(self, optimizer):
        self.shared_vars, self.unshared_vars = self._get_shared_unshared_vars()
        shared_grads = []
        unshared_train_ops = []

        for i in range(self.n_branches):
            loss = self.losses['loss_' + str(i+1)]

            # Compute gradients of shared vars for each branch
            # (but don't apply)
            if len(self.shared_vars) > 0:
                shared_grads.append(optimizer.compute_gradients(loss,
                    var_list=self.shared_vars))

            # Apply gradients for unshared vars for each branch
            if len(self.unshared_vars[i]) > 0:
                unshared_train_ops.append(optimizer.minimize(loss,
                    var_list=self.unshared_vars[i]))

        # Take average of the gradients over each branch
        mean_shared_grads = []

        for v, var in enumerate(self.shared_vars):
            grad = tf.reduce_mean(
                [shared_grads[i][v][0] for i in range(self.n_branches)],[0])
            mean_shared_grads.append((grad, var))

        if len(self.shared_vars) > 0:
            shared_train_op = optimizer.apply_gradients(mean_shared_grads)
        else:
            shared_train_op = []

        train_ops = [unshared_train_ops, shared_train_op]
        return train_ops

def _fit(train_init_op, test_init_op, train_dict, epochs, steps_per_epoch,
        loss_op, train_op, val_dict=None, save_model_path=None, callbacks={},
        schedulers={}, n_branches=1, assign_ops=[], call_step=1):
    history = {}

    # Classification (training accuracy calculation)
    train_dict_copy = copy(train_dict)
    if 'batch_size:0' in list(train_dict.keys()) and \
            'x:0' in list(train_dict.keys()):
        train_dict_copy['batch_size:0'] = len(train_dict['x:0'])

    with TFSessionGrow() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(assign_ops)

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            progbar = tf.keras.utils.Progbar(steps_per_epoch, verbose=2)

            sess.run(train_init_op, feed_dict=train_dict)
            sched_dict = {}
            for name, func in schedulers.items():
                sched_dict[name] = func(e + 1)

            for i in range(steps_per_epoch):
                progbar_vals = []
                loss, _ = sess.run([loss_op, train_op], feed_dict=sched_dict)

                for name, l in loss.items():
                    progbar_vals.append((name, l))

                if i == steps_per_epoch - 1:
                    # For classification, evaluate callbacks (e.g., accuracy)
                    # on training set
                    if callbacks != {} and (e + 1) % call_step == 0:
                        for _, func in callbacks.items():
                            results = func(sess, train_dict_copy, n_branches)
                            for name, r in results.items():
                                hist_append(history, name, r)
                                progbar_vals.append((name, r))

                    if val_dict is not None:
                        sess.run(test_init_op, feed_dict=val_dict)
                        val_loss = sess.run(loss_op)

                        for name, l in val_loss.items():
                            progbar_vals.append(('val_' + name, l))

                        if callbacks != {} and (e + 1) % call_step == 0:
                            for _, func in callbacks.items():
                                results = func(sess, val_dict, n_branches)
                                for name, r in results.items():
                                    hist_append(history, 'val_'+name, r)
                                    progbar_vals.append(('val_'+name, r))

                # Update progress bar
                progbar.update(i+1, values=progbar_vals)

            for name, l in loss.items():
                hist_append(history, name, l)

            if val_dict is not None:
                for name, l in val_loss.items():
                    hist_append(history, 'val_'+name, l)

        if not save_model_path is None:
            saver = tf.train.Saver()
            path = os.path.join(save_model_path, 'ckpt')
            saver.save(sess, path)

    return history

def hist_append(history, key, value):
    if key in list(history.keys()):
        history[key].append(value)
    else:
        history[key] = []
        history[key].append(value)
