from .functional import Network, NetworkVB
from ..losses import softmax_cross_entropy_with_logits, triplet_omniglot
from ..utils import TFSessionGrow

import tensorflow as tf
import os
from copy import copy
import numpy as np

class Model(Network):
    """The `Model` class adds training & evaluation routines to a `Network`"""

    def compile(self, optimizer, loss, train_init_op, test_init_op,
            callbacks={}, schedulers={}, assign_ops=None, **kwargs):
        """
        Args:
            - optimizer: optimizer object
            - train_init_op: single obj or list, train initializers
            - test_init_op: single obj or list, test initializers
            - loss: str, name of loss function
            - callbacks: dict of callable objects
            - schedulers: dict of callable objects
        """
        self.optimizer = optimizer

        if loss == 'softmax_cross_entropy_with_logits':
            loss = softmax_cross_entropy_with_logits(
                labels=kwargs['labels_one_hot'], logits=self.output, name='loss')
        elif loss == 'triplet_omniglot':
            loss = triplet_omniglot(self.output, A=kwargs['A'],
                P=kwargs['P'], K=kwargs['K'], name='loss')
        else:
            print('Custom loss used...')

        self.loss = {'loss' : loss }
        self.train_op = optimizer.minimize(self.loss['loss'])
        self.train_init_op = train_init_op
        self.test_init_op = test_init_op
        self.callbacks = callbacks
        self.schedulers = schedulers
        self.assign_ops = assign_ops

    def fit(self, train_dict, epochs, steps_per_epoch, val_dict, log_path=None):
        history = _fit(self.train_init_op, self.test_init_op, train_dict,
            epochs, steps_per_epoch, self.loss, self.train_op, val_dict,
            log_path, self.callbacks, self.schedulers,
            assign_ops=self.assign_ops)
        return history

class ModelVB(NetworkVB):
    def compile(self, optimizer, loss, train_init_ops, test_init_ops,
            callbacks={}, schedulers={}, assign_ops=None, **kwargs):
        self.optimizer = optimizer
        self.losses = self._get_losses(loss, **kwargs)
        self.train_ops = self._get_train_ops(optimizer)

        self.train_init_ops = train_init_ops
        self.test_init_ops = test_init_ops
        self.callbacks = callbacks
        self.schedulers = schedulers
        self.assign_ops = assign_ops

    def fit(self, train_dict, epochs, steps_per_epoch, val_dict, log_path=None):
        history = _fit(self.train_init_ops, self.test_init_ops, train_dict,
            epochs, steps_per_epoch, self.losses, self.train_ops, val_dict,
            log_path, self.callbacks, self.schedulers, self.n_branches,
            self.assign_ops)
        return history

    def _get_shared_unshared_vars(self):
        """
        Get shared variables (in order to later average gradients)
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

    def _get_losses(self, loss, **kwargs):
        losses = {}

        for i in range(self.n_branches):
            name = 'loss_'+str(i+1)

            if loss == 'softmax_cross_entropy_with_logits':
                labels = kwargs['labels_one_hot'][i]
                losses[name] = softmax_cross_entropy_with_logits(labels=labels,
                    logits=self.output[i], name=name)
            elif loss == 'triplet_omniglot':
                losses[name] = triplet_omniglot(self.output[i], A=kwargs['A'],
                    P=kwargs['P'], K=kwargs['K'], name=name)
            else:
                raise ValueError('invalid loss')

        return losses

    def _get_train_ops(self, optimizer):
        # Get variables
        self.shared_vars, self.unshared_vars = self._get_shared_unshared_vars()

        # Store gradients from shared variables over each branch
        shared_grads = []
        unshared_train_ops = []

        for i in range(self.n_branches):
            # Get loss from losses dict, key names can be predicted because
            # train ops only declared once
            loss = self.losses['loss_' + str(i+1)]

            # Compute gradients of shared vars for each branch (but don't apply)
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
            # print(v, var)
            # for i in range(self.n_branches):
            #     print(shared_grads[i][v][0])

            grad = tf.reduce_mean(
                [shared_grads[i][v][0] for i in range(self.n_branches)],[0])
            mean_shared_grads.append((grad, var))

        if len(self.shared_vars) > 0:
            shared_train_op = optimizer.apply_gradients(mean_shared_grads)
        else:
            shared_train_op = []

        train_ops = [unshared_train_ops, shared_train_op]

        return train_ops

    def _get_acc_ops(self, labels):
        # print(self.output)
        num_classes = self.output[0].get_shape().as_list()[-1]

        accs = {}

        # Train accuracies
        for i in range(self.n_branches):
            name = 'acc_'+str(i+1)
            pred_max = tf.one_hot(tf.argmax(tf.nn.softmax(self.output[i]),
                axis=-1), num_classes)
            accs[name] = tf.reduce_mean(
                tf.reduce_sum(labels[i]*pred_max,[1]), name=name)

        # Test accuracy (before average accuracy)
        pred = tf.nn.softmax(tf.reduce_mean(self.output.to_list(), [0]))
        pred_max = tf.one_hot(tf.argmax(pred, axis=-1), num_classes)
        accs['acc_ensemble'] = tf.reduce_mean(tf.reduce_sum(labels[0] * \
            pred_max, [1]), name='acc_ensemble')

        return accs

def _get_tensors(attributes, tensors):
    for attr in attributes:
        if type(attr) is list:
            _get_tensors(attr, tensors)
        else:
            if isinstance(attr, tf.Tensor):
                tensors[attr.name] = attr

def _get_operations(attributes, operations):
    for attr in attributes:
        if type(attr) is list:
            _get_operations(attr, operations)
        else:
            if isinstance(attr, tf.Operation):
                operations[attr.name] = attr

def _fit(train_init_op, test_init_op, train_dict, epochs, steps_per_epoch,
        loss_op, train_op, val_dict=None, save_model_path=None, callbacks={},
        schedulers={}, n_branches=1, assign_ops=None):
    """
    Args:
        - assign_ops: any additional ops to run before training, e.g., assign
        ops for transferring pre-trained weights (imagenet)
    """

    history = {}

    train_dict_copy = copy(train_dict)
    # Classification
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
                    if callbacks != {}:
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

                        if callbacks != {}:
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
