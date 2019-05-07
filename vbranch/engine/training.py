from .functional import Network, NetworkVB
from ..losses import softmax_cross_entropy_with_logits, triplet_omniglot

import tensorflow as tf
import os

class Model(Network):
    """The `Model` class adds training & evaluation routines to a `Network`.
    """

    def compile(self, optimizer, loss, test=False, **kwargs):
        self.optimizer = optimizer

        # Set loss
        if loss == 'softmax_cross_entropy_with_logits':
            labels_one_hot = kwargs['labels_one_hot']

            self.loss = softmax_cross_entropy_with_logits(
                labels=labels_one_hot, logits=self.output, name='loss')

            # Set prediction
            self.pred = tf.nn.softmax(self.output, name='pred')

            # Set accuracy
            num_classes = self.pred.get_shape().as_list()[-1]
            pred_max = tf.one_hot(tf.argmax(self.pred,axis=-1),num_classes)
            self.acc = tf.reduce_mean(
                tf.reduce_sum(labels_one_hot*pred_max,[1]), name='acc')

        elif loss == 'triplet_omniglot':
            self.loss = triplet_omniglot(self.output, A=kwargs['A'],
                P=kwargs['P'], K=kwargs['K'], name='loss')
        else:
            raise ValueError('invalid loss')

        if not test:
            # Set training op
            self.train_op = optimizer.minimize(self.loss)

    def fit(self, iterator, x, y, epochs, steps_per_epoch, batch_size,
            validation=None, test_model=None, save_model_path=None):

        tensors = {}
        validation_tensors = {}
        history = {}

        for t in ['loss', 'acc']:
            if hasattr(self, t):
                tensors['train_' + t] = getattr(self, t)
                history['train_' + t] = []

            if not validation is None and hasattr(test_model, t):
                validation_tensors['val_' + t] = getattr(test_model, t)
                history['val_' + t] = []

        history = _fit(history, [iterator], x, y, epochs, steps_per_epoch,
            batch_size, tensors, validation_tensors, self.train_op, validation,
            save_model_path)

        return history

class ModelVB(NetworkVB):
    def compile(self, optimizer, loss, test=False, **kwargs):
        self.optimizer = optimizer

        # Get losses and training operations
        self.losses = self._get_losses(loss, **kwargs)

        if loss == 'softmax_cross_entropy_with_logits':
            # Get accurary operations
            labels_one_hot = kwargs['labels_one_hot']
            self.accs = self._get_acc_ops(labels_one_hot)

        if not test:
            self.train_ops = self._get_train_ops(optimizer)

    def fit(self, iterators, x, y, epochs, steps_per_epoch, batch_size,
            validation=None, test_model=None, save_model_path=None):

        tensors = {}
        validation_tensors = {}
        history = {}

        for t in ['losses', 'accs']:
            if hasattr(self, t):
                attr = getattr(self, t)
                for k in attr.keys():
                    tensors['train_' + k] = attr[k]
                    history['train_' + k] = []

            if not validation is None and hasattr(test_model, t):
                attr = getattr(test_model, t)
                for k in attr.keys():
                    validation_tensors['val_' + k] = attr[k]
                    history['val_' + k] = []

        history = _fit(history, iterators, x, y, epochs, steps_per_epoch,
            batch_size, tensors, validation_tensors, self.train_ops,validation,
            save_model_path)

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
            if loss == 'softmax_cross_entropy_with_logits':
                name = 'loss_'+str(i+1)
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
            loss = list(self.losses.items())[i][1]

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
        num_classes = self.output[0].get_shape().as_list()[-1]

        accs = {}

        # Train accuracies
        for i in range(self.n_branches):
            name = 'acc_'+str(i+1)
            pred_max = tf.one_hot(tf.argmax(tf.nn.softmax(self.output[i]),
                axis=-1), num_classes)
            accs[name] = tf.reduce_mean(
                tf.reduce_sum(labels[i]*pred_max,[1]), name=name)

        # Test accuracy
        pred = tf.nn.softmax(tf.reduce_mean(self.output.to_list(), [0]))
        pred_max = tf.one_hot(tf.argmax(pred, axis=-1), num_classes)
        accs['acc'] = tf.reduce_mean(tf.reduce_sum(labels[0]*pred_max, [1]),
                                     name='acc')
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

def _fit(history, iterators, x, y, epochs, steps_per_epoch, batch_size, tensors,
        validation_tensors, ops, validation=None, save_model_path=None):

    default_graph = tf.get_default_graph()

    # Get tensors for training
    x_train = default_graph.get_tensor_by_name('x:0')
    y_train = default_graph.get_tensor_by_name('y:0')
    b_s = default_graph.get_tensor_by_name('batch_size:0')

    # Get tensors for validation
    x_test = default_graph.get_tensor_by_name('x_test:0')
    y_test = default_graph.get_tensor_by_name('y_test:0')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            progbar = tf.keras.utils.Progbar(steps_per_epoch, verbose=2)

            # Training
            sess.run([it.initializer for it in iterators],
                feed_dict={x_train:x, y_train:y, b_s: batch_size})

            for i in range(steps_per_epoch):
                progbar_vals = []
                train_tensors_v, _ = sess.run([tensors, ops])
                for t in tensors.keys():
                    progbar_vals.append((t, train_tensors_v[t]))

                if validation_tensors != {} and i == steps_per_epoch - 1:
                    val_tensors_v = sess.run(validation_tensors,
                        feed_dict={x_test:validation[0],
                                   y_test:validation[1]})

                    for t in validation_tensors.keys():
                        progbar_vals.append((t, val_tensors_v[t]))

                # Update progress bar
                progbar.update(i+1, values=progbar_vals)

            # Add training to history
            for t in train_tensors_v.keys():
                history[t].append(train_tensors_v[t])

            # Add validation to history
            if not validation is None:
                for t in validation_tensors.keys():
                    history[t].append(val_tensors_v[t])

        if not save_model_path is None:
            saver = tf.train.Saver()
            path = os.path.join(save_model_path, 'ckpt')
            saver.save(sess, path)

    return history