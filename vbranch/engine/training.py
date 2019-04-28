from .functional import Network, NetworkVB
from ..losses import softmax_cross_entropy_with_logits, triplet_omniglot

import tensorflow as tf

class Model(Network):
    """The `Model` class adds training & evaluation routines to a `Network`.
    """

    def compile(self, optimizer, loss, **kwargs):
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
            pred_max = tf.one_hot(tf.argmax(self.pred, axis=-1), num_classes)
            self.acc = tf.reduce_mean(tf.reduce_sum(labels_one_hot*pred_max,
                [1]), name='acc')

        elif loss == 'triplet_omniglot':
            self.loss = triplet_omniglot(self.output, A=kwargs['A'],
                P=kwargs['P'], K=kwargs['K'], name='loss')
        else:
            raise ValueError('invalid loss')

        # Set training op
        self.train_op = optimizer.minimize(self.loss)

    def get_tensors(self):
        attributes = dir(self)
        tensors = {}
        for attr_name in attributes:
            attr = getattr(self, attr_name)
            if isinstance(attr, tf.Tensor):
                tensors[attr.name] = attr
        return tensors

    def get_operations(self):
        attributes = dir(self)
        operations = {}
        for attr_name in attributes:
            attr = getattr(self, attr_name)
            if isinstance(attr, tf.Operation):
                operations[attr.name] = attr
        return operations

class ModelVB(NetworkVB):
    def compile(self, optimizer, loss, **kwargs):
        self.optimizer = optimizer

        # Get variables
        self.shared_vars, self.unshared_vars = self._get_shared_unshared_vars()

        # Get losses and training operations
        self.losses = self._get_losses(loss, **kwargs)
        self.train_ops = self._get_train_ops(optimizer, self.shared_vars,
            self.unshared_vars)

        if loss == 'softmax_cross_entropy_with_logits':
            # Get accurary operations
            labels_one_hot = kwargs['labels_one_hot']
            self.train_accs, self.test_acc = self._get_acc_ops(labels_one_hot)

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
        losses = []

        for i in range(self.n_branches):
            if loss == 'softmax_cross_entropy_with_logits':
                labels = kwargs['labels_one_hot'][i]
                losses.append(softmax_cross_entropy_with_logits(labels=labels,
                    logits=self.output[i], name='loss_'+str(i+1)))
            elif loss == 'triplet_omniglot':
                losses.append(triplet_omniglot(self.output[i], A=kwargs['A'],
                    P=kwargs['P'], K=kwargs['K'], name='loss_'+str(i+1)))
            else:
                raise ValueError('invalid loss')

        return losses

    def _get_train_ops(self, optimizer, shared_vars, unshared_vars):
        # Store gradients from shared variables over each branch
        shared_grads = []
        unshared_train_ops = []

        for i in range(self.n_branches):
            loss = self.losses[i]

            # Compute gradients of shared vars for each branch (but don't apply)
            if len(shared_vars) > 0:
                shared_grads.append(optimizer.compute_gradients(loss,
                    var_list=shared_vars))

            # Apply gradients for unshared vars for each branch
            if len(unshared_vars[i]) > 0:
                unshared_train_ops.append(optimizer.minimize(loss,
                    var_list=unshared_vars[i]))

        # Take average of the gradients over each branch
        mean_shared_grads = []

        for v, var in enumerate(shared_vars):
            grad = tf.reduce_mean(
                [shared_grads[i][v][0] for i in range(self.n_branches)],[0])
            mean_shared_grads.append((grad, var))

        if len(shared_vars) > 0:
            shared_train_op = optimizer.apply_gradients(mean_shared_grads)
        else:
            shared_train_op = []

        train_ops = [unshared_train_ops, shared_train_op]

        return train_ops

    def _get_acc_ops(self, labels):
        num_classes = self.output[0].get_shape().as_list()[-1]

        # Train accuracies
        train_accs = []
        for i in range(self.n_branches):
            pred_max = tf.one_hot(tf.argmax(tf.nn.softmax(self.output[i]),
                axis=-1), num_classes)
            train_accs.append(tf.reduce_mean(tf.reduce_sum(
                labels[i]*pred_max, [1]), name='train_acc_'+str(i+1)))

        # Test accuracy
        pred = tf.nn.softmax(tf.reduce_mean(self.output.to_list(), [0]))
        pred_max = tf.one_hot(tf.argmax(pred, axis=-1), num_classes)
        test_acc = tf.reduce_mean(tf.reduce_sum(labels[0]*pred_max, [1]),
            name='test_acc')

        return train_accs, test_acc

    def get_tensors(self):
        def _get_tensors(attributes, tensors):
            for attr in attributes:
                if type(attr) is list:
                    _get_tensors(attr, tensors)
                else:
                    if isinstance(attr, tf.Tensor):
                        tensors[attr.name] = attr

        attributes = [getattr(self, name) for name in dir(self)]
        tensors = {}

        _get_tensors(attributes, tensors)
        return tensors

    def get_operations(self):
        def _get_operations(attributes, operations):
            for attr in attributes:
                if type(attr) is list:
                    _get_operations(attr, operations)
                else:
                    if isinstance(attr, tf.Operation):
                        operations[attr.name] = attr

        attributes = [getattr(self, name) for name in dir(self)]
        operations = {}

        _get_operations(attributes, operations)
        return operations
