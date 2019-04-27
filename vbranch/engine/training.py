from .functional import Network, NetworkVB
from ..losses import softmax_cross_entropy_with_logits

import tensorflow as tf

class Model(Network):
    """The `Model` class adds training & evaluation routines to a `Network`.
    """

    def compile(self, optimizer, loss, labels_one_hot=None):
        self.optimizer = optimizer

        # Set loss
        assert loss in ['softmax_cross_entropy_with_logits'], 'invalid loss'

        if loss == 'softmax_cross_entropy_with_logits':
            self.loss = softmax_cross_entropy_with_logits(
                labels=labels_one_hot, logits=self.output, name='loss')

            # Set prediction
            self.pred = tf.nn.softmax(self.output, name='pred')

            # Set accuracy
            num_classes = self.pred.get_shape().as_list()[-1]
            pred_max = tf.one_hot(tf.argmax(self.pred, axis=-1), num_classes)
            self.acc = tf.reduce_mean(tf.reduce_sum(labels_one_hot*pred_max,
                [1]), name='acc')

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
    def compile(self, optimizer, loss, labels_one_hot=None):
        self.optimizer = optimizer

        # Get variables
        self.shared_vars, self.unshared_vars = self._get_shared_unshared_vars()

        # Get training ops and losses
        self.train_ops, self.losses = self._get_train_ops(labels_one_hot,
            self.output, optimizer, self.shared_vars, self.unshared_vars)

        # Get accurary ops
        self.train_acc_ops, self.test_acc_op = self._get_acc_ops(labels_one_hot,
            self.output)

    def _get_shared_unshared_vars(self):
        """
        Get shared variables (in order to later average gradients)
        and unshared variables (unique to each branch)"""

        shared_vars = []
        unshared_vars = [[] for i in range(num_branches)]

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

    def _get_train_ops(self,labels,logits,optimizer,shared_vars,unshared_vars):
        losses = []
        # Store gradients from shared variables over each branch
        shared_grads = []
        unshared_train_ops = []

        for i in range(self.n_branches):
            loss = softmax_cross_entropy_with_logits(labels=labels[i],
                logits=logits[i], name='loss_'+str(i+1))
            losses.append(loss)

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
        return train_ops, losses

    def _get_acc_ops(self, labels, logits):
        num_classes = self.output[0].get_shape().as_list()[-1]

        # Train accuracies
        train_acc_ops = []
        for i in range(self.n_branches):
            pred_max = tf.one_hot(tf.argmax(tf.nn.softmax(logits[i]), axis=-1),
                                  num_classes)
            train_acc_ops.append(tf.reduce_mean(tf.reduce_sum(
                labels[i]*pred_max, [1]), name='train_acc_'+str(i+1)))

        # Test accuracy
        pred = tf.nn.softmax(tf.reduce_mean(logits, [0]))
        pred_max = tf.one_hot(tf.argmax(pred, axis=-1), num_classes)
        test_acc_op = tf.reduce_mean(tf.reduce_sum(labels[0]*pred_max, [1]),
            name='test_acc')

        return train_acc_ops, test_acc_op
