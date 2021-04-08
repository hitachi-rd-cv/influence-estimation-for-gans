import tensorflow as tf

from models import Classifier


class Model(Classifier):
    name = 'cnnmnist'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, x, y, *args, **kwargs):
        self.nclasses = y._shape_as_list()[1]

        logits = self.get_logits(x, self.nclasses, self.hypparams['weight_decay'])
        # Calculate Loss (for both TRAIN and EVAL modes)

        if self.nclasses > 2:
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        else:
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y[:, 0], logits=logits)

        # regularizer
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg')
        bare_loss = tf.reduce_mean(losses)
        loss = bare_loss + reg_loss
        loss = tf.identity(loss, 'loss')

        optimizer = tf.train.AdamOptimizer(self.hypparams['lrs']['classifier']).minimize(loss)

        lr = tf.constant(self.hypparams['lrs']['classifier'])

        self.logits = logits

        return {'classifier': loss}, {'classifier': optimizer}, {'classifier': tf.trainable_variables()}, \
               {'classifier': lr}, {'classifier': reg_loss}, {'classifier': bare_loss}

    @staticmethod
    def get_logits(x, nclasses, weight_decay, return_features=False, **kwargs):
        with tf.variable_scope("Classification"):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=x,
                filters=8,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.sigmoid)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=8,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.sigmoid)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 8])
            net = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.sigmoid)

            # Logits Layers
            if nclasses > 2:
                logits = tf.layers.dense(inputs=net, units=nclasses,
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
            else:
                logits = tf.layers.dense(inputs=net, units=1,
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
                logits = tf.squeeze(logits, axis=1)

            if return_features:
                return logits, net
            else:
                return logits

    def cal_asgd_influence(self, *args, **kwargs):
        raise NotImplementedError('AdamOptimizer is forbidden')
