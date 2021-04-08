import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import gradients
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def clip_log_by_value(x, value=1e-10):
    return tf.log(tf.clip_by_value(x, value, x))


def partial_hessian_vector_product(ys, xs1, xs2, v):
    # Validate the input
    length = len(xs1)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = gradients(ys, xs1)
    assert len(grads) == length

    elemwise_products = [
        math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]

    # Second backprop
    grads_with_none = gradients(elemwise_products, xs2)
    partial_hvp = [
        grad_elem if grad_elem is not None \
            else tf.zeros_like(x) \
        for x, grad_elem in zip(xs2, grads_with_none)]

    return partial_hvp


def matmul_arrays(a, b):
    if len(b) != len(a):
        raise ValueError("xs and v must have the same length.")

    return tf.reduce_sum([tf.reduce_sum(a_ * b_) for a_, b_ in zip(a, b)])


def get_hard_labels_from_logits(logits):
    return tf.one_hot(tf.argmax(logits, axis=1), tf.shape(logits)[1], dtype=logits.dtype)


def is_tp_or_tn_op(labels, pred):
    return tf.reduce_sum(labels * pred, axis=1)


def get_acc(labels, is_tp_or_tn):
    return tf.divide(tf.reduce_sum(is_tp_or_tn), tf.cast(tf.shape(labels)[0], labels.dtype), name='acc')


def create_multivariate_gaussian_distribution(dtype):
    mu = np.array([1., 1.], dtype)
    sigma = np.array([[1., 0.8], [0.8, 1.]], dtype)
    dist = tfp.distributions.MultivariateNormalFullCovariance(mu, sigma)
    return dist


def inception_score(preds):
    return tf.exp(log_inception_score(preds), name='inception_score')


def log_inception_score(preds):
    kl = preds * (tf.log(preds) - tf.log(tf.expand_dims(tf.reduce_mean(preds, 0), 0)))
    return tf.reduce_mean(tf.reduce_sum(kl, 1), name='log_inception_score')


# Wasserstein losses from `Wasserstein GAN` (https://arxiv.org/abs/1701.07875).
def generator_loss(logits_gen, method, name):
    if method in ['wasserstein', 'hinge', 'softplus']:
        with tf.name_scope('wasserstein_loss'):
            losses = -logits_gen

    elif method == 'ls':  # least square loss
        with tf.name_scope('ls_loss'):
            losses = (logits_gen - tf.ones_like(logits_gen)) ** 2

    elif method == 'minmax':
        with tf.name_scope('minmax_loss'):
            y_gen = tf.zeros_like(logits_gen, dtype=tf.int32)
            losses = -1 * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_gen, logits=logits_gen)

    elif method == 'modified_minmax':
        with tf.name_scope('modified_minmax_loss'):
            y_gen = tf.ones_like(logits_gen)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_gen, logits=logits_gen)

    else:
        raise ValueError(method)

    loss = tf.reduce_mean(losses, name=name)

    return loss


def discriminator_loss(logits_real, logits_gen, method, name):
    if method == 'wasserstein':
        with tf.name_scope('wasserstein_loss'):
            loss_real = - logits_real
            loss_gen = logits_gen

    elif method == 'hinge':
        with tf.name_scope('hinge_loss'):
            # Compute the hinge.
            loss_real = tf.nn.relu(1.0 - logits_real)
            loss_gen = tf.nn.relu(1.0 + logits_gen)

    elif method == 'softplus':
        with tf.name_scope('softplus_loss'):
            # Compute the hinge.
            loss_real = tf.nn.softplus(1.0 - logits_real)
            loss_gen = tf.nn.softplus(1.0 + logits_gen)

    elif method == 'ls':  # least square loss
        with tf.name_scope('ls_loss'):
            loss_real = (logits_real - tf.ones_like(logits_real)) ** 2
            loss_gen = (logits_gen - tf.zeros_like(logits_gen)) ** 2

    elif method in ['minmax', 'modified_minmax']:
        with tf.name_scope('minmax_loss'):
            y_real = tf.ones_like(logits_real)
            y_gen = tf.zeros_like(logits_gen)
            loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_real, logits=logits_real)
            loss_gen = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_gen, logits=logits_gen)
    else:
        raise ValueError(method)

    loss_sum = tf.reduce_sum(loss_real) + tf.reduce_sum(loss_gen)

    # mod by length of g(z) in case length of x is len(g(z) -1 because of counterfactual train
    batch_size = tf.cast(tf.shape(logits_gen)[0], loss_sum.dtype, name='batch_size')
    loss = tf.div(loss_sum, batch_size, name=name)

    return loss


def influence_loss(logits_real, method, name):
    if method == 'wasserstein':
        with tf.name_scope('wasserstein_loss'):
            loss_real = - logits_real

    elif method == 'hinge':
        with tf.name_scope('hinge_loss'):
            loss_real = tf.nn.relu(1.0 - logits_real)

    elif method == 'softplus':
        with tf.name_scope('softplus_loss'):
            loss_real = tf.nn.softplus(1.0 - logits_real)

    elif method == 'ls':
        with tf.name_scope('ls_loss'):
            loss_real = (logits_real - tf.ones_like(logits_real)) ** 2

    elif method in ['minmax', 'modified_minmax']:
        with tf.name_scope('minmax_loss'):
            y_real = tf.ones_like(logits_real)
            loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_real, logits=logits_real)
    else:
        raise ValueError(method)

    loss = tf.reduce_sum(loss_real, name=name)

    return loss


class MySaver:
    '''
    handles model parameters

    Attributes:
        var_tensor_list: expect list of Variables of the parameters
        var_ph_list: list of Placeholders that accept the feeds which shapes are same as the elements in var_tensor_list
        assign_var_list: list of assign operations that assign the values of var_ph_list to corresponding elements in the var_tensor_list
    '''

    def __init__(self, var_tensor_list, sess):
        self.var_tensor_list = var_tensor_list
        self.var_ph_list = [tf.placeholder(w.dtype, w.get_shape(), w._shared_name) for w in var_tensor_list]
        self.assign_var_list = [tf.assign(w, w_ph) for w, w_ph in zip(var_tensor_list, self.var_ph_list)]
        self.sess = sess

    def restore(self, var_value_list):
        self.sess.run(self.assign_var_list, self.get_feed_dict(var_value_list))

    def get_current_vars(self):
        return self.sess.run(self.var_tensor_list)

    def get_feed_dict(self, var_value_list):
        return {k: v for k, v in zip(self.var_ph_list, var_value_list)}


def arrays_to_shapes(arrays):
    return [x.shape for x in arrays]


def zeros_from_arrays(arrays):
    return [np.zeros(shape) for shape in arrays_to_shapes(arrays)]


def add_arrays(a, b):
    return [x + y for x, y in zip(a, b)]


def sub_arrays(a, b):
    return [x - y for x, y in zip(a, b)]


def mul_arrays(a, b):
    return [x * y for x, y in zip(a, b)]


def mul_const_by_arrays(arrays, const):
    return [x * const for x in arrays]
