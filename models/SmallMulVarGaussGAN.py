from models import _2DGANBase
from modules.tf_ops import *


class Model(_2DGANBase):
    name = 'smallmulvargaussgan'
    dtype = tf.float64  # to avoid the numerical underflow

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def discriminator(self, x, weight_decay, use_bn, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            h_d = x
            for i in range(self.hypparams['h_d_layer']):
                h_d = self.dense(h_d, self.hypparams['h_d_dim'], tf.nn.relu, weight_decay, use_bn, name=f'd{i}',
                                 use_bias=False,
                                 kernel_initializer=tf.initializers.glorot_uniform(dtype=self.dtype),
                                 bias_initializer=tf.initializers.zeros(dtype=self.dtype))
            logits = self.dense(h_d, 1, None, weight_decay, name=f'd{i + 1}', use_bias=False,
                                kernel_initializer=tf.initializers.glorot_uniform(dtype=self.dtype),
                                bias_initializer=tf.initializers.zeros(dtype=self.dtype))
            logits = tf.squeeze(logits)

        return logits

    def generator(self, z, weight_decay, use_bn, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            h_g = z
            for i in range(self.hypparams['h_g_layer']):
                h_g = self.dense(h_g, self.hypparams['h_g_dim'], tf.nn.relu, weight_decay, use_bn,
                                 name=f'g{i}', use_bias=False,
                                 kernel_initializer=tf.initializers.glorot_uniform(dtype=self.dtype),
                                 bias_initializer=tf.initializers.zeros(dtype=self.dtype))

            x_gen = self.dense(h_g, 2, None, weight_decay,
                               name=f'g{i + 1}', use_bias=False,
                               kernel_initializer=tf.initializers.glorot_uniform(dtype=self.dtype),
                               bias_initializer=tf.initializers.zeros(dtype=self.dtype))
        return x_gen

    def get_log_likelihood(self):
        dist = create_multivariate_gaussian_distribution(self.dtype.as_numpy_dtype)
        return tf.reduce_mean(dist.log_prob(self.x_gen), name='log_likelihood')

    def get_likelihood(self):
        dist = create_multivariate_gaussian_distribution(self.dtype.as_numpy_dtype)
        return tf.reduce_mean(dist.prob(self.x_gen), name='log_likelihood')

    def get_metric(self, name, **kwargs):
        if name == 'log_likelihood':
            return self.get_log_likelihood()
        elif name == 'likelihood':
            return self.get_likelihood()
        else:
            return super().get_metric(name, **kwargs)
