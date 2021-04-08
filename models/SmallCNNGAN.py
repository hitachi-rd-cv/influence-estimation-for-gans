import tensorflow as tf

from models import MNISTGANBase


class Model(MNISTGANBase):
    name = 'smallcnngan'
    ks = [4, 3, 3, 2, 3, 3, 2]  # kernel sizes
    ss = [1, 2, 1, 1, 2, 1, 1]  # stride sizes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def discriminator(self, x, weight_decay, use_bn, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            h = x
            for i, k, s in zip(range(len(self.ks)), self.ks, self.ss):
                h = self.conv2d(h, self.hypparams['h_d_dim'], k, s, tf.nn.sigmoid, use_bn, padding='valid',
                                name='d{}'.format(i), weight_decay=weight_decay)
                print(i, h.shape)
            h = tf.squeeze(h, axis=[1, 2])
            logits = tf.squeeze(self.dense(h, 1, None, weight_decay=weight_decay), 1)

        return logits

    def generator(self, z, weight_decay, use_bn, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            h = tf.reshape(z, [-1, 1, 1, z.shape[-1]])
            for i, k, s in zip(range(len(self.ks)), self.ks[::-1], self.ss[::-1]):
                h = self.deconv2d(h, self.hypparams['h_g_dim'], k, s, tf.nn.sigmoid, use_bn, padding='valid',
                                  name='h{}'.format(i), weight_decay=weight_decay)
                print(i, h.shape)
            return tf.nn.tanh(self.conv2d(h, 1, 1, 1, None, False, padding='valid', weight_decay=weight_decay))
