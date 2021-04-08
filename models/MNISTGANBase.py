import os

import tensorflow as tf

from models import GANBase
from modules.tf_ops import MySaver
from modules.utils import load


class MNISTGANBase(GANBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_classifier_ops(self, x, classifier_conf=None):
        '''
        append classifier ops to calculate FID/IS and to execute Isolation Forest

        Args:
            x: tensor of the input
            classifier_conf: dict of the hyper-parameters of classifier

        Returns:
            preds (Tensor): class probabilities of x
            logits (Tensor): logits of x
            features (Tensor): features of x

        '''
        from models import models_dict

        scope = 'classifier'

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            params = classifier_conf['params']
            dataset_info = classifier_conf['dataset_info']
            Model = models_dict[classifier_conf['model_type']]
            logits, features = Model.get_logits(x, nclasses=dataset_info['nclasses'], **params, return_features=True)

            if logits is None:
                preds = None  # when vae
            else:
                preds = tf.nn.softmax(logits, axis=1)

            # restore weights
            vars = [x for x in tf.trainable_variables() if scope in x.name]
            is_initialized = self.sess.run(tf.is_variable_initialized(vars[0]))
            if not is_initialized:
                vars_trained = load(os.path.join(classifier_conf['weight_dir'], 'params_latest.pkl'))
                saver = MySaver(vars, self.sess)
                saver.restore(vars_trained)

        return preds, logits, features
